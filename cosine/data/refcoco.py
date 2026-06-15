import os
import pickle
import copy
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.data import detection_utils as utils
from detectron2.structures import BitMasks, Boxes, Instances, BoxMode
from detectron2.data import transforms as T

from cosine.data.grefer import G_REFER
from cosine.data.refer import REFER
from cosine.data import detection_utils as utils

from pycocotools import mask as coco_mask


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class RefCOCODataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_size,
        sam_image_size,
        clip_image_size,
        root='datasets',
        dataset_name='refcoco',
        is_train=True,
        visual_prompt=False,
        text_prompt=True,
        transform=None,
        dino_transform=None,
        ignore_label=255,
        img_format='RGB',
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        num_classes_per_sample=10000,
        sam_pixel_mean=[123.675, 116.28, 103.53],
        sam_pixel_std=[58.395, 57.12, 57.375],
        clip_pixel_mean=[122.7709383, 116.7460125, 104.09373615],
        clip_pixel_std=[68.5005327, 66.6321579, 70.32316305],
    ):

        self.is_train = is_train
        assert is_train, "RefCOCODataset only used in training"

        self.root_ = root
        self.dataset_name = dataset_name

        self.ignore_label = ignore_label
        self.transform = transform
        self.dino_transform = dino_transform

        self.visual_prompt = visual_prompt
        self.text_prompt = text_prompt

        DATA_DIR = os.path.join(root, "refer_seg")
        self.refer_seg_ds_list = refer_seg_data.split(
            "||"
        )  # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
            ref_ids_train = refer_api.getRefIds(split="train")
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/mscoco/images/train2014", item["file_name"]
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_train

            print(
                "dataset {} (refs {}) (train split) has {} images and {} annotations.".format(
                    ds,
                    splitBy,
                    len(refer_seg_ds["images"]),
                    len(refer_seg_ds["annotations"]),
                )
            )

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_data[ds] = refer_seg_ds

        self.num_classes_per_sample = num_classes_per_sample

        self.length = 0
        for ds, refer_seg_ds in self.refer_seg_data.items():
            self.length += len(refer_seg_ds["images"])

        self.sam_pixel_mean = torch.Tensor(sam_pixel_mean).view(-1, 1, 1)
        self.sam_pixel_std = torch.Tensor(sam_pixel_std).view(-1, 1, 1)
        self.clip_pixel_mean = torch.Tensor(clip_pixel_mean).view(-1, 1, 1)
        self.clip_pixel_std = torch.Tensor(clip_pixel_std).view(-1, 1, 1)

        self.img_format = img_format

        self.image_size = image_size
        self.sam_image_size = sam_image_size
        self.clip_image_size = clip_image_size

    def __len__(self):
        return self.length

    def pad_img(self, x, pad_size):

        assert isinstance(x, torch.Tensor)
        # Pad
        h, w = x.shape[-2:]
        padh = pad_size - h
        padw = pad_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def process_img_dict(self, dataset_dict, keep_cat_ids=None):

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        while True:
            try:
                image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            except OSError as e:
                print(f"Catched exception: {str(e)}. Re-trying...")
                import time
                time.sleep(3)
            else:
                break
        try:
            utils.check_image_size(dataset_dict, image)
        except ValueError as e:
            image = image.transpose(1, 0, 2)

        tfm_gens = self.transform

        image, transforms = T.apply_transform_gens(tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        if self.dino_transform is None:
            dino_image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        else:
            dino_image = self.dino_transform(image)
        dataset_dict["image"] = self.pad_img(dino_image, self.image_size)

        # SAM image input
        sam_image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        vaild = torch.ones_like(sam_image)
        sam_image_pad = self.pad_img(sam_image, self.image_size)
        vaild_pad = self.pad_img(vaild, self.image_size)
        sam_image_pad = F.interpolate(
            sam_image_pad[None, ...], self.sam_image_size, mode="bilinear", align_corners=False, antialias=True
        )[0]
        vaild_pad = F.interpolate(
            vaild_pad[None, ...], self.sam_image_size, mode='nearest'
        )[0]
        sam_image_pad = (sam_image_pad - self.sam_pixel_mean) / self.sam_pixel_std
        dataset_dict["sam_image"] = sam_image_pad * vaild_pad

        # CLIP image input
        clip_image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        vaild = torch.ones_like(clip_image)
        clip_image_pad = self.pad_img(clip_image, self.image_size)
        vaild_pad = self.pad_img(vaild, self.image_size)
        clip_image_pad = F.interpolate(
            clip_image_pad[None, ...], self.clip_image_size, mode="bilinear", align_corners=False, antialias=True
        )[0]
        vaild_pad = F.interpolate(
            vaild_pad[None, ...], self.clip_image_size, mode='nearest'
        )[0]
        clip_image_pad = (clip_image_pad - self.clip_pixel_mean) / self.clip_pixel_std
        dataset_dict["clip_image"] = clip_image_pad * vaild_pad

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict


        if "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            text_prompts = dataset_dict.pop("text_prompts")

            for anno in annotations:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in annotations if obj.get("iscrowd", 0) == 0 and obj['id'] in keep_cat_ids
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            if isinstance(annotations[0]['segmentation'], np.ndarray): # annos may be empty
                instances = utils.annotations_to_instances(annos, image_shape, mask_format='bitmask')
            else:
                instances = utils.annotations_to_instances(annos, image_shape, mask_format='polygon')
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            try:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            except:
                dataset_dict["instances"] = instances
                return dataset_dict

            ins_ids = [anno['id'] for anno in annos]
            new_ins_ids = []
            if len(ins_ids) > 0:
                for ins_id in ins_ids:
                    if isinstance(ins_id, str):
                        ins_id = int(ins_id.replace('_', '0'))
                    new_ins_ids.append(ins_id)
            ins_ids = new_ins_ids

            ins_ids = np.array(ins_ids)
            instances.ins_ids = torch.tensor(ins_ids, dtype=torch.int64)
            assert len(instances) == len(text_prompts)
            instances.cap_ids = torch.tensor(np.array(list(range(len(text_prompts)))), dtype=torch.int64)
            # Need to filter empty instances first (due to augmentation)
            # instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                if isinstance(gt_masks, BitMasks):
                    gt_masks = gt_masks.tensor
                else:
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                gt_masks = self.pad_img(gt_masks, self.image_size)
                if len(gt_masks) == 0:
                    gt_masks = torch.zeros((len(text_prompts), self.image_size, self.image_size))
                instances.gt_masks = gt_masks

            cap_ids = list(instances.get('cap_ids').numpy())
            instances.remove('cap_ids')
            dataset_dict["instances"] = instances
            dataset_dict["text_prompts"] = [text_prompts[cap_id] for cap_id in cap_ids]

        return dataset_dict

    def get_ref_tar_dict(self,):

        ds = random.randint(0, len(self.refer_seg_ds_list) - 1)
        ds = self.refer_seg_ds_list[ds]
        refer_seg_ds = self.refer_seg_data[ds]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.get_ref_tar_dict()

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])

        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))

        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_classes = sampled_sents
        annotations = [annotations[ann_id] for ann_id in sampled_ann_ids]
        for anno in annotations:
            anno['bbox_mode'] = BoxMode.XYWH_ABS

        text_prompts = []
        for sampled_ann_id, sampled_class in zip(sampled_ann_ids, sampled_classes):
            text_prompts.append(
                {
                    'id': sampled_ann_id if not isinstance(sampled_ann_id, str) else int(sampled_ann_id.replace('_', '0')),
                    'caption': sampled_class
                }
            )

        image_info['annotations'] = annotations
        image_info['text_prompts'] = text_prompts

        return image_info


    def __getitem__(self, idx):

        image_info = self.get_ref_tar_dict()
        keep_cat_ids = list(set([anno['id'] for anno in image_info['annotations']]))
        target = self.process_img_dict(image_info, keep_cat_ids=keep_cat_ids)

        if self.text_prompt and not self.visual_prompt:
            target = target
            visual_prompt = None
            text_prompt = target['text_prompts']
        else:
            raise NotImplementedError

        return {"target": target, "visual_prompt": visual_prompt, "text_prompt": text_prompt, "tag": "refer"}


if __name__ == '__main__':

    import argparse
    from detectron2.data import transforms as T
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from dinov2.data.transforms import MaybeToTensor, make_normalize_transform

    parser = argparse.ArgumentParser('coco dataset', add_help=False)
    parser.add_argument('--random_flip', default="horizontal", type=str)
    parser.add_argument('--min_size', default=(560, 588, 616, 644, 672, 700), type=float)
    parser.add_argument('--max_size', default=896, type=float) # no aug
    parser.add_argument('--image_size', default=896, type=int)
    parser.add_argument('--sam_image_size', default=1024, type=int)
    parser.add_argument('--clip_image_size', default=1024, type=int)
    args = parser.parse_args()

    # LSJ aug
    augmentation = [T.ResizeShortestEdge(args.min_size, args.max_size, "choice")]


    dino_transform = transforms.Compose([
            MaybeToTensor(),
            make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    dino_pixel_mean = [i*255 for i in [0.485, 0.456, 0.406]]
    dino_pixel_std = [i*255 for i in [0.229, 0.224, 0.225]]
    sam_pixel_mean = [123.675, 116.28, 103.53]
    sam_pixel_std = [58.395, 57.12, 57.375]
    clip_pixel_mean = [122.7709383, 116.7460125, 104.09373615],
    clip_pixel_std = [68.5005327, 66.6321579, 70.32316305],

    dataset = RefCOCODataset(
        image_size=args.image_size,
        sam_image_size=args.sam_image_size,
        clip_image_size=args.clip_image_size,
        transform=augmentation,
        dino_transform=dino_transform,
        refer_seg_data="refclef",
        sam_pixel_mean=sam_pixel_mean,
        sam_pixel_std=sam_pixel_std,
        clip_pixel_mean=clip_pixel_mean,
        clip_pixel_std=clip_pixel_std
    )

    show_size = (224, 224)

    for id in range(len(dataset)):

        print(id)

        # if id < 260:
        #     continue

        target, visual_prompt, text_prompt, tag = list(dataset[id].values())
        # ref_dino_img = (visual_prompt['image'] * torch.Tensor(dino_pixel_std).view(-1, 1, 1)) + torch.Tensor(
        #     dino_pixel_mean).view(-1, 1, 1)

        # tar_dino_img = (target['image'] * torch.Tensor(dino_pixel_std).view(-1, 1, 1)) + torch.Tensor(
        #     dino_pixel_mean).view(-1, 1, 1)
        # tar_sam_img = (target['sam_image'] * torch.Tensor(sam_pixel_std).view(-1, 1, 1)) + torch.Tensor(
        #     sam_pixel_mean).view(-1, 1, 1)
        # tar_clip_img = (target['clip_image'] * torch.Tensor(clip_pixel_std).view(-1, 1, 1)) + torch.Tensor(
        #     clip_pixel_mean).view(-1, 1, 1)
        #
        #
        # tar_dino_img = F.interpolate(
        #     tar_dino_img[None, ...], show_size, mode="bilinear", align_corners=False, antialias=True
        # )[0]
        # tar_sam_img = F.interpolate(
        #     tar_sam_img[None, ...], show_size, mode="bilinear", align_corners=False, antialias=True
        # )[0]
        # tar_clip_img = F.interpolate(
        #     tar_clip_img[None, ...], show_size, mode="bilinear", align_corners=False, antialias=True
        # )[0]
        #
        # tar_masks = target['instances'].gt_masks
        # tar_masks = F.interpolate(
        #     tar_masks[None, ...].float(), show_size
        # )[0] > 0
        #
        # tar_dino_img_np = tar_dino_img.permute(1, 2, 0).numpy()
        # tar_sam_img_np = tar_sam_img.permute(1, 2, 0).numpy()
        # tar_clip_img_np = tar_clip_img.permute(1, 2, 0).numpy()
        #
        # show_imgs = torch.cat([ tar_dino_img, tar_sam_img, tar_clip_img], dim=-1).permute(1,2,0).numpy() /255.
        #
        # tar_dino_mask = []
        # for mask in tar_masks:
        #     color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        #     mask = mask.numpy()
        #     h, w = mask.shape[-2:]
        #     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        #     tar_dino_mask.append(mask_image)
        # # tar_dino_mask = sum(tar_dino_mask)
        # tar_caption = [item['caption'] for item in text_prompt]
        # tar_ins_id = [item['id'] for item in text_prompt]
        # for mask_i, cap_i, id_i in zip(tar_dino_mask, tar_caption, tar_ins_id):
        #     show_masks = np.concatenate([mask_i, mask_i, mask_i], axis=1)
        #
        #     if not os.path.exists(f'shows_ref/coco'):
        #         os.makedirs(f'shows_ref/coco')
        #
        #     cap_i = cap_i.replace('/', ' ')
        #     save_path = f'shows_ref/coco/img{id}_{id_i}_{cap_i}.jpg'
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(show_imgs)
        #     ax = plt.gca()
        #     ax.imshow(show_masks)
        #     plt.axis('off')
        #     plt.savefig(save_path)
