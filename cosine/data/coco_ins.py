import os
import pickle
import copy
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.data import transforms as T
from detectron2.utils.file_io import PathManager

import sys
sys.path.append('./')

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

class COCOInsDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_size,
        sam_image_size,
        clip_image_size,
        root='datasets',
        dataset_name='coco_2017_train_ins',
        is_train=True,
        visual_prompt=True,
        text_prompt=False,
        use_all_classes=False,
        crop_ratio=1.0,
        tfm_gens_crop_pair=None,
        tfm_gens_sel_pair=None,
        dino_transform=None,
        img_format='RGB',
        serialize=True,
        sam_pixel_mean=[123.675, 116.28, 103.53],
        sam_pixel_std=[58.395, 57.12, 57.375],
        clip_pixel_mean=[122.7709383, 116.7460125, 104.09373615],
        clip_pixel_std=[68.5005327, 66.6321579, 70.32316305],
    ):

        self.is_train = is_train
        assert is_train, "COCOInsDataset only used in training"
        self._serialize = serialize

        self.root_ = root
        self.dataset_name = dataset_name
        coco_ins_meta = MetadataCatalog.get(dataset_name)
        coco_ins_data = {item['file_name']: item for item in DatasetCatalog.get(dataset_name)}

        self.catid2img = self.load_catid2img(coco_ins_data)
        self.class_ids = list(self.catid2img.keys())
        self.class_names = coco_ins_meta.thing_classes

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            coco_ins_data = {k: _serialize(v) for k, v in coco_ins_data.items()}

        self.coco_ins_data = coco_ins_data

        self.ignore_label = coco_ins_meta.get('ignore_label')

        self.crop_ratio = crop_ratio
        self.tfm_gens_crop_pair = tfm_gens_crop_pair
        self.tfm_gens_sel_pair = tfm_gens_sel_pair
        self.dino_transform = dino_transform

        self.visual_prompt = visual_prompt
        self.text_prompt = text_prompt
        self.use_all_classes = use_all_classes

        self.sam_pixel_mean = torch.Tensor(sam_pixel_mean).view(-1, 1, 1)
        self.sam_pixel_std = torch.Tensor(sam_pixel_std).view(-1, 1, 1)
        self.clip_pixel_mean = torch.Tensor(clip_pixel_mean).view(-1, 1, 1)
        self.clip_pixel_std = torch.Tensor(clip_pixel_std).view(-1, 1, 1)

        self.img_format = img_format
        self.image_size = image_size
        self.sam_image_size = sam_image_size
        self.clip_image_size = clip_image_size

    def set_multimodal(self, visual_prompt=True, text_prompt=False):
        self.visual_prompt = visual_prompt
        self.text_prompt = text_prompt

    def load_catid2img(self, coco_ins_data):
        if not os.path.exists(os.path.join(self.root_, 'cosine_pkls', f'{self.dataset_name}_catid2img.pkl')):
            if not os.path.exists(os.path.join(self.root_, 'cosine_pkls')) :os.makedirs(os.path.join(self.root_, 'cosine_pkls'))
            catid2img = {}
            for item in tqdm(coco_ins_data.values()):
                for anno in item['annotations']:
                    if anno['category_id'] not in catid2img:
                        catid2img[anno['category_id']] = []
                    if item['file_name'] not in catid2img[anno['category_id']]:
                        catid2img[anno['category_id']].append(item['file_name'])
            with open(os.path.join(self.root_, 'cosine_pkls', f'{self.dataset_name}_catid2img.pkl'), 'wb') as file:
                pickle.dump(catid2img, file)
        else:
            with open(os.path.join(self.root_, 'cosine_pkls', f'{self.dataset_name}_catid2img.pkl'), 'rb') as file:
                catid2img = pickle.load(file)
        return catid2img


    def __len__(self):
        return len(self.coco_ins_data)

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def get_ref_tar_dict(self, class_sample, crop_pair_flag=True):

        def load_dict(file):
            dict = self.coco_ins_data[file]
            if self._serialize:
                dict = memoryview(dict)
                dict = pickle.loads(dict)
            return dict

        if len(self.catid2img[class_sample]) < 2: crop_pair_flag = True
        ref_file = np.random.choice(self.catid2img[class_sample], 1, replace=False)[0]
        ref_dict = load_dict(ref_file)
        if crop_pair_flag:
            tar_dict = copy.deepcopy(ref_dict)
        else:
            while True:
                tar_file = np.random.choice(self.catid2img[class_sample], 1, replace=False)[0]
                if ref_file != tar_file: break
            tar_dict = load_dict(tar_file)

        return ref_dict, tar_dict

    def pad_img(self, x, pad_size):

        assert isinstance(x, torch.Tensor)
        # Pad
        h, w = x.shape[-2:]
        padh = pad_size - h
        padw = pad_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def process_img_dict(self, dataset_dict, crop_pair_flag, keep_cat_ids=None):

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

        utils.check_image_size(dataset_dict, image)

        if crop_pair_flag:
            tfm_gens = self.tfm_gens_crop_pair
        else:
            tfm_gens = self.tfm_gens_sel_pair

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


        if 'segmentation_file' in dataset_dict:
            with PathManager.open(dataset_dict['segmentation_file'], "rb") as f:
                segm_info = json.load(f)

            assert segm_info["image_id"] == dataset_dict["image_id"]
            for anno in dataset_dict['annotations']:
                anno_id = anno["id"]
                segm = segm_info["segmentations"][str(anno_id)]
                anno["segmentation"] = coco_mask.frPyObjects(segm, *segm["size"])


        if "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            # sort
            cat_ids = [info['category_id'] for info in annotations]
            annotations = [annotations[idx] for idx in sorted(range(len(cat_ids)), key=lambda k: cat_ids[k])]

            for anno in annotations:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in annotations if obj.get("iscrowd", 0) == 0 and obj['category_id'] in keep_cat_ids
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
            ins_ids = np.array(ins_ids)
            instances.ins_ids = torch.tensor(ins_ids, dtype=torch.int64)
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                gt_masks = self.pad_img(gt_masks, self.image_size)
                instances.gt_masks = gt_masks

            dataset_dict["instances"] = instances

        return dataset_dict

    def __getitem__(self, idx):

        ref_mask_num = 0
        tar_mask_num = 0

        while ref_mask_num == 0 or tar_mask_num == 0:

            # sample category
            class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]

            #sample reference and target dataset dict
            crop_pair_flag = self._rand_range() < self.crop_ratio
            ref_dict, tar_dict = self.get_ref_tar_dict(class_sample, crop_pair_flag)
            ref_cat_ids = list(set([anno['category_id'] for anno in ref_dict['annotations']]))
            ref_dict = self.process_img_dict(ref_dict, crop_pair_flag, keep_cat_ids=ref_cat_ids)
            ref_cat_ids = list(set(ref_dict['instances'].gt_classes.tolist()))
            tar_dict = self.process_img_dict(tar_dict, crop_pair_flag, keep_cat_ids=ref_cat_ids)

            ref_mask_num = len(ref_dict['instances'])
            tar_mask_num = len(tar_dict['instances'])

        if self.visual_prompt and not self.text_prompt:
            target = tar_dict
            visual_prompt = ref_dict
            text_prompt = None
            tag = 'visual'
        elif self.text_prompt and not self.visual_prompt:
            target = ref_dict
            visual_prompt = None
            if self.use_all_classes:
                text_prompt = {cat_id: cat for cat_id, cat in enumerate(self.class_names)}
            else:
                text_prompt = {cat_id: self.class_names[cat_id] for cat_id in ref_cat_ids}
            tag = 'sem'
        elif self.visual_prompt and self.text_prompt:
            target = tar_dict
            visual_prompt = ref_dict
            visual_gt_classes = set(list(visual_prompt['instances'].gt_classes.numpy()))
            # text_prompt = {cat_id: self.class_names[cat_id] for cat_id in ref_cat_ids}
            text_prompt = {cat_id: self.class_names[cat_id] for cat_id in visual_gt_classes}
            tag = 'visual'
        else:
            raise NotImplementedError

        return {"target": target, "visual_prompt": visual_prompt, "text_prompt": text_prompt, "tag": tag}


if __name__ == '__main__':

    import argparse
    from detectron2.data import transforms as T
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import cosine.data
    from dinov2.data.transforms import MaybeToTensor, make_normalize_transform

    parser = argparse.ArgumentParser('coco dataset', add_help=False)
    parser.add_argument('--random_flip', default="horizontal", type=str)
    parser.add_argument('--min_scale', default=0.1, type=float)
    parser.add_argument('--max_scale', default=2.0, type=float)
    parser.add_argument('--image_size', default=896, type=int)
    parser.add_argument('--sam_image_size', default=1024, type=int)
    parser.add_argument('--clip_image_size', default=1024, type=int)
    args = parser.parse_args()

    # LSJ aug
    augmentation = []
    if args.random_flip != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=args.random_flip == "horizontal",
                vertical=args.random_flip == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=args.min_scale, max_scale=args.max_scale, target_height=args.image_size, target_width=args.image_size
        ),
        T.FixedSizeCrop(crop_size=(args.image_size, args.image_size), pad=False),
    ])

    dino_transform = transforms.Compose([
            MaybeToTensor(),
            make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    dino_pixel_mean = [i*255 for i in [0.485, 0.456, 0.406]]
    dino_pixel_std = [i*255 for i in [0.229, 0.224, 0.225]]

    # sam_transform = ResizeLongestSide(args.sam_image_size)
    sam_pixel_mean = [123.675, 116.28, 103.53]
    sam_pixel_std = [58.395, 57.12, 57.375]
    clip_pixel_mean = [122.7709383, 116.7460125, 104.09373615],
    clip_pixel_std = [68.5005327, 66.6321579, 70.32316305],

    dataset = COCOInsDataset(
        dataset_name="coco_2017_val_ins",
        image_size=args.image_size,
        sam_image_size=args.sam_image_size,
        clip_image_size=args.clip_image_size,
        crop_ratio=0.5,
        visual_prompt=False,
        text_prompt=True,
        use_all_classes=False,
        tfm_gens_crop_pair=augmentation,
        tfm_gens_sel_pair=augmentation,
        dino_transform=dino_transform,
        sam_pixel_mean=sam_pixel_mean,
        sam_pixel_std=sam_pixel_std,
        clip_pixel_mean=clip_pixel_mean,
        clip_pixel_std=clip_pixel_std
    )

    show_size = (224, 224)

    for id in range(len(dataset)):

        if id > 20:
            break

        target, visual_prompt, text_prompt, tag = list(dataset[id].values())
        ref_dino_img = (visual_prompt['image'] * torch.Tensor(dino_pixel_std).view(-1, 1, 1)) + torch.Tensor(
            dino_pixel_mean).view(-1, 1, 1)
        tar_dino_img = (target['image'] * torch.Tensor(dino_pixel_std).view(-1, 1, 1)) + torch.Tensor(
            dino_pixel_mean).view(-1, 1, 1)
        tar_sam_img = (target['sam_image'] * torch.Tensor(sam_pixel_std).view(-1, 1, 1)) + torch.Tensor(
            sam_pixel_mean).view(-1, 1, 1)
        tar_clip_img = (target['clip_image'] * torch.Tensor(clip_pixel_std).view(-1, 1, 1)) + torch.Tensor(
            clip_pixel_mean).view(-1, 1, 1)

        ref_dino_img = F.interpolate(
            ref_dino_img[None, ...], show_size, mode="bilinear", align_corners=False, antialias=True
        )[0]
        tar_dino_img = F.interpolate(
            tar_dino_img[None, ...], show_size, mode="bilinear", align_corners=False, antialias=True
        )[0]
        tar_sam_img = F.interpolate(
            tar_sam_img[None, ...], show_size, mode="bilinear", align_corners=False, antialias=True
        )[0]
        tar_clip_img = F.interpolate(
            tar_clip_img[None, ...], show_size, mode="bilinear", align_corners=False, antialias=True
        )[0]

        ref_masks = visual_prompt['instances'].gt_masks
        tar_masks = target['instances'].gt_masks
        ref_masks = F.interpolate(
            ref_masks[None, ...].float(), show_size
        )[0] > 0
        tar_masks = F.interpolate(
            tar_masks[None, ...].float(), show_size
        )[0] > 0

        ref_dino_img_np = ref_dino_img.permute(1,2,0).numpy()
        tar_dino_img_np = tar_dino_img.permute(1, 2, 0).numpy()
        tar_sam_img_np = tar_sam_img.permute(1, 2, 0).numpy()
        tar_clip_img_np = tar_clip_img.permute(1, 2, 0).numpy()

        show_imgs = torch.cat([ref_dino_img, tar_dino_img, tar_sam_img, tar_clip_img], dim=-1).permute(1,2,0).numpy() /255.

        ref_dino_mask = []
        for mask in ref_masks:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            mask = mask.numpy()
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ref_dino_mask.append(mask_image)
        ref_dino_mask = sum(ref_dino_mask)

        tar_dino_mask = []
        for mask in tar_masks:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            mask = mask.numpy()
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            tar_dino_mask.append(mask_image)
        tar_dino_mask = sum(tar_dino_mask)

        show_masks = np.concatenate([ref_dino_mask, tar_dino_mask, tar_dino_mask, tar_dino_mask], axis=1)

        if not os.path.exists(f'shows_ins/coco'):
            os.makedirs(f'shows_ins/coco')

        save_path = f'shows_ins/coco/img{id}.jpg'
        plt.figure(figsize=(10, 10))
        plt.imshow(show_imgs)
        ax = plt.gca()
        ax.imshow(show_masks)
        plt.axis('off')
        plt.savefig(save_path)
