import glob
import os
import random
import copy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask

from torchvision import transforms
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform
from segment_anything.utils.transforms import ResizeLongestSide
from detectron2.structures import BitMasks, Instances

# from transformers import CLIPImageProcessor

# from model.llava import conversation as conversation_lib
# from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
#                                    IMAGE_TOKEN_INDEX)
# from model.llava.mm_utils import tokenizer_image_token
# # from model.segment_anything.utils.transforms import ResizeLongestSide

from inference_referseg.data.refer import REFER
# from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
#                     DEFAULT_IMAGE_TOKEN)



# def collate_fn(
#     batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
# ):
#     image_path_list = []
#     images_list = []
#     images_clip_list = []
#     conversation_list = []
#     masks_list = []
#     label_list = []
#     resize_list = []
#     questions_list = []
#     sampled_classes_list = []
#     offset_list = [0]
#     cnt = 0
#     inferences = []
#     for (
#         image_path,
#         images,
#         images_clip,
#         conversations,
#         masks,
#         label,
#         resize,
#         questions,
#         sampled_classes,
#         inference,
#     ) in batch:
#         image_path_list.append(image_path)
#         images_list.append(images)
#         images_clip_list.append(images_clip)
#         conversation_list.extend(conversations)
#         label_list.append(label)
#         masks_list.append(masks.float())
#         resize_list.append(resize)
#         questions_list.append(questions)
#         sampled_classes_list.append(sampled_classes)
#         cnt += len(conversations)
#         offset_list.append(cnt)
#         inferences.append(inference)
#
#     if use_mm_start_end:
#         # replace <image> token
#         for i in range(len(conversation_list)):
#             replace_token = DEFAULT_IMAGE_TOKEN
#             replace_token = (
#                 DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
#             )
#             conversation_list[i] = conversation_list[i].replace(
#                 DEFAULT_IMAGE_TOKEN, replace_token
#             )
#     input_ids = [
#         tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
#         for prompt in conversation_list
#     ]
#     input_ids = torch.nn.utils.rnn.pad_sequence(
#         input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
#     )
#     attention_masks = input_ids.ne(tokenizer.pad_token_id)
#
#     conv = conversation_lib.default_conversation.copy()
#     targets = input_ids.clone()
#
#     if conv_type == "llava_v1":
#         sep = conv.sep + conv.roles[1] + ": "
#     else:
#         sep = "[/INST] "
#     for conversation, target in zip(conversation_list, targets):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())
#
#         rounds = conversation.split(conv.sep2)
#         cur_len = 1
#         target[:cur_len] = IGNORE_INDEX
#         for i, rou in enumerate(rounds):
#             if rou == "":
#                 break
#
#             parts = rou.split(sep)
#             # if len(parts) != 2:
#             #     break
#             assert len(parts) == 2, (len(parts), rou)
#             parts[0] += sep
#
#             if DEFAULT_IMAGE_TOKEN in conversation:
#                 round_len = len(tokenizer_image_token(rou, tokenizer))
#                 instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
#             else:
#                 round_len = len(tokenizer(rou).input_ids)
#                 instruction_len = len(tokenizer(parts[0]).input_ids) - 2
#
#             target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
#
#             cur_len += round_len
#         target[cur_len:] = IGNORE_INDEX
#
#         if False:
#             z = target.clone()
#             z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
#             if local_rank == 0:
#                 print(
#                     "conversation: ",
#                     conversation,
#                     "tokenizer.decode(z): ",
#                     tokenizer.decode(z),
#                 )
#
#         if cur_len < tokenizer.model_max_length:
#             assert cur_len == total_len
#
#     if inferences[0] == False:
#         truncate_len = tokenizer.model_max_length - 255
#
#         if input_ids.shape[1] > truncate_len:
#             input_ids = input_ids[:, :truncate_len]
#             targets = targets[:, :truncate_len]
#             attention_masks = attention_masks[:, :truncate_len]
#
#     return {
#         "image_paths": image_path_list,
#         "images": torch.stack(images_list, dim=0),
#         "images_clip": torch.stack(images_clip_list, dim=0),
#         "input_ids": input_ids,
#         "labels": targets,
#         "attention_masks": attention_masks,
#         "masks_list": masks_list,
#         "label_list": label_list,
#         "resize_list": resize_list,
#         "offset": torch.LongTensor(offset_list),
#         "questions_list": questions_list,
#         "sampled_classes_list": sampled_classes_list,
#         "inference": inferences[0],
#         "conversation_list": conversation_list,
#     }


class ValDataset(torch.utils.data.Dataset):

    sam_pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    sam_pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    clip_pixel_mean = torch.Tensor([122.7709383, 116.7460125, 104.09373615]).view(-1, 1, 1)
    clip_pixel_std = torch.Tensor([68.5005327, 66.6321579, 70.32316305]).view(-1, 1, 1)
    dino_transform = transforms.Compose([
        MaybeToTensor(),
        make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


    def __init__(
        self,
        base_image_dir,
        val_dataset,
        image_size=896,
        sam_image_size=1024,
        clip_image_size=1024,
        ignore_label=255
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")

        assert len(splits) == 3
        ds, splitBy, split = splits
        refer_api = REFER(os.path.join(self.base_image_dir, "refer_seg"), ds, splitBy)
        ref_ids_val = refer_api.getRefIds(split=split)
        images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
        refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
        refer_seg_ds = {}
        refer_seg_ds["images"] = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
        for item in loaded_images:
            item = item.copy()
            if ds == "refclef":
                item["file_name"] = os.path.join(
                    base_image_dir, "refer_seg", "images/saiapr_tc-12", item["file_name"]
                )
            elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                item["file_name"] = os.path.join(
                    base_image_dir, "refer_seg",
                    "images/mscoco/images/train2014",
                    item["file_name"],
                )
            refer_seg_ds["images"].append(item)
        refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

        img2refs = {}
        for ref in refs_val:
            image_id = ref["image_id"]
            img2refs[image_id] = img2refs.get(image_id, []) + [
                ref,
            ]
        refer_seg_ds["img2refs"] = img2refs
        self.refer_seg_ds = refer_seg_ds
        self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.sam_image_size = sam_image_size
        self.clip_image_size = clip_image_size
        self.transform = ResizeLongestSide(image_size)

        self.ignore_label = ignore_label

        # self.tokenizer = tokenizer
        # self.transform = ResizeLongestSide(image_size)
        # self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def pad_img(self, x, pad_size):

        assert isinstance(x, torch.Tensor)
        # Pad
        h, w = x.shape[-2:]
        padh = pad_size - h
        padw = pad_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):

        refer_seg_ds = self.refer_seg_ds
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]

        dataset_dict = images[idx]

        dataset_dict = copy.deepcopy(dataset_dict)

        image_path = dataset_dict["file_name"]
        image_id = dataset_dict["id"]

        # image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize image
        image = self.transform.apply_image(image)
        image_shape = image.shape[:2]  # h, w

        # preprocess image for dino
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

        refs = img2refs[image_id]
        if len(refs) == 0:
            raise ValueError("image {} has no refs".format(image_id))

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                sents.append(sent["sent"].strip().lower())
                ann_ids.append(ref["ann_id"])

        sampled_sents = sents
        sampled_ann_ids = ann_ids

        text_prompts = []
        for sampled_ann_id, sampled_sent in zip(sampled_ann_ids, sampled_sents):
            text_prompts.append(
                {
                    'id': sampled_ann_id if not isinstance(sampled_ann_id, str) else int(sampled_ann_id.replace('_', '0')),
                    'caption': sampled_sent
                }
            )

        masks = []
        for i, ann_id in enumerate(sampled_ann_ids):
            ann = annotations[ann_id]
            if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                m = np.zeros((dataset_dict["height"], dataset_dict["width"], 1))
            else:
                if type(ann["segmentation"][0]) == list:  # polygon
                    rle = mask.frPyObjects(
                        ann["segmentation"],
                        dataset_dict["height"],
                        dataset_dict["width"],
                    )
                else:
                    rle = ann["segmentation"]
                    for i in range(len(rle)):
                        if not isinstance(rle[i]["counts"], bytes):
                            rle[i]["counts"] = rle[i]["counts"].encode()
                m = mask.decode(rle)
            m = np.sum(
                m, axis=2
            )  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        dataset_dict['resize'] = image_shape
        dataset_dict['gt_masks'] = masks
        dataset_dict['text_prompts'] = text_prompts
        dataset_dict['labels'] = labels

        return {"target": dataset_dict, "visual_prompt": None, "text_prompt": text_prompts, "tag": "refer"}


if __name__ == '__main__':

    dataset = ValDataset(
        base_image_dir='datasets',
        val_dataset='refcoco|unc|val',
    )

    for id in range(len(dataset)):

        data = dataset[id]
        print(id)