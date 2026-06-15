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


clip_pixel_mean = torch.Tensor([122.7709383, 116.7460125, 104.09373615]).view(-1, 1, 1)
clip_pixel_std = torch.Tensor([68.5005327, 66.6321579, 70.32316305]).view(-1, 1, 1)
dino_transform = transforms.Compose([
    MaybeToTensor(),
    make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
image_size=896
clip_image_size=1024

def pad_img(x, pad_size):

    assert isinstance(x, torch.Tensor)
    # Pad
    h, w = x.shape[-2:]
    padh = pad_size - h
    padw = pad_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def inf_read_data(image_dir):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    transform = ResizeLongestSide(image_size)
    dataset_dicts = []

    for image_path in image_paths:
        dataset_dict = {}
        # image
        assert os.path.exists(image_path), image_path
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dataset_dict["file_name"] = image_path.split("/")[-1]
        dataset_dict["file_path"] = image_path

        h, w = image.shape[:2]  # h, w
        dataset_dict['height'] = h
        dataset_dict['width'] = w

        # resize image
        image = transform.apply_image(image)
        image_shape = image.shape[:2]  # h, w
        dataset_dict['resize'] = image_shape

        # preprocess image for dino
        dino_image = dino_transform(image)
        dataset_dict["image"] = pad_img(dino_image, image_size)

        # CLIP image input
        clip_image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        vaild = torch.ones_like(clip_image)
        clip_image_pad = pad_img(clip_image, image_size)
        vaild_pad = pad_img(vaild, image_size)
        clip_image_pad = F.interpolate(
            clip_image_pad[None, ...], clip_image_size, mode="bilinear", align_corners=False, antialias=True
        )[0]
        vaild_pad = F.interpolate(
            vaild_pad[None, ...], clip_image_size, mode='nearest'
        )[0]
        clip_image_pad = (clip_image_pad - clip_pixel_mean) / clip_pixel_std
        dataset_dict["clip_image"] = clip_image_pad * vaild_pad

        dataset_dicts.append(dataset_dict)
    return dataset_dicts