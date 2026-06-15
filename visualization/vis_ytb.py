import os.path
import torch
import numpy as np
from PIL import Image

import sys
sys.path.append('./')

import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
from visualization.visualizer import Visualizer as MaskVisualizer

def show_mask(mask, ax, color=False):

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

im_mean = (124, 116, 104)

im_normalization = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

inv_im_trans = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])


def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)

    for ni, l in enumerate(labels):
        Ms[ni] = (masks == l).astype(np.uint8)

    return Ms


class MaskMapper:
    """
    This class is used to convert a indexed-mask to a one-hot representation.
    It also takes care of remapping non-continuous indices
    It has two modes:
        1. Default. Only masks with new indices are supposed to go into the remapper.
        This is also the case for YouTubeVOS.
        i.e., regions with index 0 are not "background", but "don't care".

        2. Exhaustive. Regions with index 0 are considered "background".
        Every single pixel is considered to be "labeled".
    """
    def __init__(self):
        self.labels = []
        self.remappings = {}

        # if coherent, no mapping is required
        self.coherent = True

    def convert_mask(self, mask, exhaustive=False):
        # mask is in index representation, H*W numpy array
        labels = np.unique(mask).astype(np.uint8)
        labels = labels[labels!=0].tolist()

        new_labels = list(set(labels) - set(self.labels))
        if not exhaustive:
            assert len(new_labels) == len(labels), 'Old labels found in non-exhaustive mode'

        # add new remappings
        for i, l in enumerate(new_labels):
            self.remappings[l] = i+len(self.labels)+1
            if self.coherent and i+len(self.labels)+1 != l:
                self.coherent = False

        if exhaustive:
            new_mapped_labels = range(1, len(self.labels)+len(new_labels)+1)
        else:
            if self.coherent:
                new_mapped_labels = new_labels
            else:
                new_mapped_labels = range(len(self.labels)+1, len(self.labels)+len(new_labels)+1)

        self.labels.extend(new_labels)
        mask = torch.from_numpy(all_to_onehot(mask, self.labels)).float()

        # mask num_objects*H*W
        return mask, new_mapped_labels


    def remap_index_mask(self, mask):
        # mask is in index representation, H*W numpy array
        if self.coherent:
            return mask

        new_mask = np.zeros_like(mask)
        for l, i in self.remappings.items():
            new_mask[mask==i] = l
        return new_mask


if __name__ == '__main__':

    colors = [
        np.array([30 / 255, 144 / 255, 255 / 255, 0.6]),
        np.array([255 / 255, 106 / 255, 106 / 255, 0.6]),
        np.array([0 / 255, 250 / 255, 154 / 255, 0.6]),
        np.array([186 / 255, 85 / 255, 211 / 255, 0.6]),
        np.array([78 / 255, 238 / 255, 148 / 255, 0.6]),
        np.array([160 / 255, 32 / 255, 240 / 255, 0.6]),
        np.array([135 / 255, 206 / 255, 235 / 255, 0.6]),
        np.array([255 / 255, 215 / 255, 0 / 255, 0.6]),
        np.array([ 127 / 255,  255 / 255,  212 / 255, 0.6]),
    ]

    root  = "outputs/vos/mscosine_vos_fd1_md6_lr1e-4_ep50/24ep"
    prediction_path = os.path.join(root, "Y19/Annotations")
    video_path = "datasets/vos/Youtube-VOS2019"

    output_path = os.path.join(root, 'vis_Y19_new')

    # prediction_path = os.path.join(prediction_path, 'Annotations')
    video_path = os.path.join(video_path, 'valid/JPEGImages')

    video_dir = os.listdir(prediction_path)
    video_dir = sorted(video_dir)
    video_dir = [v for v in video_dir if '.csv' not in v and 'vis' not in v]

    for video_name in tqdm(video_dir):
        video_save_path = os.path.join(output_path, video_name)
        video_s_path = os.path.join(video_path, video_name)
        labels_path = os.path.join(prediction_path, video_name)
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)

        frame_list = os.listdir(labels_path)
        frame_list = sorted(frame_list)
        for frame in frame_list:

            anno_path = os.path.join(labels_path, frame)
            image_name = frame.split('.png')[0]+'.jpg'
            image_path = os.path.join(video_s_path, image_name)

            frame_save_path = os.path.join(video_save_path, frame)

            img = Image.open(image_path).convert('RGB')
            img1 = np.array(img, dtype=np.uint8)

            mask1 = Image.open(anno_path).convert('P')
            mask1 = np.array(mask1, dtype=np.uint8)

            mapper1 = MaskMapper()
            mask1, labels1 = mapper1.convert_mask(mask1, exhaustive=False)

            tgt_img = transforms.ToTensor()(img1)[None, ...]
            masks = F.interpolate(mask1[None, ...], tgt_img.shape[-2:], mode='nearest')[0]

            mask_visualizer = MaskVisualizer(tgt_img[0].permute(1,2,0).cpu().numpy() * 255)
            vis_output = mask_visualizer.draw_masks(masks.cpu().numpy())
            vis_output.save(frame_save_path)
