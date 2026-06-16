#!/usr/bin/env python
import os
import sys
import glob
from time import time
import argparse
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', default='outputs/vos/mscosine_vos_fd1_md6_lr1e-4_ep50/24ep/D17', type=str)
parser.add_argument('--flow_path', default='datasets/vos/DAVIS/2017/trainval/flow/npy', type=str)
parser.add_argument('--split_path', default='datasets/vos/DAVIS/2017/trainval/ImageSets/2017/val.txt', type=str)

args, _ = parser.parse_known_args()

print(args.results_path)

def read_video_list(txt_path):
    """
    读取 DAVIS 图像子目录列表（每行一个 video_name）
    """
    with open(txt_path, 'r') as f:
        lines = f.read().splitlines()
    return [line.strip() for line in lines if line.strip()]

def compute_temporal_consistency(res1_img, res2_img, flow_npy):
    """
    Args:
        res1_img: np.ndarray, shape [H, W], int, label of t-1
        res2_img: np.ndarray, shape [H, W], int, label of t
        flow_npy: np.ndarray, shape [H, W, 2], float32, optical flow from t-1 to t

    Returns:
        consistency score in [0, 1]
    """
    H, W = res1_img.shape
    flow = flow_npy

    # meshgrid of coordinates
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

    # flow coordinates
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    # warp res1 to frame2 using flow (use nearest to keep discrete label)
    warped_res1 = cv2.remap(res1_img.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)

    # mask for valid flow (remap may go outside image)
    valid_mask = (map_x >= 0) & (map_x < W) & (map_y >= 0) & (map_y < H) & (warped_res1 != -1)

    # match where label is preserved
    match = (warped_res1 == res2_img) & valid_mask
    total = np.sum(valid_mask)
    correct = np.sum(match)

    return float(correct) / total if total > 0 else 0.0

davis_vid = read_video_list(args.split_path)

consistency_dict = {}
for vid in tqdm(davis_vid):

    results_path = os.path.join(args.results_path, vid)
    flow_path = os.path.join(args.flow_path, vid)

    results = glob.glob(os.path.join(results_path, '*.png')) + \
            glob.glob(os.path.join(results_path, '*.jpg'))

    flows = glob.glob(os.path.join(flow_path, '*.npy'))

    results = sorted(results)
    flows = sorted(flows)

    consistency_list = []

    for res1, res2, flow in zip(results[:-1], results[1:], flows):

        res1_name = os.path.splitext(os.path.basename(res1))[0]
        res2_name = os.path.splitext(os.path.basename(res2))[0]
        flow_name = os.path.splitext(os.path.basename(flow))[0]

        assert f'flow_{res1_name}_to_{res2_name}' == flow_name

        res1_img = np.array(Image.open(res1))
        res2_img = np.array(Image.open(res2))
        flow_npy = np.load(flow)

        h1, w1 = res1_img.shape[:2]
        h2, w2 = res2_img.shape[:2]
        hf, wf = flow_npy.shape[:2]

        if (h1, w1) != (h2, w2):
            raise ValueError(f"Image sizes do not match: {res1_img.shape} vs {res2_img.shape}")

        if (hf, wf) != (h1, w1):
            raise ValueError(f"Flow shape {flow_npy.shape} does not match image size ({h1}, {w1})")

        consistency = compute_temporal_consistency(res1_img, res2_img, flow_npy)
        consistency_list.append(consistency)

    consistency_dict[vid] = sum(consistency_list) / len(consistency_list)

consistency_values = list(consistency_dict.values())
consistency_dict['mean'] = sum(consistency_values) / len(consistency_values)

for k, v in consistency_dict.items():
    print(f'{k}: {v}')
