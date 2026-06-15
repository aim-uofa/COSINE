import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import tqdm
import cv2
import json
from tqdm import tqdm

ID2CLASS = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
    }
CLASS2ID = {v: k for k, v in ID2CLASS.items()}

def get_ref_masks(data_path, data_dir, out_dir, sel_cat="elephant"):
    data = json.load(open(data_path))
    imgid_imganno = {a["id"]: a for a in data["images"]}
    sel_id = CLASS2ID[sel_cat]
    obj_anno = []
    os.makedirs(out_dir, exist_ok=True)
    for img_anno in tqdm(data["annotations"]):
        obj_annos = img_anno["segments_info"]
        image_id = img_anno["image_id"]
        file_name = img_anno["file_name"]
        for a in obj_annos:
            if a["category_id"] == sel_id:
                a_new = a.copy()
                a_new["file_name"] = file_name
                img_path = os.path.join(data_dir, file_name.replace(".png", ".jpg"))
                assert os.path.exists(img_path), img_path
                out_path = os.path.join(out_dir, file_name.replace(".png", ".jpg"))
                image = cv2.imread(img_path)
                cv2.imwrite(out_path, image)
                print(out_path)
                print(a_new)
                input()

if __name__ == '__main__':
    data_path = "datasets/coco/annotations/panoptic_val2017.json"
    data_dir = "datasets/coco/val2017"
    out_dir = "outputs/visualization/combseg/example/car"
    class_name = "car"
    get_ref_masks(data_path, data_dir, out_dir, class_name)