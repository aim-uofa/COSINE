import argparse
import json
import os
import random
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="datasets/lvis/lvis_v1_train.json", help="path to the annotation file"
    )
    parser.add_argument("--shots", type=int, default=10, help="number of shots")
    args = parser.parse_args()
    return args


def get_shots(args):
    data = json.load(open(args.data, "r"))

    CAT_NAMES = [cat['name'] for cat in data['categories']]

    ann = data["annotations"]

    anno_cat = {i: [] for i in range(len(CAT_NAMES))}
    for a in ann:
        anno_cat[a["category_id"] - 1].append(a)

    anno = []
    for i, c in enumerate(CAT_NAMES):
        if len(anno_cat[i]) < args.shots:
            shots = anno_cat[i]
        else:
            shots = random.sample(anno_cat[i], args.shots)
        print(c, len(shots))
        anno.extend(shots)
    print(i, len(anno))
    new_data = {
        "info": data["info"],
        "licenses": data["licenses"],
        "categories": data["categories"],
        "images": data["images"],
        "annotations": anno,
    }

    save_path = os.path.join("datasets/lvissplit", "lvis_v1_shots.json")
    with open(save_path, "w") as f:
        json.dump(new_data, f)


if __name__ == "__main__":
    random.seed(0)

    args = parse_args()
    get_shots(args)