import os

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from ..load_data.coco import register_coco_instances
from ..load_data.lvis import get_lvis_instances_meta, register_lvis_instances
from ..load_data.paco import register_paco_instances, get_paco_instances_meta
from ..load_data.object365 import register_object365_od, get_o365_metadata

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train_ins": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val_ins": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2017_train_ins": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val_ins": ("coco/val2017", "coco/annotations/instances_val2017.json"),
}

def register_instance_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train_ins": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val_ins": ("coco/", "lvis/lvis_v1_val.json"),
    },
}

def register_instance_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )



# ==== Predefined datasets and splits for PACO ==========


_PREDEFINED_PACO = {
    "paco_lvis_v1_train": (
        "coco/", "paco/paco_lvis_v1_train.json"
    ),
    "paco_lvis_v1_val": (
        "coco/", "paco/paco_lvis_v1_val.json"
    ),
    "paco_lvis_v1_test": (
        "coco/", "paco/paco_lvis_v1_test.json"
    ),
}


def register_all_paco(root):
    for key, (image_root, json_file) in _PREDEFINED_PACO.items():
        register_paco_instances(
            key,
            get_paco_instances_meta(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



# ==== Predefined datasets and splits for O365 with SAM masks ==========

_PREDEFINED_SPLITS_OBJECT365 = {
    "object365_train": (
        "Objects365",
        # "Objects365/annotations/zhiyuan_objv2_train.json",
        "Objects365/annotations/sam_obj365_train_1742k.json"
    ),
    "object365_val": (
        "Objects365",
        # "Objects365/annotations/zhiyuan_objv2_val.json",
        # "Objects365/annotations/sam_obj365_val_5k.json"
        "Objects365/annotations/sam_obj365_train_75k.json"
    ),
}

def register_all_object365(root):
    for (
        prefix,
        (image_root, od_json),
    ) in _PREDEFINED_SPLITS_OBJECT365.items():
        register_object365_od(
            prefix,
            get_o365_metadata(),
            os.path.join(root, image_root),
            os.path.join(root, od_json),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_instance_coco(_root)
register_instance_lvis(_root)
register_all_paco(_root)
register_all_object365(_root)