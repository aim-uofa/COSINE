# Dataset Layout

COSINE uses simple relative paths so the same checkout can run on a workstation,
a cluster, or a NAS. Datasets can be real directories or symlinks.

## Expected Tree

```text
datasets/
├── coco/
│   ├── annotations/
│   ├── train2014/
│   ├── val2014/
│   ├── train2017/
│   └── val2017/
├── fss/
│   ├── COCO2014/
│   ├── LVIS/
│   └── VOC2012/
├── refer_seg/
│   ├── images/
│   ├── refcoco/
│   ├── refcoco+/
│   └── refcocog/
└── vos/
    ├── DAVIS/
    │   ├── 2016/
    │   └── 2017/
    └── Youtube-VOS2019/

inference_fsod/datasets/
├── coco -> ../../datasets/coco
├── coco2017 -> ../../datasets/coco
├── lvis -> ../../datasets/lvis
└── lvissplit/
```

## Task Mapping

| Task | Code default | Notes |
| --- | --- | --- |
| FSS COCO-20i | `datasets/fss/COCO2014` | Used by `--benchmark coco`. |
| FSS LVIS-92i | `datasets/fss/LVIS` | Used by `--benchmark lvis`. |
| FSS Pascal-5i | `datasets/fss/VOC2012` | Used by `--benchmark pascal`. |
| RefCOCO / RefCOCO+ / RefCOCOg | `datasets/refer_seg` | Used by `tools/eval_referseg*.py`. |
| DAVIS 2017 | `datasets/vos/DAVIS/2017` | Used by `tools/eval_vos_ms.py --dataset D17`. |
| YouTube-VOS 2019 | `datasets/vos/Youtube-VOS2019` | Used by `tools/eval_vos_ms.py --dataset Y19`. |
| FSOD COCO/LVIS | `inference_fsod/datasets/...` | Detectron2-style configs use `coco`, `coco2017`, `lvis`, and `lvissplit`. |

## Validation

After preparing datasets and weights, run:

```bash
bash scripts/check_required_assets.sh
```

This only checks the expected filesystem layout. It does not verify dataset
integrity or benchmark annotations.
