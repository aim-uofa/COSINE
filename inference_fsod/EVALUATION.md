## Evaluation - Few-shot Instance Segmentation


### Prepare data

Download following datasets:


> #### COCO 2014
> - Download [cocosplit](https://drive.google.com/file/d/12jGNdhdL8jz5YO8Gz5P-liNtY7eAz6Av/view).
> #### VOC
> - Download [voc-cocostyle.zip](https://drive.google.com/file/d/16P6ewBkjg5WguQEzH3-GfnhHVht1cN-9/view?usp=sharing).

Create a directory 'datasets' for the above datasets in 'inference_fsod/' and appropriately place each dataset to have following directory structure:

    datasets/
    ├── coco/
    │   ├── annotations/
    │   │   └── instances_{train,val}2014.json
    │   └── {train,val}2014/
    ├── cocosplit/
    │   ├── datasplit/
    │   ├── seed0/
    │   ├── seed1/
    │   └── ...
    ├── VOCOutput/
    │   ├── annotations
    │   │   ├── train.json
    │   │   ├── val.json
    │   │   └── val_converted.json
    │   ├── train/
    │   ├── val/



### Testing


```
cd inference_fsod

# COCO/VOC few-shot instance segmentation.
bash scripts/coco_ms.sh

# LVIS few-shot instance segmentation.
bash scripts/lvis_ms_fcclip.sh
```

The scripts use COSINE configuration keys such as `MODEL.COSINE.preprocess`
and read checkpoints from `../models/cosine` by default. Override the model
root with `WEIGHT_ROOT=/path/to/models/cosine` when needed.

For example, a single LVIS command expands to:

```
python tools/preprocess.py \
  --config-file configs/LVIS/MSCOSINE_all_fc.yaml \
  --opts OUTPUT_DIR outputs/fsod/lvis/MSCOSINE_all_fc/vis_text \
  MODEL.COSINE.use_visual True \
  MODEL.COSINE.use_text True \
  MODEL.COSINE.preprocess True \
  MODEL.WEIGHTS ../models/cosine/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin

python tools/test.py \
  --num-gpus=4 \
  --config-file configs/LVIS/MSCOSINE_all_fc.yaml \
  --opts MODEL.COSINE.preprocess False \
  MODEL.COSINE.use_visual True \
  MODEL.COSINE.use_text True \
  OUTPUT_DIR outputs/fsod/lvis/MSCOSINE_all_fc/vis_text \
  MODEL.WEIGHTS ../models/cosine/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin
```
