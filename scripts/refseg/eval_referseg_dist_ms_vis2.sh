#!/bin/bash

model=mscosine_refseg_fd1_md6_lr1e-4_ep50
weight=${WEIGHT_ROOT:-models/cosine}/${model}/pytorch_model_19ep/pytorch_model.bin

data="refcoco|unc|val"
output=${OUTPUT_ROOT:-outputs/visualization}/refcoco_2

CUDA_VISIBLE_DEVICES=0 python tools/eval_referseg_ms_vis.py \
  --num-gpus 1 \
  --val_dataset ${data} \
  --weights ${weight} \
  --output_dir ${output} \
  --min_lenth 4


data="refcoco+|unc|val"

CUDA_VISIBLE_DEVICES=0 python tools/eval_referseg_ms_vis.py \
  --num-gpus 1 \
  --val_dataset ${data} \
  --weights ${weight} \
  --output_dir ${output} \
  --min_lenth 4

data="refcocog|umd|val"

CUDA_VISIBLE_DEVICES=0 python tools/eval_referseg_ms_vis.py \
  --num-gpus 1 \
  --val_dataset ${data} \
  --weights ${weight} \
  --output_dir ${output} \
  --min_lenth 4
