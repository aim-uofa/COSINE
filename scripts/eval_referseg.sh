#!/bin/bash

model=cosine_fd1_md6_lr1e-4_ep50
weight=${WEIGHT_ROOT:-models/cosine}/${model}/pytorch_model.bin

data="refcoco|unc|val"
output=outputs/referseg/${model}/refcoco

CUDA_VISIBLE_DEVICES=5 python tools/eval_referseg.py \
  --num-gpus 1 \
  --val_dataset ${data} \
  --weights ${weight} \
  --output_dir ${output}


data="refcoco+|unc|val"
output=outputs/referseg/${model}/refcoco+

CUDA_VISIBLE_DEVICES=5 python tools/eval_referseg.py \
  --num-gpus 1 \
  --val_dataset ${data} \
  --weights ${weight} \
  --output_dir ${output}


data="refcocog|umd|val"
output=outputs/referseg/${model}/refcocog

CUDA_VISIBLE_DEVICES=5 python tools/eval_referseg.py \
  --num-gpus 1 \
  --val_dataset ${data} \
  --weights ${weight} \
  --output_dir ${output}
