#!/bin/bash

model=cosine_fd1_md6_lr1e-4_ep50
weight=${WEIGHT_ROOT:-models/cosine}/${model}/pytorch_model.bin

data="refcoco|unc|val"
output=outputs/referseg_dist/${model}/refcoco

python tools/eval_referseg.py \
  --num-gpus 8 \
  --val_dataset ${data} \
  --weights ${weight} \
  --output_dir ${output} \
  --dist_eval


data="refcoco+|unc|val"
output=outputs/referseg_dist/${model}/refcoco+

python tools/eval_referseg.py \
  --num-gpus 8 \
  --val_dataset ${data} \
  --weights ${weight} \
  --output_dir ${output} \
  --dist_eval


data="refcocog|umd|val"
output=outputs/referseg_dist/${model}/refcocog

python tools/eval_referseg.py \
  --num-gpus 8 \
  --val_dataset ${data} \
  --weights ${weight} \
  --output_dir ${output} \
  --dist_eval
