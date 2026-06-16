#!/bin/bash

shot=1
data=coco
fold=0
weight=${WEIGHT_ROOT:-models/cosine}/cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin
output=outputs/fss/${data}/${shot}shot/fold${fold}_visual
echo ${output}

CUDA_VISIBLE_DEVICES=1 python tools/eval_fss.py  \
  --benchmark ${data} \
  --fold ${fold} \
  --nshot ${shot} \
  --weights ${weight} \
  --log-root ${output} \
  --use_visual
