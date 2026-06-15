#!/bin/bash

data=coco
weight=${WEIGHT_ROOT:-models/cosine}/cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin

for shot in 1 5;do
  for fold in 0 1 2 3;do

  output=outputs/fss/${data}/${shot}shot/fold${fold}_text
  echo ${output}

  CUDA_VISIBLE_DEVICES=0 python tools/eval_fss.py  \
    --benchmark ${data} \
    --fold ${fold} \
    --nshot ${shot} \
    --weights ${weight} \
    --log-root ${output} \
    --use_text
  done
done
