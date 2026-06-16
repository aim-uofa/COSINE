#!/bin/bash

data=coco
model=cosine_fd1_md6_lr1e-4_ep50
weight=${WEIGHT_ROOT:-models/cosine}/${model}/pytorch_model.bin

# text
for shot in 1 5;do
  for fold in 0 1 2 3;do

  output=outputs/fss/${model}/${data}/${shot}shot/fold${fold}_text
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

# visual
for shot in 1 5;do
  for fold in 0 1 2 3;do

  output=outputs/fss/${model}/${data}/${shot}shot/fold${fold}_visual
  echo ${output}

  CUDA_VISIBLE_DEVICES=0 python tools/eval_fss.py  \
    --benchmark ${data} \
    --fold ${fold} \
    --nshot ${shot} \
    --weights ${weight} \
    --log-root ${output} \
    --use_visual
  done
done

# visual_text
for shot in 1 5;do
  for fold in 0 1 2 3;do

  output=outputs/fss/${model}/${data}/${shot}shot/fold${fold}_visual_text
  echo ${output}

  CUDA_VISIBLE_DEVICES=0 python tools/eval_fss.py  \
    --benchmark ${data} \
    --fold ${fold} \
    --nshot ${shot} \
    --weights ${weight} \
    --log-root ${output} \
    --use_text --use_visual
  done
done
