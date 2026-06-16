#!/bin/bash

weights=${WEIGHT_ROOT:-../models/cosine}/cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin

use_visual=True
use_text=False

for shots in 1;do
  for seed in 3 4 5;do

  config=configs/COCO/${shots}shots/seed${seed}.yaml
  output_dir=outputs/fsod/cosine/coco_vis/${shots}shot/seed${seed}

  echo ${output_dir}

  CUDA_VISIBLE_DEVICES=4 python tools/preprocess.py \
  --config-file ${config} \
  --opts OUTPUT_DIR ${output_dir} \
  MODEL.COSINE.use_visual ${use_visual} \
  MODEL.COSINE.use_text ${use_text} \
  MODEL.COSINE.preprocess True \
  MODEL.WEIGHTS ${weights}

  CUDA_VISIBLE_DEVICES=4,5 python tools/test.py \
  --num-gpus=2 \
  --config-file ${config} \
  --opts MODEL.COSINE.preprocess False \
  MODEL.COSINE.use_visual ${use_visual} \
  MODEL.COSINE.use_text ${use_text} \
  OUTPUT_DIR ${output_dir} \
  MODEL.WEIGHTS ${weights}

  done
done
