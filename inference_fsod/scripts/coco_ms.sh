#!/bin/bash

weights=${WEIGHT_ROOT:-../models/cosine}/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin

use_visual=True
use_text=False

# for shots in 1 5;do
#   for seed in 0 1 2 3 4 5 6 7 8 9;do

#   config=configs/COCO_MS/${shots}shots/seed${seed}.yaml
#   output_dir=outputs/fsod/mscosine/coco_vis/${shots}shot/seed${seed}

#   echo ${output_dir}

#   CUDA_VISIBLE_DEVICES=0 python tools/preprocess.py \
#   --config-file ${config} \
#   --opts OUTPUT_DIR ${output_dir} \
#   MODEL.COSINE.use_visual ${use_visual} \
#   MODEL.COSINE.use_text ${use_text} \
#   MODEL.COSINE.preprocess True \
#   MODEL.WEIGHTS ${weights}

#   CUDA_VISIBLE_DEVICES=0 python tools/test.py \
#   --num-gpus=1 \
#   --config-file ${config} \
#   --opts MODEL.COSINE.preprocess False \
#   MODEL.COSINE.use_visual ${use_visual} \
#   MODEL.COSINE.use_text ${use_text} \
#   OUTPUT_DIR ${output_dir} \
#   MODEL.WEIGHTS ${weights}

#   done
# done


for shots in 1 5;do
  for seed in 0 1 2 3 4 5 6 7 8 9;do

  config=configs/VOC_MS/${shots}shots/seed${seed}.yaml
  output_dir=outputs/fsod/mscosine/voc_vis/${shots}shot/seed${seed}

  echo ${output_dir}

  CUDA_VISIBLE_DEVICES=1 python tools/preprocess.py \
  --config-file ${config} \
  --opts OUTPUT_DIR ${output_dir} \
  MODEL.COSINE.use_visual ${use_visual} \
  MODEL.COSINE.use_text ${use_text} \
  MODEL.COSINE.preprocess True \
  MODEL.WEIGHTS ${weights}

  CUDA_VISIBLE_DEVICES=1,2 python tools/test.py \
  --num-gpus=2 \
  --config-file ${config} \
  --opts MODEL.COSINE.preprocess False \
  MODEL.COSINE.use_visual ${use_visual} \
  MODEL.COSINE.use_text ${use_text} \
  OUTPUT_DIR ${output_dir} \
  MODEL.WEIGHTS ${weights}

  done
done




# use_visual=False
# use_text=True

# for shots in 1;do
#   for seed in 0;do

#   config=configs/COCO/${shots}shots/seed${seed}.yaml
#   output_dir=outputs/fsod/cosine/coco_text/${shots}shot/seed${seed}

#   echo ${output_dir}

#   CUDA_VISIBLE_DEVICES=1 python tools/preprocess.py \
#   --config-file ${config} \
#   --opts OUTPUT_DIR ${output_dir} \
#   MODEL.COSINE.use_visual ${use_visual} \
#   MODEL.COSINE.use_text ${use_text} \
#   MODEL.COSINE.preprocess True \
#   MODEL.WEIGHTS ${weights}

#   CUDA_VISIBLE_DEVICES=1,2 python tools/test.py \
#   --num-gpus=2 \
#   --config-file ${config} \
#   --opts MODEL.COSINE.preprocess False \
#   MODEL.COSINE.use_visual ${use_visual} \
#   MODEL.COSINE.use_text ${use_text} \
#   OUTPUT_DIR ${output_dir} \
#   MODEL.WEIGHTS ${weights}

#   done
# done



# use_visual=True
# use_text=True

# for shots in 1 5;do
#   for seed in 0;do

#   config=configs/COCO/${shots}shots/seed${seed}.yaml
#   output_dir=outputs/fsod/cosine/coco_vis_text/${shots}shot/seed${seed}

#   echo ${output_dir}

#   CUDA_VISIBLE_DEVICES=1 python tools/preprocess.py \
#   --config-file ${config} \
#   --opts OUTPUT_DIR ${output_dir} \
#   MODEL.COSINE.use_visual ${use_visual} \
#   MODEL.COSINE.use_text ${use_text} \
#   MODEL.COSINE.preprocess True \
#   MODEL.WEIGHTS ${weights}

#   CUDA_VISIBLE_DEVICES=1,2 python tools/test.py \
#   --num-gpus=2 \
#   --config-file ${config} \
#   --opts MODEL.COSINE.preprocess False \
#   MODEL.COSINE.use_visual ${use_visual} \
#   MODEL.COSINE.use_text ${use_text} \
#   OUTPUT_DIR ${output_dir} \
#   MODEL.WEIGHTS ${weights}

#   done
# done