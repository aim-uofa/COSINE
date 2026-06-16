#!/bin/bash

weights=${WEIGHT_ROOT:-../models/cosine}/cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin


use_visual=True
use_text=True


cfg=COSINE_all_fc

config=configs/LVIS/${cfg}.yaml
output_dir=outputs/fsod/lvis/${cfg}/vis_text

echo ${output_dir}

# CUDA_VISIBLE_DEVICES=0 python tools/preprocess.py \
# --config-file ${config} \
# --opts OUTPUT_DIR ${output_dir} \
# MODEL.COSINE.use_visual ${use_visual} \
# MODEL.COSINE.use_text ${use_text} \
# MODEL.COSINE.preprocess True \
# MODEL.WEIGHTS ${weights}

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/test.py \
--num-gpus=4 \
--config-file ${config} \
--opts MODEL.COSINE.preprocess False \
MODEL.COSINE.use_visual ${use_visual} \
MODEL.COSINE.use_text ${use_text} \
OUTPUT_DIR ${output_dir} \
MODEL.WEIGHTS ${weights}


# use_visual=True
# use_text=False


# cfg=COSINE_all

# config=configs/LVIS/${cfg}.yaml
# output_dir=outputs/fsod/lvis/${cfg}/vis

# echo ${output_dir}

# # CUDA_VISIBLE_DEVICES=0 python tools/preprocess.py \
# # --config-file ${config} \
# # --opts OUTPUT_DIR ${output_dir} \
# # MODEL.COSINE.use_visual ${use_visual} \
# # MODEL.COSINE.use_text ${use_text} \
# # MODEL.COSINE.preprocess True \
# # MODEL.WEIGHTS ${weights}

# CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/test.py \
# --num-gpus=4 \
# --config-file ${config} \
# --opts MODEL.COSINE.preprocess False \
# MODEL.COSINE.use_visual ${use_visual} \
# MODEL.COSINE.use_text ${use_text} \
# OUTPUT_DIR ${output_dir} \
# MODEL.WEIGHTS ${weights}


# use_visual=False
# use_text=True


# config=configs/LVIS/${cfg}.yaml
# output_dir=outputs/fsod/lvis/${cfg}/text

# echo ${output_dir}

# CUDA_VISIBLE_DEVICES=0 python tools/preprocess.py \
# --config-file ${config} \
# --opts OUTPUT_DIR ${output_dir} \
# MODEL.COSINE.use_visual ${use_visual} \
# MODEL.COSINE.use_text ${use_text} \
# MODEL.COSINE.preprocess True \
# MODEL.WEIGHTS ${weights}

# CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/test.py \
# --num-gpus=4 \
# --config-file ${config} \
# --opts MODEL.COSINE.preprocess False \
# MODEL.COSINE.use_visual ${use_visual} \
# MODEL.COSINE.use_text ${use_text} \
# OUTPUT_DIR ${output_dir} \
# MODEL.WEIGHTS ${weights}


# use_visual=True
# use_text=True

# config=configs/LVIS/${cfg}.yaml
# output_dir=outputs/fsod/lvis/${cfg}/vis_text

# echo ${output_dir}

# CUDA_VISIBLE_DEVICES=0 python tools/preprocess.py \
# --config-file ${config} \
# --opts OUTPUT_DIR ${output_dir} \
# MODEL.COSINE.use_visual ${use_visual} \
# MODEL.COSINE.use_text ${use_text} \
# MODEL.COSINE.preprocess True \
# MODEL.WEIGHTS ${weights}

# CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/test.py \
# --num-gpus=4 \
# --config-file ${config} \
# --opts MODEL.COSINE.preprocess False \
# MODEL.COSINE.use_visual ${use_visual} \
# MODEL.COSINE.use_text ${use_text} \
# OUTPUT_DIR ${output_dir} \
# MODEL.WEIGHTS ${weights}
