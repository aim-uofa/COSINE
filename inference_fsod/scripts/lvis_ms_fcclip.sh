#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/scripts/common.sh"
cd "${REPO_ROOT}/inference_fsod"

FSOD_OUTPUT_ROOT="${FSOD_OUTPUT_ROOT:-${REPO_ROOT}/inference_fsod/outputs}"
NUM_GPUS=${NUM_GPUS:-4}
weights=${WEIGHT_ROOT}/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin
PREPROCESS_GPU=${PREPROCESS_GPU:-${GPU%%,*}}

# use_visual=True
# use_text=False


cfg=MSCOSINE_all_fc

# config=configs/LVIS/${cfg}.yaml
# output_dir=outputs/fsod/lvis/${cfg}/vis

# echo ${output_dir}

# CUDA_VISIBLE_DEVICES=7 python tools/preprocess.py \
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




# use_visual=False
# use_text=True


# config=configs/LVIS/${cfg}.yaml
# output_dir=outputs/fsod/lvis/${cfg}/text

# echo ${output_dir}

# CUDA_VISIBLE_DEVICES=7 python tools/preprocess.py \
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


use_visual=True
use_text=True


config=configs/LVIS/${cfg}.yaml
output_dir=${FSOD_OUTPUT_ROOT}/fsod/lvis/${cfg}/vis_text

echo ${output_dir}

if [[ ! -f "${output_dir}/checkpoint.pth" || "${FORCE_PREPROCESS:-0}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES=${PREPROCESS_GPU} "${PYTHON_BIN}" tools/preprocess.py \
  --config-file ${config} \
  --opts OUTPUT_DIR ${output_dir} \
  MODEL.COSINE.use_visual ${use_visual} \
  MODEL.COSINE.use_text ${use_text} \
  MODEL.COSINE.preprocess True \
  MODEL.WEIGHTS ${weights}
fi

CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON_BIN}" tools/test.py \
--num-gpus=${NUM_GPUS} \
--config-file ${config} \
--opts MODEL.COSINE.preprocess False \
MODEL.COSINE.use_visual ${use_visual} \
MODEL.COSINE.use_text ${use_text} \
OUTPUT_DIR ${output_dir} \
MODEL.WEIGHTS ${weights}
