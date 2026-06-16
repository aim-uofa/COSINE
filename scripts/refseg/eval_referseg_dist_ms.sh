#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
cd "${REPO_ROOT}"

model=mscosine_refseg_fd1_md6_lr1e-4_ep50
weight=${WEIGHT_ROOT}/${model}/pytorch_model_19ep/pytorch_model.bin
NUM_GPUS=${NUM_GPUS:-4}

data="refcoco|unc|val"
output=${OUTPUT_ROOT}/referseg/${model}/refcoco

CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON_BIN}" tools/eval_referseg_ms.py \
  --num-gpus ${NUM_GPUS} \
  --val_dataset ${data} \
  --dataset_dir "${DATA_ROOT}" \
  --weights ${weight} \
  --output_dir ${output} \
  --dist_eval


data="refcoco+|unc|val"
output=${OUTPUT_ROOT}/referseg/${model}/refcoco+

CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON_BIN}" tools/eval_referseg_ms.py \
  --num-gpus ${NUM_GPUS} \
  --val_dataset ${data} \
  --dataset_dir "${DATA_ROOT}" \
  --weights ${weight} \
  --output_dir ${output} \
  --dist_eval


data="refcocog|umd|val"
output=${OUTPUT_ROOT}/referseg/${model}/refcocog

CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON_BIN}" tools/eval_referseg_ms.py \
  --num-gpus ${NUM_GPUS} \
  --val_dataset ${data} \
  --dataset_dir "${DATA_ROOT}" \
  --weights ${weight} \
  --output_dir ${output} \
  --dist_eval
