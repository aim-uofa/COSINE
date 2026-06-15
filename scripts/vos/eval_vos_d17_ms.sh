#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
cd "${REPO_ROOT}"

TYPE=${TYPE:-linear}
RATIO=${RATIO:-20}
# MODEL=mscosine_unified_fd1_md6_lr1e-4_ep50
# WEIGHT=${WEIGHT_ROOT:-models/cosine}/${MODEL}/pytorch_model_49ep/pytorch_model.bin

for ep in 9 14 19 24;do

MODEL=mscosine_vos_fd1_md6_lr1e-4_ep50
WEIGHT=${WEIGHT_ROOT}/${MODEL}/pytorch_model/pytorch_model_${ep}ep/pytorch_model.bin

dataset=D17
for N_FRAME in 6;do

  LOG=${MODEL}/${ep}ep/${dataset}
  echo ${LOG}

  CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON_BIN}" tools/eval_vos_ms.py \
    --dataset ${dataset} \
    --d17_path "${DATA_ROOT}/vos/DAVIS/2017" \
    --weights ${WEIGHT} \
    --output ${OUTPUT_ROOT}/vos/${LOG} \
    --num_frame ${N_FRAME} \
    --memory_decay_type ${TYPE} \
    --memory_decay_ratio ${RATIO} \
    --fix_first_frame
  "${PYTHON_BIN}" inference_vos/davis2017-evaluation/evaluation_method.py \
    --davis_path "${DATA_ROOT}/vos/DAVIS/2017/trainval" \
    --set val \
    --task semi-supervised \
    --results_path ${OUTPUT_ROOT}/vos/${LOG}

done
done
