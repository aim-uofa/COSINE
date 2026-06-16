#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"

cd "${REPO_ROOT}"

LOG="outputs/repro/fsod_full/MSCOSINE_all_fc/vis_text/preprocess_and_test.log"
WEIGHTS="../models/cosine/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin"
OUT="outputs/repro/fsod_full/MSCOSINE_all_fc/vis_text"

mkdir -p "$(dirname "${LOG}")"

{
  echo "START $(date +%F_%T)"
  echo "TASK fsod_lvis_mscosine_all_fc_vis_text_preprocess_and_test"
  echo "GPU ${GPU}"
  echo "OUTPUT_DIR inference_fsod/${OUT}"
  cd inference_fsod

  echo "COMMAND preprocess"
  /usr/bin/time -v env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" tools/preprocess.py \
    --config-file configs/LVIS/MSCOSINE_all_fc.yaml \
    --opts OUTPUT_DIR "${OUT}" \
    MODEL.COSINE.use_visual True \
    MODEL.COSINE.use_text True \
    MODEL.COSINE.preprocess True \
    MODEL.WEIGHTS "${WEIGHTS}"

  test -f "${OUT}/checkpoint.pth"
  echo "CHECKPOINT_READY ${OUT}/checkpoint.pth $(stat -c%s "${OUT}/checkpoint.pth" 2>/dev/null || stat -f%z "${OUT}/checkpoint.pth")"

  echo "COMMAND test"
  /usr/bin/time -v env CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" tools/test.py \
    --num-gpus=1 \
    --config-file configs/LVIS/MSCOSINE_all_fc.yaml \
    --opts MODEL.COSINE.preprocess False \
    MODEL.COSINE.use_visual True \
    MODEL.COSINE.use_text True \
    OUTPUT_DIR "${OUT}" \
    MODEL.WEIGHTS "${WEIGHTS}"

  echo "END $(date +%F_%T) rc=0"
} 2>&1 | tee -a "${LOG}"
