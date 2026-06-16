#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"

cd "${REPO_ROOT}" || exit 2

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

run_one() {
  local name="$1"; shift
  local run_dir="$1"; shift
  mkdir -p "${run_dir}"
  {
    echo "START $(date +%F_%T)"
    echo "TASK ${name}"
    echo "GPU ${GPU}"
    echo "CMD $*"
    /usr/bin/time -v env CUDA_VISIBLE_DEVICES="${GPU}" "$@"
    rc=$?
    echo "END $(date +%F_%T) rc=${rc}"
    exit "${rc}"
  } 2>&1 | tee "${run_dir}/run.log"
  return "${PIPESTATUS[0]}"
}

run_one "fss_lvis_ms_fold0_1shot_visual_text" \
  "outputs/repro/fss_full/mscosine_unified_fd1_md6_lr1e-4_ep50_49ep/lvis/1shot/fold0_visual_text" \
  "${PYTHON_BIN}" tools/eval_fss_ms.py \
    --datapath datasets/fss \
    --benchmark lvis \
    --fold 0 \
    --nshot 1 \
    --weights models/cosine/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin \
    --log-root outputs/repro/fss_full/mscosine_unified_fd1_md6_lr1e-4_ep50_49ep/lvis/1shot/fold0_visual_text \
    --use_text --use_visual \
    --nworker 0
rc1=$?

run_one "fss_lvis_single_fold0_1shot_visual_text" \
  "outputs/repro/fss_full/cosine_fd1_md6_lr1e-4_ep50/lvis/1shot/fold0_visual_text" \
  "${PYTHON_BIN}" tools/eval_fss.py \
    --datapath datasets/fss \
    --benchmark lvis \
    --fold 0 \
    --nshot 1 \
    --weights models/cosine/cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin \
    --log-root outputs/repro/fss_full/cosine_fd1_md6_lr1e-4_ep50/lvis/1shot/fold0_visual_text \
    --use_text --use_visual \
    --nworker 0
rc2=$?

echo "SUMMARY rc_ms=${rc1} rc_single=${rc2}"
exit $(( rc1 != 0 ? rc1 : rc2 ))
