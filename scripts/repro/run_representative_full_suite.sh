#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-14000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-20}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-120}"

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cd "${REPO_ROOT}"
mkdir -p outputs/repro/suite_logs
SUITE_LOG="outputs/repro/suite_logs/representative_full_suite_$(date +%Y%m%d_%H%M%S).log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${SUITE_LOG}"
}

pick_gpu() {
  local task="$1"
  local min_free_mb="${2:-${GPU_MIN_FREE_MB}}"
  while true; do
    local choice
    choice="$(
      nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F, -v min_free="${min_free_mb}" -v max_util="${GPU_MAX_UTIL}" '
        {
          idx=$1+0; total=$2+0; used=$3+0; util=$4+0; free=total-used;
          if (free >= min_free && util <= max_util) {
            print idx, free, util;
            exit;
          }
        }'
    )"
    if [[ -n "${choice}" ]]; then
      log "${task}: selected GPU ${choice} (min_free=${min_free_mb}MB max_util=${GPU_MAX_UTIL}%)" >&2
      awk '{print $1}' <<<"${choice}"
      return 0
    fi
    log "${task}: waiting for a GPU with >=${min_free_mb}MB free and <=${GPU_MAX_UTIL}% util" >&2
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee -a "${SUITE_LOG}" >&2
    sleep "${GPU_POLL_SECONDS}"
  done
}

run_task() {
  local name="$1"
  local min_free_mb="$2"
  local run_dir="$3"
  shift 3

  mkdir -p "${run_dir}"
  local task_log="${run_dir}/run.log"
  local gpu
  gpu="$(pick_gpu "${name}" "${min_free_mb}")"

  {
    echo "START $(date '+%F %T')"
    echo "TASK ${name}"
    echo "GPU ${gpu}"
    echo "RUN_DIR ${run_dir}"
    echo "COMMAND CUDA_VISIBLE_DEVICES=${gpu} $*"
  } | tee -a "${task_log}" "${SUITE_LOG}"

  set +e
  /usr/bin/time -v env CUDA_VISIBLE_DEVICES="${gpu}" "$@" 2>&1 | tee -a "${task_log}"
  local rc=${PIPESTATUS[0]}
  set -e

  {
    echo "END $(date '+%F %T') rc=${rc}"
    echo
  } | tee -a "${task_log}" "${SUITE_LOG}"

  return "${rc}"
}

run_task \
  "fss_coco_single_fold0_visual_text" 12000 \
  "outputs/repro/fss_full/cosine_fd1_md6_lr1e-4_ep50/coco/1shot/fold0_visual_text" \
  "${PYTHON_BIN}" tools/eval_fss.py \
    --datapath datasets/fss \
    --benchmark coco \
    --fold 0 \
    --nshot 1 \
    --weights models/cosine/cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin \
    --log-root outputs/repro/fss_full/cosine_fd1_md6_lr1e-4_ep50/coco/1shot/fold0_visual_text \
    --use_text --use_visual \
    --nworker 0

run_task \
  "refseg_refcoco_val" 16000 \
  "outputs/repro/referseg_full/mscosine_refseg_fd1_md6_lr1e-4_ep50/refcoco_val" \
  "${PYTHON_BIN}" tools/eval_referseg_ms.py \
    --val_dataset "refcoco|unc|val" \
    --dataset_dir datasets \
    --weights models/cosine/mscosine_refseg_fd1_md6_lr1e-4_ep50/pytorch_model_19ep/pytorch_model.bin \
    --output_dir outputs/repro/referseg_full/mscosine_refseg_fd1_md6_lr1e-4_ep50/refcoco_val \
    --num_workers 0 \
    --device cuda

run_task \
  "vos_davis17_val_24ep" 16000 \
  "outputs/repro/vos_full/mscosine_vos_fd1_md6_lr1e-4_ep50/24ep/D17" \
  "${PYTHON_BIN}" tools/eval_vos_ms.py \
    --dataset D17 \
    --d17_path datasets/vos/DAVIS/2017 \
    --weights models/cosine/mscosine_vos_fd1_md6_lr1e-4_ep50/pytorch_model/pytorch_model_24ep/pytorch_model.bin \
    --output outputs/repro/vos_full/mscosine_vos_fd1_md6_lr1e-4_ep50/24ep/D17 \
    --num_frame 6 \
    --memory_decay_type linear \
    --memory_decay_ratio 20 \
    --fix_first_frame

run_task \
  "vos_davis17_official_eval_24ep" 1000 \
  "outputs/repro/vos_full/mscosine_vos_fd1_md6_lr1e-4_ep50/24ep/D17_eval" \
  "${PYTHON_BIN}" inference_vos/davis2017-evaluation/evaluation_method.py \
    --davis_path datasets/vos/DAVIS/2017/trainval \
    --set val \
    --task semi-supervised \
    --results_path outputs/repro/vos_full/mscosine_vos_fd1_md6_lr1e-4_ep50/24ep/D17

run_task \
  "vos_youtube2019_val_24ep" 16000 \
  "outputs/repro/vos_full/mscosine_vos_fd1_md6_lr1e-4_ep50/24ep/Y19" \
  "${PYTHON_BIN}" tools/eval_vos_ms.py \
    --dataset Y19 \
    --y19_path datasets/vos/Youtube-VOS2019 \
    --weights models/cosine/mscosine_vos_fd1_md6_lr1e-4_ep50/pytorch_model/pytorch_model_24ep/pytorch_model.bin \
    --output outputs/repro/vos_full/mscosine_vos_fd1_md6_lr1e-4_ep50/24ep/Y19 \
    --num_frame 6 \
    --memory_decay_type linear \
    --memory_decay_ratio 20 \
    --fix_first_frame

log "representative full suite finished"
