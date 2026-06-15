#!/usr/bin/env bash
set -euo pipefail

WEIGHT_ROOT="${WEIGHT_ROOT:-models/cosine}"
DATA_ROOT="${DATA_ROOT:-datasets}"
FSOD_DATA_ROOT="${FSOD_DATA_ROOT:-inference_fsod/datasets}"
WEIGHTS_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --weights-only) WEIGHTS_ONLY=1 ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

missing=0

check_path() {
  local path="$1"
  local label="$2"
  if [[ -e "$path" ]]; then
    printf '[ok] %s: %s\n' "$label" "$path"
  else
    printf '[missing] %s: %s\n' "$label" "$path" >&2
    missing=1
  fi
}

check_path "${WEIGHT_ROOT}/cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin" "FSS single-scale checkpoint"
check_path "${WEIGHT_ROOT}/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin" "FSS/FSOD multi-scale checkpoint"
check_path "${WEIGHT_ROOT}/mscosine_refseg_fd1_md6_lr1e-4_ep50/pytorch_model_19ep/pytorch_model.bin" "RefSeg checkpoint"
check_path "${WEIGHT_ROOT}/mscosine_vos_fd1_md6_lr1e-4_ep50/pytorch_model/pytorch_model_24ep/pytorch_model.bin" "VOS checkpoint"

if [[ "$WEIGHTS_ONLY" -eq 0 ]]; then
  check_path "models/dinov2_vitl14_pretrain.pth" "DINOv2 ViT-L backbone"
  check_path "models/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin" "OpenCLIP ConvNeXt-L backbone"
  check_path "${DATA_ROOT}/fss/COCO2014" "FSS COCO-20i images"
  check_path "${DATA_ROOT}/fss/LVIS" "FSS LVIS-92i data"
  check_path "${DATA_ROOT}/refer_seg" "RefSeg annotations"
  check_path "${DATA_ROOT}/vos/DAVIS/2017" "DAVIS 2017"
  check_path "${DATA_ROOT}/vos/Youtube-VOS2019" "YouTube-VOS 2019"
  check_path "${FSOD_DATA_ROOT}/coco" "FSOD COCO dataset"
  check_path "${FSOD_DATA_ROOT}/coco2017" "FSOD COCO 2017 alias"
  check_path "${FSOD_DATA_ROOT}/lvis" "FSOD LVIS dataset"
  check_path "${FSOD_DATA_ROOT}/lvissplit" "FSOD LVIS few-shot split metadata"
fi

exit "$missing"
