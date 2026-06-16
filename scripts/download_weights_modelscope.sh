#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-zzzmmz/COSINE-Public-Weights}"
LOCAL_DIR="${LOCAL_DIR:-models/modelscope-cosine-weights}"
WEIGHT_ROOT="${WEIGHT_ROOT:-models/cosine}"

if ! command -v ms >/dev/null 2>&1; then
  echo "Missing ModelScope CLI command: ms" >&2
  echo "Install ModelScope first, then rerun this script." >&2
  exit 1
fi

if [[ -n "${MODELSCOPE_TOKEN:-}" ]]; then
  if command -v modelscope >/dev/null 2>&1; then
    modelscope login --token "${MODELSCOPE_TOKEN}"
  else
    echo "MODELSCOPE_TOKEN is set, but modelscope login command is unavailable; continuing with current ms auth." >&2
  fi
fi

mkdir -p "${LOCAL_DIR}" "${WEIGHT_ROOT}"

release_files=(
  "README.md"
  "MANIFEST.txt"
  "SIZES.txt"
  "SHA256SUMS.txt"
  "weights/cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin"
  "weights/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin"
  "weights/mscosine_refseg_fd1_md6_lr1e-4_ep50/pytorch_model_19ep/pytorch_model.bin"
  "weights/mscosine_vos_fd1_md6_lr1e-4_ep50/pytorch_model/pytorch_model_24ep/pytorch_model.bin"
)

ms download "${MODEL_REPO}" "${release_files[@]}" \
  --repo-type model \
  --local-dir "${LOCAL_DIR}"

if [[ ! -d "${LOCAL_DIR}/weights" ]]; then
  echo "Expected downloaded weights at ${LOCAL_DIR}/weights, but that directory does not exist." >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1 && [[ -f "${LOCAL_DIR}/SHA256SUMS.txt" ]]; then
  (cd "${LOCAL_DIR}" && sha256sum -c SHA256SUMS.txt)
fi

rsync -a "${LOCAL_DIR}/weights/" "${WEIGHT_ROOT}/"

"$(dirname "$0")/check_required_assets.sh" --weights-only
