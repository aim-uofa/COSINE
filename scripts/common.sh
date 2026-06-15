#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
WEIGHT_ROOT="${WEIGHT_ROOT:-${REPO_ROOT}/models/cosine}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"
FSS_DATA_ROOT="${FSS_DATA_ROOT:-${DATA_ROOT}/fss}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
GPU="${GPU:-0}"

mkdir -p "${OUTPUT_ROOT}"
