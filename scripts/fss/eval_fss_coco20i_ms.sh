#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
cd "${REPO_ROOT}"

ep=49
data=coco
model=mscosine_unified_fd1_md6_lr1e-4_ep50
weight=${WEIGHT_ROOT}/${model}/pytorch_model_${ep}ep/pytorch_model.bin
model_out=${model}_${ep}ep

# text
for shot in 1 5;do
  for fold in 0 1 2 3;do

  output=${OUTPUT_ROOT}/fss/${model_out}/${data}/${shot}shot/fold${fold}_text
  echo ${output}

  CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON_BIN}" tools/eval_fss_ms.py  \
    --datapath "${FSS_DATA_ROOT}" \
    --benchmark ${data} \
    --fold ${fold} \
    --nshot ${shot} \
    --weights ${weight} \
    --log-root ${output} \
    --use_text
  done
done

# visual
for shot in 1 5;do
  for fold in 0 1 2 3;do

  output=${OUTPUT_ROOT}/fss/${model_out}/${data}/${shot}shot/fold${fold}_visual
  echo ${output}

  CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON_BIN}" tools/eval_fss_ms.py  \
    --datapath "${FSS_DATA_ROOT}" \
    --benchmark ${data} \
    --fold ${fold} \
    --nshot ${shot} \
    --weights ${weight} \
    --log-root ${output} \
    --use_visual
  done
done

# visual_text
for shot in 1 5;do
  for fold in 0 1 2 3;do

  output=${OUTPUT_ROOT}/fss/${model_out}/${data}/${shot}shot/fold${fold}_visual_text
  echo ${output}

  CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON_BIN}" tools/eval_fss_ms.py  \
    --datapath "${FSS_DATA_ROOT}" \
    --benchmark ${data} \
    --fold ${fold} \
    --nshot ${shot} \
    --weights ${weight} \
    --log-root ${output} \
    --use_text --use_visual
  done
done
