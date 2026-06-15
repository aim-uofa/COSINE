#!/bin/bash

TYPE=linear
RATIO=20
MODEL=cosine_fd1_md6_lr1e-4_ep50
WEIGHT=${WEIGHT_ROOT:-models/cosine}/${MODEL}/pytorch_model.bin
dataset=D16
N_FRAME=6

LOG=${MODEL}_1036_30e_1110_${N_FRAME}_fixfirst_${dataset}
echo ${LOG}

CUDA_VISIBLE_DEVICES=5 python tools/eval_vos.py \
  --dataset ${dataset} \
  --weights ${WEIGHT} \
  --output outputs/vos/${dataset}/${LOG} \
  --img-size 1036 \
  --pad-size 1036 \
  --num_frame ${N_FRAME} \
  --memory_decay_type ${TYPE} \
  --memory_decay_ratio ${RATIO} \
  --fix_first_frame
python inference_vos/davis2017-evaluation/evaluation_method.py \
  --davis_path datasets/vos/DAVIS17/trainval \
  --set val \
  --year 2016 \
  --task semi-supervised \
  --results_path outputs/vos/${dataset}/${LOG}
