#!/bin/bash

TYPE=linear
RATIO=20
MODEL=cosine_fd1_md6_lr1e-4_ep50
WEIGHT=${WEIGHT_ROOT:-models/cosine}/${MODEL}/pytorch_model.bin
dataset=Y19
N_FRAME=6

LOG=${MODEL}_${N_FRAME}/${dataset}
echo ${LOG}

CUDA_VISIBLE_DEVICES=4 python tools/eval_vos_ss.py \
  --dataset ${dataset} \
  --weights ${WEIGHT} \
  --output outputs/vos/${LOG} \
  --num_frame ${N_FRAME} \
  --memory_decay_type ${TYPE} \
  --memory_decay_ratio ${RATIO} \
  --fix_first_frame
# python inference_vos/davis2017-evaluation/evaluation_method.py \
#   --davis_path datasets/vos/DAVIS/2017/trainval \
#   --set val \
#   --task semi-supervised \
#   --results_path outputs/vos/${LOG}
