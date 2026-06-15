#!/bin/bash

model=mscosine_refseg_fd1_md6_lr1e-4_ep50
weight=${WEIGHT_ROOT:-models/cosine}/${model}/pytorch_model_19ep/pytorch_model.bin

infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test1
data_path="${DATA_ROOT:-datasets}/coco/annotations/instances_val2017.json"
image_dir="${DATA_ROOT:-datasets}/coco/val2017"
gt_dir="${DATA_ROOT:-datasets}/coco/panoptic_val2017"

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis2
# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir}

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis2-0.8
# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_catalpha 0.8

output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis2-1.0
CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
  --num-gpus 1 \
  --weights ${weight} \
  --output_dir ${output} \
  --infer_dir ${infer_dir} \
  --data_path ${data_path} \
  --image_dir ${image_dir} \
  --gt_dir ${gt_dir} \
  --prompts_catalpha 1.0


# weight=${WEIGHT_ROOT:-models/cosine}/${model}/pytorch_model_13ep/pytorch_model.bin
# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis2-13ep
# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir}