#!/bin/bash

model=mscosine_refseg_fd1_md6_lr1e-4_ep50
weight=${WEIGHT_ROOT:-models/cosine}/${model}/pytorch_model_19ep/pytorch_model.bin

output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis1
infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test1
data_path="${DATA_ROOT:-datasets}/coco/annotations/instances_val2017.json"
image_dir="${DATA_ROOT:-datasets}/coco/val2017"
gt_dir="${DATA_ROOT:-datasets}/coco/panoptic_val2017"

# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir}

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis1-0.1
# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_catalpha 0.1

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis1-0.3
# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_catalpha 0.3

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis1-0.0
# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_catalpha 0.0

# weight=${WEIGHT_ROOT:-models/cosine}/${model}/pytorch_model_13ep/pytorch_model.bin
# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis1-0.0-13ep
# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_catalpha 0.0

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis1-red
# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir}

infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test2
# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis2-red-0.0
# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_catalpha 0.0 \
#   --prompts_cls "elephant" \
#   --prompts_txt "red toys like this one"

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis2-red-elephant-0.5
# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_catalpha 0.5 \
#   --prompts_cls "elephant" \
#   --prompts_txt "red toys like this one"

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis2-yellow-elephant-0.2
# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_catalpha 0.2 \
#   --prompts_cls "elephant" \
#   --prompts_txt "yellow toys like this one"

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis2-red-car-0.5
# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_catalpha 0.5 \
#   --prompts_cls "car" \
#   --prompts_txt "red toys like this one"

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis2-yellow-car-0.3
# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_catalpha 0.3 \
#   --prompts_cls "car" \
#   --prompts_txt "yellow toys like this one"

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis2-yellow-car-0.4
# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_catalpha 0.4 \
#   --prompts_cls "car" \
#   --prompts_txt "yellow toys like this one"

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis2-yellow-car-onlytext
# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_catalpha 0.0 \
#   --prompts_cls "car" \
#   --prompts_txt "yellow toys like car"

output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis2-yellow-elephant-onlytext
CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all.py \
  --num-gpus 1 \
  --weights ${weight} \
  --output_dir ${output} \
  --infer_dir ${infer_dir} \
  --data_path ${data_path} \
  --image_dir ${image_dir} \
  --gt_dir ${gt_dir} \
  --prompts_catalpha 0.0 \
  --prompts_cls "car" \
  --prompts_txt "yellow toys like elephant"