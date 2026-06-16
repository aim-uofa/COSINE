#!/bin/bash

weight=${WEIGHT_ROOT:-models/cosine}/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin

output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin5_cc_onlytxt
infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin5_cc
data_path=home
image_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/image
gt_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/label

CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
  --num-gpus 1 \
  --weights ${weight} \
  --output_dir ${output} \
  --infer_dir ${infer_dir} \
  --data_path ${data_path} \
  --image_dir ${image_dir} \
  --gt_dir ${gt_dir} \
  --prompts_txt "the right defects" \
  --prompts_cls "defects" \
  --prompts_catalpha 0.0 \
  --vis_font_scale 2.3

output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin5_cc_onlyvis
infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin5_cc
data_path=home
image_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/image
gt_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/label

# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_txt "the right defects" \
#   --prompts_cls "defects" \
#   --prompts_catalpha 1.0 \
#   --vis_notext

output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin5_cc
infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin5_cc
data_path=home
image_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/image
gt_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/label

CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
  --num-gpus 1 \
  --weights ${weight} \
  --output_dir ${output} \
  --infer_dir ${infer_dir} \
  --data_path ${data_path} \
  --image_dir ${image_dir} \
  --gt_dir ${gt_dir} \
  --prompts_txt "the right defects like this" \
  --prompts_cls "defects" \
  --prompts_catalpha 0.7 \
  --vis_font_scale 2.3
