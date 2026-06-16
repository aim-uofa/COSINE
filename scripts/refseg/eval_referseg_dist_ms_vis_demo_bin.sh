#!/bin/bash

model=mscosine_refseg_fd1_md6_lr1e-4_ep50
weight=${WEIGHT_ROOT:-models/cosine}/${model}/pytorch_model_19ep/pytorch_model.bin

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin
# infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin
# # infer_dir=${DATA_ROOT:-datasets}/fss/LungSegmentation/CXR_png
# data_path=home
# image_dir=${DATA_ROOT:-datasets}/fss/LungSegmentation/CXR_png
# gt_dir=${DATA_ROOT:-datasets}/fss/LungSegmentation/masks

# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir}


# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin3
# infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin3
# data_path=home
# image_dir=${DATA_ROOT:-datasets}/fss/ISIC/ISIC2018_Task1-2_Training_Input
# gt_dir=${DATA_ROOT:-datasets}/fss/ISIC/ISIC2018_Task1_Training_GroundTruth

# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_txt "nevus like this" \
#   --prompts_cls "nevus"

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin4
# infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin4
# data_path=home
# image_dir=${DATA_ROOT:-datasets}/fss/iSAID/iSAID_patches/val/images
# gt_dir=${DATA_ROOT:-datasets}/fss/iSAID/iSAID_patches/val/semantic_mask

# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_txt "truck" \
#   --prompts_cls "truck"

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin5
# infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin5
# data_path=home
# image_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/image
# gt_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/label

# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_txt "defects like this" \
#   --prompts_cls "defects"

# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_txt "defects like this" \
#   --prompts_cls "defects" \
#   --prompts_catalpha 0.8




weight=${WEIGHT_ROOT:-models/cosine}/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin5_0.8
# infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin5
# data_path=home
# image_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/image
# gt_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/label

# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_txt "defects like this" \
#   --prompts_cls "defects" \
#   --prompts_catalpha 0.8

output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin3
infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin3
data_path=home
image_dir=${DATA_ROOT:-datasets}/fss/ISIC/ISIC2018_Task1-2_Training_Input
gt_dir=${DATA_ROOT:-datasets}/fss/ISIC/ISIC2018_Task1_Training_GroundTruth

CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
  --num-gpus 1 \
  --weights ${weight} \
  --output_dir ${output} \
  --infer_dir ${infer_dir} \
  --data_path ${data_path} \
  --image_dir ${image_dir} \
  --gt_dir ${gt_dir} \
  --prompts_txt "nevus like this" \
  --prompts_cls "nevus" \
  --vis_font_scale 3.5

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin4
# infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin4
# data_path=home
# image_dir=${DATA_ROOT:-datasets}/fss/iSAID/iSAID_patches/val/images
# gt_dir=${DATA_ROOT:-datasets}/fss/iSAID/iSAID_patches/val/semantic_mask

# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_txt "objects like this" \
#   --prompts_cls "truck" \
#   --vis_font_scale 1.0

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin4_1
# infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin4_1
# data_path=home
# image_dir=${DATA_ROOT:-datasets}/fss/iSAID/iSAID_patches/val/images
# gt_dir=${DATA_ROOT:-datasets}/fss/iSAID/iSAID_patches/val/semantic_mask

# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_txt "objects like this" \
#   --prompts_cls "airplane" \
#   --vis_font_scale 1.0 \
#   --prompts_catalpha 0.4

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin4_2
# infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin4_2
# data_path=home
# image_dir=${DATA_ROOT:-datasets}/fss/iSAID/iSAID_patches/val/images
# gt_dir=${DATA_ROOT:-datasets}/fss/iSAID/iSAID_patches/val/semantic_mask

# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_txt "the top airplane" \
#   --prompts_cls "airplane" \
#   --vis_font_scale 1.0 \
#   --prompts_catalpha 0.0

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin4_3
# infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin4_3
# data_path=home
# image_dir=${DATA_ROOT:-datasets}/fss/iSAID/iSAID_patches/val/images
# gt_dir=${DATA_ROOT:-datasets}/fss/iSAID/iSAID_patches/val/semantic_mask

# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_txt "boat" \
#   --prompts_cls "car" \
#   --vis_font_scale 1.0 \
#   --prompts_catalpha 0.0

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin5_cc_onlytxt
# infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin5_cc
# data_path=home
# image_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/image
# gt_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/label

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
#   --prompts_catalpha 0.0

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin5_cc_onlyvis
# infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin5_cc
# data_path=home
# image_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/image
# gt_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/label

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

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin5_cc
# infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin5_cc
# data_path=home
# image_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/image
# gt_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Cable/thunderbolt/label

# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_txt "the right defects like this" \
#   --prompts_cls "defects" \
#   --prompts_catalpha 0.7


# model=mscosine_refseg_fd1_md6_lr1e-4_ep50
# weight=${WEIGHT_ROOT:-models/cosine}/${model}/pytorch_model_19ep/pytorch_model.bin

# output=${OUTPUT_ROOT:-outputs/visualization}/combseg/vis_bin5_1
# infer_dir=${OUTPUT_ROOT:-outputs/visualization}/combseg/test_bin5_1
# data_path=home
# image_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Screw/front/image
# gt_dir=${DATA_ROOT:-datasets}/fss/test-data-for-ZJU/Screw/front/label

# CUDA_VISIBLE_DEVICES=0 python tools/eval_combseg_ms_vis_all_bin.py \
#   --num-gpus 1 \
#   --weights ${weight} \
#   --output_dir ${output} \
#   --infer_dir ${infer_dir} \
#   --data_path ${data_path} \
#   --image_dir ${image_dir} \
#   --gt_dir ${gt_dir} \
#   --prompts_txt "thread damage" \
#   --prompts_cls "defects2" \
#   --prompts_catalpha 1.0