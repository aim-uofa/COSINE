import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.data import transforms as T
from detectron2.modeling.postprocessing import sem_seg_postprocess

import sys
sys.path.append("./")
from cosine.data.coco_ins import COCOInsDataset
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform
from segment_anything.utils.transforms import ResizeLongestSide

from cosine.model.model_ms_eval import build_model
# from cosine.model.model_ms_fcclip_eval import build_model


import matplotlib.pyplot as plt
from tqdm import tqdm

import random
from inference_fss.common import utils
random.seed(1)
utils.fix_randseed(1)


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Matcher Pytorch Implementation for One-shot Segmentation')

    # Dataset parameters
    parser.add_argument('--datapath', type=str, default='datasets_oss')
    parser.add_argument('--benchmark', type=str, default='coco',
                        choices=['fss', 'coco', 'pascal', 'lvis', 'paco_part', 'pascal_part'])
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=896)
    parser.add_argument('--pad-size', type=int, default=896)
    # parser.add_argument('--sam-size', type=int, default=1024)
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--log-root', type=str, default='outputs/vis/fsod_vis_text/lvis')
    parser.add_argument('--visualize', type=int, default=0)
    parser.add_argument('--vis', type=int, default=0)


     # Model parameters
    parser.add_argument('--feat_chans', default=256, type=int)
    parser.add_argument('--image_enc_use_fc', action="store_true")
    # parser.add_argument('--pt_model', type=str, default="dinov2")
    parser.add_argument('--dinov2-size', type=str, default="vit_large")
    parser.add_argument('--dinov2-weights', type=str, default="models/dinov2_vitl14_pretrain.pth")
    parser.add_argument('--weights', type=str, default="models/cosine/mscosine_unified_fd1_md6_lr1e-4_ep50/pytorch_model_49ep/pytorch_model.bin")
    parser.add_argument('--sam-size', type=str, default="vit_b")
    parser.add_argument('--sam-weights', type=str, default="models/sam_vit_b_01ec64.pth")
    parser.add_argument('--clip-weights', type=str, default="models/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin")
    parser.add_argument('--neck_in_features', default="p2||p3||p4||p5", type=str)
    parser.add_argument('--neck_encoder_in_features', default="p3||p4||p5", type=str)
    parser.add_argument('--neck_conv_dim', default=256, type=int)
    parser.add_argument('--neck_mask_dim', default=256, type=int)
    parser.add_argument('--neck_transformer_dropout', default=0., type=float)
    parser.add_argument('--neck_transformer_nheads', default=8, type=int)
    parser.add_argument('--neck_dim_feedforward', default=1024, type=int)
    parser.add_argument('--neck_encoder_layers', default=6, type=int)
    parser.add_argument('--neck_common_stride', default=4, type=int)

    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--transformer_nheads', default=8, type=int)
    parser.add_argument('--transformer_mlp_dim', default=2048, type=int)
    parser.add_argument('--transformer_mask_dim', default=256, type=int)
    parser.add_argument('--transformer_fusion_layer_depth', default=1, type=int)
    parser.add_argument('--transformer_num_queries', default=200, type=int)
    parser.add_argument("--transformer_pre_norm", action="store_true", default=True)
    parser.add_argument('--score_threshold', default=0.7, type=float)

    parser.add_argument('--class_weight', default=2.0, type=float)
    parser.add_argument('--mask_weight', default=5.0, type=float)
    parser.add_argument('--dice_weight', default=5.0, type=float)
    parser.add_argument('--no_object_weight', default=0.1, type=float)
    parser.add_argument('--train_num_points', default=12544, type=int)
    parser.add_argument('--oversample_ratio', default=3.0, type=float)
    parser.add_argument('--importance_sample_ratio', default=0.75, type=float)
    parser.add_argument("--deep_supervision", action="store_true", default=True)


    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Model initialization
    model = build_model(args)
    print(f"model: {model}")
    state_dict = torch.load(args.weights, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"msg: {msg}")
    model.to(device)
    model.eval()

    # Dataset initialization
    augmentation = [T.ResizeShortestEdge((args.img_size), args.img_size, "choice")]

    dino_transform = transforms.Compose([
            MaybeToTensor(),
            make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    dino_pixel_mean = [i*255 for i in [0.485, 0.456, 0.406]]
    dino_pixel_std = [i*255 for i in [0.229, 0.224, 0.225]]

    # sam_transform = ResizeLongestSide(args.sam_image_size)
    sam_pixel_mean = [123.675, 116.28, 103.53]
    sam_pixel_std = [58.395, 57.12, 57.375]

    dataset = COCOInsDataset(
        dataset_name="lvis_v1_val_ins",
        image_size=896,
        sam_image_size=1024,
        clip_image_size=1024,
        root='datasets',
        visual_prompt=True,
        text_prompt=False,
        use_all_classes=False,
        crop_ratio=0.,
        tfm_gens_crop_pair=augmentation,
        tfm_gens_sel_pair=augmentation,
        dino_transform=dino_transform,
    )
    dataset.set_multimodal(visual_prompt=True, text_prompt=True)

    model.semantic_on = False
    model.instance_on = True
    model.identity_on = False
    model.sem_seg_postprocess_before_inference = True
    model.score_threshold = 0.7

    def update_dict(dict_, vaild_idx):
        old_instance = dict_['instances']
        gt_boxes = old_instance.gt_boxes[vaild_idx]
        gt_classes = old_instance.gt_classes[vaild_idx]
        gt_masks = old_instance.gt_masks[vaild_idx]
        ins_ids = old_instance.ins_ids[vaild_idx]

        instances = Instances(old_instance.image_size)
        instances.gt_boxes = gt_boxes
        instances.gt_classes = gt_classes
        instances.gt_masks = gt_masks
        instances.ins_ids = ins_ids

        dict_['instances'] = instances

        return dict_

    save_root = args.log_root
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    max_num = 500
    for id in tqdm(range(max_num)):

        # if id > 500:
        #     break

        sample = dataset[id]
        output= model([sample])['ins_seg']

        if len(output) == 0:
            continue

        from visualization.visualizer import Visualizer as MaskVisualizer
        from detectron2.utils.visualizer import _OFF_WHITE

        query_name = sample['target']['file_name'].split('/')[-1].replace('.jpg', '')
        support_name = sample['visual_prompt']['file_name'].split('/')[-1].replace('.jpg', '')
        save_name = f"{support_name}_{query_name}"

        support_img = sample['visual_prompt']['image']
        support_img = (support_img * torch.Tensor(dino_pixel_std).view(-1, 1, 1)) + torch.Tensor(dino_pixel_mean).view(-1, 1, 1)
        h, w  = sample['visual_prompt']['instances'].image_size
        support_img = support_img[:, :h, :w]
        oh, ow = sample['visual_prompt']['height'], sample['visual_prompt']['width']
        support_img = F.interpolate(
            support_img[None, ...], (oh, ow), mode="bilinear", align_corners=False, antialias=True
        )[0]

        support_mask = sample['visual_prompt']['instances'].gt_masks
        support_mask = support_mask[:, :h, :w]
        support_mask = F.interpolate(
            support_mask[None, ...].float(), (oh, ow)
        )[0] > 0

        tgt_img = sample['target']['image']
        tgt_img = (tgt_img * torch.Tensor(dino_pixel_std).view(-1, 1, 1)) + torch.Tensor(dino_pixel_mean).view(-1, 1, 1)
        h, w  = sample['target']['instances'].image_size
        tgt_img = tgt_img[:, :h, :w]
        oh, ow = sample['target']['height'], sample['target']['width']
        tgt_img = F.interpolate(
            tgt_img[None, ...], (oh, ow), mode="bilinear", align_corners=False, antialias=True
        )[0]
        tgt_pred_mask = output.pred_masks > 0

        tgt_gt_mask = sample['target']['instances'].gt_masks
        tgt_gt_mask = tgt_gt_mask[:, :h, :w]
        tgt_gt_mask = F.interpolate(
            tgt_gt_mask[None, ...].float(), (oh, ow)
        )[0] > 0


        support_visualizer = MaskVisualizer(support_img.permute(1,2,0).cpu().numpy())
        tgt_pred_visualizer = MaskVisualizer(tgt_img.permute(1,2,0).cpu().numpy())
        tgt_gt_visualizer = MaskVisualizer(tgt_img.permute(1,2,0).cpu().numpy())

        support_vis_output = support_visualizer.draw_masks(
            support_mask.cpu().numpy(), random_color=True
        )
        tgt_pred_vis_output = tgt_pred_visualizer.draw_masks(
            tgt_pred_mask.cpu().numpy(), random_color=True
        )
        tgt_gt_vis_output = tgt_gt_visualizer.draw_masks(
            tgt_gt_mask.cpu().numpy(), random_color=True
        )

        support_vis_output.save(os.path.join(save_root, save_name+'_sup.jpg'))
        tgt_pred_vis_output.save(os.path.join(save_root, save_name+'_tgt_pred.jpg'))
        tgt_gt_vis_output.save(os.path.join(save_root, save_name+'_tgt_gt.jpg'))

        # ref_name = ref_dict['file_name'].split('/')[-1].split('.jpg')[0]
        # ref_path = os.path.join(args.log_root, f'{ref_name}')

        # if not os.path.exists(ref_path):
        #     os.makedirs(ref_path)

        # ref_labels = ref_dict['instances'].gt_classes
        # tar_labels = torch.cat([tar_dict['instances'].gt_classes for tar_dict in tar_dicts], dim=0)

        # unique_ref_labels = torch.unique(ref_labels)
        # unique_tar_labels = torch.unique(tar_labels)

        # union_labels = torch.isin(unique_ref_labels, unique_tar_labels)

        # vaild_labels = unique_ref_labels[union_labels]


        # ref_vaild_idx = torch.isin(ref_labels, vaild_labels)

        # ref_dict = update_dict(ref_dict, ref_vaild_idx)

        # # show_img(ref_dict, ref=True, save_path=ref_path)

        # for i, tar_dict in enumerate(tar_dicts):

        #     tar_label = tar_dict['instances'].gt_classes
        #     tar_vaild_idx = torch.isin(tar_label, vaild_labels)

        #     tar_dict = update_dict(tar_dict, tar_vaild_idx)

        #     img_dict = [{
        #         'ref_dict': ref_dict,
        #         'tar_dict': tar_dict,
        #     }]

        #     result = model(img_dict)['ins_seg']

        #     # tar_path = os.path.join(ref_path, f'{i}_gt')
        #     # if not os.path.exists(tar_path):
        #     #     os.makedirs(tar_path)

        #     # show_img(tar_dict, ref=True, save_path=tar_path)

        #     tar_path = os.path.join(ref_path, f'{i}_pred')
        #     if not os.path.exists(tar_path):
        #         os.makedirs(tar_path)

        #     result.gt_masks = result.pred_masks
        #     tar_dict['old_instances'] = tar_dict['instances']
        #     tar_dict['instances'] = result

        #     # show_img(tar_dict, ref=False, save_path=tar_path)