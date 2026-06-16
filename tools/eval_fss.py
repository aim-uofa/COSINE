r""" Matcher testing code for one-shot segmentation """
import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import transforms

from detectron2.structures import BitMasks, Boxes, Instances

import sys
sys.path.append("./")

from dinov2.data.transforms import MaybeToTensor, make_normalize_transform
from inference_fss.common.logger import Logger, AverageMeter
from inference_fss.common.vis_new import Visualizer
from inference_fss.common.evaluation import Evaluator
from inference_fss.common import utils
from inference_fss.data.dataset import FSSDataset
from inference_fss.model.model_single_scale import build_model

import random
random.seed(0)


def wrap_data(batch, args):
    def pad_img(x, pad_size):

        assert isinstance(x, torch.Tensor)
        # Pad
        h, w = x.shape[-2:]
        padh = pad_size - h
        padw = pad_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    # transforms for image encoder
    encoder_transform = transforms.Compose([
        MaybeToTensor(),
        make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    sam_pixel_mean = [i / 255. for i in [123.675, 116.28, 103.53]]
    sam_pixel_mean = torch.Tensor(sam_pixel_mean).view(-1, 1, 1)
    sam_pixel_std = [i / 255. for i in [58.395, 57.12, 57.375]]
    sam_pixel_std = torch.Tensor(sam_pixel_std).view(-1, 1, 1)

    clip_pixel_mean = [i / 255. for i in [122.7709383, 116.7460125, 104.09373615]]
    clip_pixel_mean = torch.Tensor(clip_pixel_mean).view(-1, 1, 1)
    clip_pixel_std = [i / 255. for i in [68.5005327, 66.6321579, 70.32316305]]
    clip_pixel_std = torch.Tensor(clip_pixel_std).view(-1, 1, 1)

    ref_list = []
    tar_dict = {}

    # visual prompts
    all_ref_imgs = batch['support_imgs'][0]
    all_ref_masks = batch['support_masks'][0]

    for ith, (ref_img, ref_mask) in enumerate(zip(all_ref_imgs, all_ref_masks)):
        ref_dict = {}
        ref_image_shape = ref_img.shape[-2:]
        ref_dict["image"] = pad_img(encoder_transform(ref_img), args.pad_size)

        # sam image
        ref_sam_image = ref_img
        ref_vaild = torch.ones_like(ref_sam_image)
        ref_sam_image_pad = pad_img(ref_sam_image, args.pad_size)
        ref_vaild_pad = pad_img(ref_vaild, args.pad_size)
        ref_sam_image_pad = F.interpolate(
            ref_sam_image_pad[None, ...], args.sam_image_size, mode="bilinear", align_corners=False, antialias=True
        )[0]
        ref_vaild_pad = F.interpolate(
            ref_vaild_pad[None, ...], args.sam_image_size, mode='nearest'
        )[0]
        ref_sam_image_pad = (ref_sam_image_pad - sam_pixel_mean.to(ref_sam_image_pad.device)) / sam_pixel_std.to(
            ref_sam_image_pad.device)
        ref_dict["sam_image"] = ref_sam_image_pad * ref_vaild_pad

        # clip image
        ref_clip_image = ref_img
        ref_vaild = torch.ones_like(ref_clip_image)
        ref_clip_image_pad = pad_img(ref_clip_image, args.pad_size)
        ref_vaild_pad = pad_img(ref_vaild, args.pad_size)
        ref_clip_image_pad = F.interpolate(
            ref_clip_image_pad[None, ...], args.clip_image_size, mode="bilinear", align_corners=False, antialias=True
        )[0]
        ref_vaild_pad = F.interpolate(
            ref_vaild_pad[None, ...], args.clip_image_size, mode='nearest'
        )[0]
        ref_clip_image_pad = (ref_clip_image_pad - clip_pixel_mean.to(ref_clip_image_pad.device)) / clip_pixel_std.to(
            ref_clip_image_pad.device)
        ref_dict["clip_image"] = ref_clip_image_pad * ref_vaild_pad


        # label
        ref_dict['height'], ref_dict['width'] = ref_image_shape
        ref_instances = Instances(ref_image_shape)
        ref_instances.gt_classes = torch.tensor([batch['class_id'].item()], dtype=torch.int64)
        ref_masks = pad_img(ref_mask[None, ...], args.pad_size)
        ref_masks = BitMasks(ref_masks)
        ref_instances.gt_masks = ref_masks.tensor
        ref_instances.gt_boxes = ref_masks.get_bounding_boxes()
        ref_instances.ins_ids = torch.tensor([ith], dtype=torch.int64)
        ref_dict["instances"] = ref_instances
        ref_list.append(ref_dict)

    # target
    tar_img = batch['query_img'][0]
    tar_image_shape = tar_img.shape[-2:]  # h, w
    tar_dict["image"] = pad_img(encoder_transform(tar_img), args.pad_size)

    # sam image
    tar_sam_image = tar_img
    tar_vaild = torch.ones_like(tar_sam_image)
    tar_sam_image_pad = pad_img(tar_sam_image, args.pad_size)
    tar_vaild_pad = pad_img(tar_vaild, args.pad_size)
    tar_sam_image_pad = F.interpolate(
        tar_sam_image_pad[None, ...], args.sam_image_size, mode="bilinear", align_corners=False, antialias=True
    )[0]
    tar_vaild_pad = F.interpolate(
        tar_vaild_pad[None, ...], args.sam_image_size, mode='nearest'
    )[0]
    tar_sam_image_pad = (tar_sam_image_pad - sam_pixel_mean.to(tar_sam_image_pad.device)) / sam_pixel_std.to(
        tar_sam_image_pad.device)
    tar_dict["sam_image"] = tar_sam_image_pad * tar_vaild_pad

    # clip image
    tar_clip_image = tar_img
    tar_vaild = torch.ones_like(tar_clip_image)
    tar_clip_image_pad = pad_img(tar_clip_image, args.pad_size)
    tar_vaild_pad = pad_img(tar_vaild, args.pad_size)
    tar_clip_image_pad = F.interpolate(
        tar_clip_image_pad[None, ...], args.clip_image_size, mode="bilinear", align_corners=False, antialias=True
    )[0]
    tar_vaild_pad = F.interpolate(
        tar_vaild_pad[None, ...], args.clip_image_size, mode='nearest'
    )[0]
    tar_clip_image_pad = (tar_clip_image_pad - clip_pixel_mean.to(tar_clip_image_pad.device)) / clip_pixel_std.to(
        tar_clip_image_pad.device)
    tar_dict["clip_image"] = tar_clip_image_pad * tar_vaild_pad


    # label
    tar_dict['height'], tar_dict['width'] = tar_image_shape
    tar_instances = Instances(tar_image_shape)
    tar_instances.gt_classes = torch.tensor([batch['class_id'].item()], dtype=torch.int64)
    tar_masks = pad_img(batch['query_mask'], args.pad_size)
    tar_masks = BitMasks(tar_masks)
    tar_instances.gt_masks = tar_masks.tensor
    tar_instances.gt_boxes = tar_masks.get_bounding_boxes()
    tar_instances.ins_ids = torch.tensor([1], dtype=torch.int64)
    tar_dict["instances"] = tar_instances

    tar_list = [tar_dict] + [None for i in range(len(ref_list) - 1)]


    # text prompt
    if args.use_text:
        text = {batch['class_id'].item():batch['class_name'][0]}
    else:
        text = None

    if not args.use_visual:
        tag = 'sem'
    else:
        tag = 'visual'

    data = []
    for ref, tar in zip(ref_list, tar_list):
        data.append({
            'visual_prompt': ref,
            'target': tar,
            'text_prompt': text,
            'tag': tag
        })

    return data


def test(model, dataloader, args=None):
    r""" Test Matcher """

    # Freeze randomness during testing for reproducibility
    # Follow HSNet
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        batch = utils.to_cuda(batch)

        wrap_batch = wrap_data(batch, args)

        # Forward
        res = model(wrap_batch)["sem_seg"]
        res = res > model.score_threshold
        pred_mask = res.float()

        assert pred_mask.size() == batch['query_mask'].size(), \
            'pred {} ori {}'.format(pred_mask.size(), batch['query_mask'].size())

        # Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou, _ = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='COSINE PyTorch Implementation for Few-shot Segmentation')

    # Dataset parameters
    parser.add_argument('--datapath', type=str, default='datasets/fss')
    parser.add_argument('--benchmark', type=str, default='coco',
                        choices=['fss', 'coco', 'pascal', 'lvis', 'paco_part', 'pascal_part', 'isaid', 'deepglobe', 'isic', 'lung', 'vision'])
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=518)
    parser.add_argument('--pad-size', type=int, default=896)
    parser.add_argument('--sam_image_size', type=int, default=1024)
    parser.add_argument('--clip_image_size', type=int, default=1024)
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--log-root', type=str, default='outputs/fss/debug')
    parser.add_argument('--visualize', type=int, default=0)

    # Model parameters
    parser.add_argument('--feat_chans', default=256, type=int)
    parser.add_argument('--image_enc_use_fc', action="store_true")
    # parser.add_argument('--pt_model', type=str, default="dinov2")
    parser.add_argument('--dinov2-size', type=str, default="vit_large")
    parser.add_argument('--dinov2-weights', type=str, default="models/dinov2_vitl14_pretrain.pth")
    parser.add_argument('--weights', type=str, default="models/cosine/cosine_fd1_md6_lr1e-4_ep50/pytorch_model.bin")
    parser.add_argument('--sam-size', type=str, default="vit_b")
    parser.add_argument('--sam-weights', type=str, default="models/sam_vit_b_01ec64.pth")
    parser.add_argument('--clip-weights', type=str, default="models/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin")

    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--transformer_nheads', default=8, type=int)
    parser.add_argument('--transformer_mlp_dim', default=2048, type=int)
    parser.add_argument('--transformer_mask_dim', default=256, type=int)
    parser.add_argument('--transformer_fusion_layer_depth', default=1, type=int)
    parser.add_argument('--transformer_num_queries', default=200, type=int)
    parser.add_argument("--transformer_pre_norm", action="store_true", default=True)
    parser.add_argument('--score_threshold', default=0.7, type=float)
    parser.add_argument("--use_text", action="store_true")
    parser.add_argument("--use_visual", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    Logger.initialize(args, root=args.log_root)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Model initialization
    model = build_model(args)
    print(f"model: {model}")
    state_dict = torch.load(args.weights, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"msg: {msg}")
    model.to(device)
    model.eval()

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize, root=args.log_root)

    # Dataset initialization
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)
    print("dataset size: {}".format(len(dataloader_test)))

    # Test COSINE
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args=args)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')