r""" Matcher testing code for one-shot segmentation """
import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from panopticapi.utils import rgb2id
import json
import PIL.Image as Image
import numpy as np
import pickle
from tqdm import tqdm
import cv2

from detectron2.structures import BitMasks, Boxes, Instances, ImageList
from detectron2.data import detection_utils
from detectron2.layers import nms

import sys
sys.path.append("./")

from dinov2.data.transforms import MaybeToTensor, make_normalize_transform
# from inference_fss.common.logger import Logger, AverageMeter
# from inference_fss.common.vis_new import Visualizer
# from inference_fss.common.evaluation import Evaluator
from inference_referseg.utils import utils
# from inference_fss.data.dataset import FSSDataset
from inference_referseg.model.model_ms_comb import build_model
from eval_combseg_ms_readdata import inf_read_data
from segment_anything.utils.transforms import ResizeLongestSide
from visualization.visualizer_ade import Visualizer_ade

import random
random.seed(0)

COLORS = {
    "red": (0.333, 0, 0),
    "yellow": (1, 1, 0.333),
}

# def get_ref_masks(args):
#     data_path = args.data_path
#     data = json.load(open(data_path))
#     obj_anno = []
#     img_anno = []
#     imgid_imganno = {a["id"]: a for a in data["images"]}

#     for a in data["annotations"]:
#         if a["id"] == "580125":
#             a_new = a.copy()
#             a_imganno = imgid_imganno[a["image_id"]]
#             a_new["file_name"] = a_imganno["file_name"]
#             obj_anno.append(a_new)
#             img_anno.append(a_imganno)

#     assert len(obj_anno) * len(img_anno) > 0
#     return {
#         "images": img_anno,
#         "annotations": obj_anno
#     }

def get_ref_masks(args):
    obj_dict = {
        "elephant": {'id': 3157566, 'category_id': 22, 'iscrowd': 0, 'bbox': [5, 110, 314, 277], 'area': 44219, 'file_name': '000000021903.png'},
        # "car": {'id': 9735305, 'category_id': 3, 'iscrowd': 0, 'bbox': [221, 8, 26, 5], 'area': 60, 'file_name': '000000019042.png'}
        # "car": {'id': 8950931, 'category_id': 3, 'iscrowd': 0, 'bbox': [572, 248, 28, 19], 'area': 412, 'file_name': '000000005037.png'}
        # "car": {'id': 6511741, 'category_id': 3, 'iscrowd': 0, 'bbox': [522, 260, 81, 48], 'area': 2411, 'file_name': '000000026204.png'}
        # "car": {'id': 8221038, 'category_id': 3, 'iscrowd': 0, 'bbox': [133, 261, 97, 76], 'area': 6053, 'file_name': '000000084170.png'}
        "car": {'id': 9342875, 'category_id': 3, 'iscrowd': 0, 'bbox': [83, 263, 59, 41], 'area': 1970, 'file_name': '000000084170.png'}
    }
    obj_anno = obj_dict[args.prompts_cls]
    return {
        "annotations": [obj_anno]
    }


def wrap_data_vis(batch, args):
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

    clip_pixel_mean=[122.7709383, 116.7460125, 104.09373615]
    clip_pixel_std=[68.5005327, 66.6321579, 70.32316305]
    clip_pixel_mean = torch.Tensor(clip_pixel_mean).view(-1, 1, 1)
    clip_pixel_std = torch.Tensor(clip_pixel_std).view(-1, 1, 1)

    ref_list = []
    tar_dict = {}

    # visual prompts
    all_ref_imgs = batch['support_imgs']
    all_ref_masks = batch['support_masks']
    all_anno_cls = batch['anno_cls']

    for ith, (ref_img, ref_mask) in enumerate(zip(all_ref_imgs, all_ref_masks)):

        ref_dict = {}
        # ref_image_shape = ref_img.shape[-2:]
        ref_image_shape = args.image_size
        ref_dict["image"] = pad_img(encoder_transform(ref_img), args.image_size)

        # clip image
        ref_clip_image = torch.as_tensor(np.ascontiguousarray(ref_img.transpose(2, 0, 1))).to(dtype=torch.float32)
        ref_vaild = torch.ones_like(ref_clip_image)
        ref_clip_image_pad = pad_img(ref_clip_image, args.image_size)
        ref_vaild_pad = pad_img(ref_vaild, args.image_size)
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
        ref_dict['height'], ref_dict['width'] = args.image_size, args.image_size
        ref_instances = Instances(ref_image_shape)
        ref_instances.gt_classes = torch.tensor([int(all_anno_cls[ith])], dtype=torch.int64)
        ref_masks = pad_img(ref_mask[None, ...], args.image_size)
        ref_masks = BitMasks(ref_masks)
        ref_instances.gt_masks = ref_masks.tensor
        ref_instances.gt_boxes = ref_masks.get_bounding_boxes()
        ref_instances.ins_ids = torch.tensor([ith], dtype=torch.int64)
        ref_dict["instances"] = ref_instances
        ref_list.append(ref_dict)

        # print(ref_dict["image"].shape, ref_masks.tensor.shape)

    text = None

    data = []
    for ref in ref_list:
        data.append({
            'visual_prompt': ref,
            # 'target': tar,
            'text_prompt': text,
            'tag': 'visual'
        })

    return data


def read_data(ref_data_cls, data_info, image_size, args):
    all_data = {}
    image_dir = data_info["image_dir"]
    gt_dir = data_info["gt_dir"]
    support_imgs = []
    support_masks = []
    anno_cls = []

    # ori_transform = transforms.Compose([
    #         transforms.Resize(size=(img_size, img_size)),
    #         transforms.ToTensor()
    #     ])
    ori_transform = ResizeLongestSide(image_size)

    for ind, ann in enumerate(ref_data_cls["annotations"]):
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        pan_image = detection_utils.read_image(image_file, "RGB")
        pan_image_ori = pan_image.copy()
        pan_image = ori_transform.apply_image(pan_image)
        pan_seg_gt = detection_utils.read_image(label_file, "RGB")
        pan_seg_gt = rgb2id(pan_seg_gt)
        classes = []
        mask = pan_seg_gt == ann["id"]
        assert mask.sum() > 0

        new_size = (pan_image.shape[0], pan_image.shape[1])
        smask = torch.from_numpy(np.ascontiguousarray(mask.copy()))
        smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), new_size, mode='nearest').squeeze()


        support_imgs.append(pan_image)
        support_masks.append(smask)
        anno_cls.append(ann["category_id"])

        save_result_path = os.path.join(args.output_dir, f"ref_{ind}.jpg")
        v = Visualizer_ade(pan_image_ori, font_size_scale=2.0)
        out = v.draw_sem_seg([mask])
        print(save_result_path)
        cv2.imwrite(save_result_path, cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR))

    # support_imgs = torch.stack([ori_transform(support_img) for support_img in support_imgs])
    # support_imgs = torch.stack(support_imgs)
    # for midx, smask in enumerate(support_masks):
    #     smask = torch.from_numpy(np.ascontiguousarray(smask.copy()))
    #     support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
    # support_masks = torch.stack(support_masks)

    all_data['support_imgs'] = support_imgs
    all_data['support_masks'] = support_masks
    all_data['anno_cls'] = anno_cls
    return all_data



def generate_prompts(model, ref_data, txt_data, data_info, args=None):
    r""" Test Matcher """

    # Freeze randomness during testing for reproducibility
    # Follow HSNet

    all_data = read_data(ref_data, data_info, args.image_size, args)
    all_data = utils.to_cuda(all_data)
    # batch = utils.to_cuda(batch)

    wrap_batch_vis = wrap_data_vis(all_data, args)

    batched_visual_prompt = [item['visual_prompt'] for item in wrap_batch_vis if item['visual_prompt'] is not None]
    batched_text_prompt = [item['text_prompt'] for item in wrap_batch_vis]
    batched_tag = [item['tag'] for item in wrap_batch_vis]
    vis_prompts_all = model.encoder_soup.encode_prompt(batched_visual_prompt, batched_text_prompt, batched_tag)
    vis_prompts = vis_prompts_all['sem_prompt_list'][0].clone()

    batched_visual_prompt = [None]
    batched_text_prompt = [[{"id": 0, "caption": txt_data}]]
    batched_tag = ["refer"]
    txt_prompts_all = model.encoder_soup.encode_prompt(batched_visual_prompt, batched_text_prompt, batched_tag)
    txt_prompts = txt_prompts_all['id_prompt_list'][0].clone()

    if args.prompts_cattype == "avg":
        alpha = args.prompts_catalpha
        print(alpha)
        final_prompts = alpha * vis_prompts + (1-alpha) * txt_prompts
    else:
        raise RuntimeError

    txt_prompts_all['id_prompt_list'] = [final_prompts]

    # res_file = args.ref_img_json.split('.')[0] + args.out_name
    # with open(res_file, "wb") as f:
    #     pickle.dump(all_prompts, f)
    # print(f"Saved in {res_file}")
    return txt_prompts_all


def test(model, ref_data, txt_data_all, data_info, args=None):
    dataset_dicts = inf_read_data(args.infer_dir)
    assert len(dataset_dicts) == 1
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_dict = dataset_dicts[0]
    dataset_dict = utils.to_cuda(dataset_dict)
    file_name = dataset_dict["file_name"]
    file_path = dataset_dict["file_path"]
    save_result_path = os.path.join(args.output_dir, file_name)
    assert os.path.exists(file_path), file_path
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    v = Visualizer_ade(image, font_size_scale=args.vis_font_scale)

    pred_masks_all = []
    prompt_list_all = []

    txt_data_list = txt_data_all.split(";")
    for txt_data in txt_data_list:
        txt_prompts_all = generate_prompts(model, ref_data, txt_data, data_info, args)
        print("Get text prompts!")

        batched_inputs = {
            "target": dataset_dict,
            "txt_prompts_all": txt_prompts_all,
            "tag": "refer",
        }
        output_dict = model([batched_inputs])

        pred_masks = output_dict["id_seg"].pred_masks
        pred_boxes = BitMasks(pred_masks > 0).get_bounding_boxes()
        pred_scores = torch.rand(len(pred_boxes))
        nms_results = nms(pred_boxes.tensor, pred_scores, 0.3)
        pred_masks = pred_masks[nms_results]

        prompt_list = [txt_data for _ in range(len(pred_masks))]
        print(prompt_list)
        pred_masks_all.append(pred_masks)
        prompt_list_all += prompt_list

    pred_masks_all = torch.cat(pred_masks_all, dim=0)

    if len(args.choose_color) > 0:
        color_list = args.choose_color.split(";")
        color_list_all = [COLORS[c] for c in color_list]
    else:
        color_list_all = None

    out = v.draw_sem_seg(pred_masks_all.to("cpu"), prompt_list_all, color_list_all)

    print(save_result_path)
    cv2.imwrite(save_result_path, cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='COSINE PyTorch Implementation for Few-shot Segmentation')

    # Dataset parameters
    parser.add_argument('--datapath', type=str, default='datasets/fss')
    parser.add_argument('--benchmark', type=str, default='coco',
                        choices=['fss', 'coco', 'pascal', 'lvis', 'paco_part', 'pascal_part'])
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--log-root', type=str, default='outputs/fss/debug')
    parser.add_argument('--visualize', type=int, default=0)

    # Model parameters
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--data_root', default="datasets", type=str)
    parser.add_argument('--dataset', default="pano_seg", type=str)
    parser.add_argument('--sample_rate', default="1", type=str)
    parser.add_argument('--pano_seg_data', default="coco", type=str)
    parser.add_argument('--pano_sample_rate', default="1", type=str)
    parser.add_argument('--ins_seg_data', default="coco||paco||o365", type=str)
    parser.add_argument('--ins_sample_rate', default="1,1,1", type=str)
    parser.add_argument('--refer_seg_data', default="refclef", type=str)
    parser.add_argument('--refer_sample_rate', default="1", type=str)
    parser.add_argument('--multimodal_choice', default="visual_text", type=str) # 'visual', 'text', 'visual_text'
    parser.add_argument('--multimodal_rate', default="1", type=str)
    parser.add_argument('--use_all_classes', action='store_true')
    parser.set_defaults(use_all_classes=True)

    parser.add_argument('--random_flip', default="horizontal", type=str)
    parser.add_argument('--min_scale', default=0.1, type=float)
    parser.add_argument('--max_scale', default=2.0, type=float)
    parser.add_argument('--min_size', default=(560, 588, 616, 644, 672, 700), type=tuple)
    parser.add_argument('--max_size', default=896, type=int)
    parser.add_argument('--image_size', default=896, type=int)
    parser.add_argument('--sam_image_size', default=1024, type=int)
    parser.add_argument('--clip_image_size', default=1024, type=int)
    parser.add_argument('--crop_ratio', default=0.5, type=float)

    parser.add_argument('--feat_chans', default=256, type=int)
    parser.add_argument('--image_enc_use_fc', action="store_true")

    # parser.add_argument('--pt_model', type=str, default="dinov2")
    parser.add_argument('--weights', type=str, default="models/cosine/mscosine_refseg_fd1_md6_lr1e-4_ep50/pytorch_model_19ep/pytorch_model.bin")
    parser.add_argument('--dinov2-size', type=str, default="vit_large")
    parser.add_argument('--dinov2-weights', type=str, default="models/dinov2_vitl14_pretrain.pth")
    parser.add_argument('--sam-size', type=str, default="vit_l")
    parser.add_argument('--sam-weights', type=str, default="models/sam_vit_l_0b3195.pth")
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

    parser.add_argument('--class_weight', default=2.0, type=float)
    parser.add_argument('--mask_weight', default=5.0, type=float)
    parser.add_argument('--dice_weight', default=5.0, type=float)
    parser.add_argument('--no_object_weight', default=0.1, type=float)
    parser.add_argument('--train_num_points', default=12544, type=int)
    parser.add_argument('--oversample_ratio', default=3.0, type=float)
    parser.add_argument('--importance_sample_ratio', default=0.75, type=float)
    parser.add_argument("--deep_supervision", action="store_true", default=True)
    # evaluation
    parser.add_argument('--score_threshold', default=0.8, type=float)

    parser.add_argument("--ref_img_json", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--out_name", type=str, default="_visual_prompt.pkl")

    parser.add_argument("--prompts_txt", type=str, default="")
    parser.add_argument("--prompts_cls", type=str, default="elephant")
    parser.add_argument("--prompts_cattype", type=str, default="avg")
    parser.add_argument("--prompts_catalpha", type=float, default=0.5)
    parser.add_argument("--infer_dir", type=str, default="./infer_imgs")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument('--output_dir', type=str, default='outputs/referseg/debug')
    parser.add_argument('--vis_font_scale', type=float, default=2.0)
    parser.add_argument('--choose_color', type=str, default="")

    args = parser.parse_args()

    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    # Logger.initialize(args, root=args.log_root)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    # Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    print('# available GPUs: %d' % torch.cuda.device_count())

    # Model initialization
    model = build_model(args)
    print(f"model: {model}")
    assert os.path.exists(args.weights)
    state_dict = torch.load(args.weights, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"msg: {msg}")
    model.to(device)
    model.eval()

    # Helper classes (for testing) initialization
    # Evaluator.initialize()
    # Visualizer.initialize(args.visualize, root=args.log_root)

    # # Dataset initialization
    # FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    # dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)
    # print("dataset size: {}".format(len(dataloader_test)))

    # data_path = args.ref_img_json
    # ref_data = json.load(open(data_path))
    ref_data = get_ref_masks(args)
    data_info = {
        "image_dir" : args.image_dir,
        "gt_dir" : args.gt_dir
    }

    print("Get reference data!")

    # txt_data = "toys like this one"
    # txt_data = "red toys like this one"
    # txt_data = "toys like cars"
    txt_data = args.prompts_txt
    # Test COSINE
    with torch.no_grad():
        test(model, ref_data, txt_data, data_info, args=args)