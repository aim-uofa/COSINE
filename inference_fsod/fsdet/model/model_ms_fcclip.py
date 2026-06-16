from typing import Tuple, List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY

from dinov2.models import vision_transformer as vits
import dinov2.utils.utils as dinov2_utils
from fsdet.model.image_encoder.encodersoup_ms import build_encodersoup

from fsdet.model.transofrmer_decoder.mformer_ms import build_mformer

import sys
sys.path.append('../')
from cosine.model.pixel_decoder.msdeformattn import build_pixel_decoder
from detectron2.data import MetadataCatalog
from cosine.data.load_data.object365 import O365_CATEGORIES
from detectron2.data.datasets.lvis_v0_5_categories import LVIS_CATEGORIES

VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]

class MaskPooling(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, H, W]
            mask: [B, Q, H, W]
        """
        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        with torch.no_grad():
            mask = mask.detach()
            mask = (mask > 0).to(mask.dtype)
            denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask / denorm,
        )
        return mask_pooled_x

def get_classification_logits(x, text_classifier, logit_scale, num_templates=None):
    # x in shape of [B, *, C]
    # text_classifier in shape of [num_classes, C]
    # logit_scale is a learnable scalar https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L201
    # return: [B, *, num_classes]
    x = F.normalize(x, dim=-1)
    logit_scale = torch.clamp(logit_scale.exp(), max=100)
    pred_logits = logit_scale * x @ text_classifier.T # B, *, N + 1
    # max ensembel as in OpenSeg/ODISE
    final_pred_logits = []
    cur_idx = 0
    for num_t in num_templates:
        final_pred_logits.append(pred_logits[:, :, cur_idx: cur_idx + num_t].max(-1).values)
        cur_idx += num_t
    # final_pred_logits.append(pred_logits[:, :, -1]) # the last classifier is for void
    final_pred_logits = torch.stack(final_pred_logits, dim=-1)
    return final_pred_logits

@META_ARCH_REGISTRY.register()
class MSCOSINEFCCLIP(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        encoder_soup: nn.Module,
        pixel_decoder: nn.Module = None,
        transformer_decoder: nn.Module = None,
        use_visual: bool = True,
        use_text: bool = False,
        # FC-CLIP
        geometric_ensemble_alpha: float = 0.4,
        geometric_ensemble_beta: float = 0.8,
        test_categories: list = [],
        # inference
        preprocess: bool = True,
        sem_seg_postprocess_before_inference: bool = True,
        num_classes: int = 1,
        test_topk_per_image: int = 100,
        score_threshold: float = 0.,
        use_id_query: bool = True,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        super(MSCOSINEFCCLIP, self).__init__()

        self.encoder_soup = encoder_soup
        self.pixel_decoder = pixel_decoder
        self.transformer_decoder = transformer_decoder
        self.preprocess = preprocess
        self.num_classes = num_classes
        self.test_topk_per_image = test_topk_per_image
        self.score_threshold = score_threshold
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.use_visual = use_visual
        self.use_text = use_text

        if preprocess:
            self.ref_id_feat_dict = {}
            self.ref_sem_feat_dict = {}

        # additional args
        self.use_id_query = use_id_query
        assert self.sem_seg_postprocess_before_inference

        # FC-CLIP args
        self.mask_pooling = MaskPooling()
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.geometric_ensemble_beta = geometric_ensemble_beta

        COCO_CATEGORIES = MetadataCatalog.get('coco_2017_train_panoptic').stuff_classes
        coco_prompt_file = os.environ.get(
            "COSINE_COCO_PROMPT_FILE",
            os.path.join(os.path.dirname(__file__), "..", "data", "coco_panoptic_with_prompt_eng.txt"),
        )
        if os.path.exists(coco_prompt_file):
            COCO_CATEGORIES_ = []
            coco_id_names = open(coco_prompt_file).read().splitlines()
            for line in coco_id_names:
                idx, name = line.split(':')
                if idx == 0 or name == "invalid_class_id":
                    continue
                COCO_CATEGORIES_.append(name)
            COCO_CATEGORIES = COCO_CATEGORIES_

        o365_class_names = [i['name'] for i in O365_CATEGORIES]
        o365_class_names = [s.lower() for s in o365_class_names]

        assert len(LVIS_CATEGORIES) == len(test_categories)
        self.test_class_names = []
        for lvis_item, name in zip(LVIS_CATEGORIES, test_categories):
            assert lvis_item['name'] == name
            synonyms = [i.replace('_', ' ') for i in lvis_item['synonyms']]
            synonyms = ','.join(synonyms)
            self.test_class_names.append(synonyms)

        self.test_text_classifier = None
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = \
            self.prepare_class_names_from_metadata(self.test_class_names, COCO_CATEGORIES, o365_class_names)

    def prepare_class_names_from_metadata(self, class_names, coco_class_names, o365_class_names):
        def split_labels1(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        def split_labels2(x):
            res = []
            for x_ in x:
                x_ = x_.split('/') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier

        class_names = split_labels1(class_names)
        coco_class_names = split_labels1(coco_class_names)
        o365_class_names = split_labels2(o365_class_names)

        coco_class_names = {l for label in coco_class_names for l in label}
        o365_class_names = {l for label in o365_class_names for l in label}

        train_class_set = set(coco_class_names).union(set(o365_class_names))

        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not train_class_set.isdisjoint(set(test_class_names))
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)

        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)

        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
        class_names = templated_class_names
        #print("text for classification:", class_names)
        return category_overlapping_mask, num_templates, class_names


    @classmethod
    def from_config(cls, cfg):

        from types import SimpleNamespace
        cfg_dict = dict(
            # dinov2
            dinov2_size=cfg.MODEL.DINO.SIZE,
            dinov2_weights=cfg.MODEL.DINO.WEIGHTS,
            # clip
            clip_weights=cfg.MODEL.CLIP.WEIGHTS,
            feat_chans=cfg.MODEL.COSINE.Transformer.feat_chans,
            # neck
            neck_in_features=cfg.MODEL.COSINE.PIXEL_DECODER.neck_in_features,
            neck_encoder_in_features=cfg.MODEL.COSINE.PIXEL_DECODER.neck_encoder_in_features,
            neck_conv_dim=cfg.MODEL.COSINE.PIXEL_DECODER.neck_conv_dim,
            neck_mask_dim=cfg.MODEL.COSINE.PIXEL_DECODER.neck_mask_dim,
            neck_transformer_dropout=cfg.MODEL.COSINE.PIXEL_DECODER.neck_transformer_dropout,
            neck_transformer_nheads=cfg.MODEL.COSINE.PIXEL_DECODER.neck_transformer_nheads,
            neck_dim_feedforward=cfg.MODEL.COSINE.PIXEL_DECODER.neck_dim_feedforward,
            neck_encoder_layers=cfg.MODEL.COSINE.PIXEL_DECODER.neck_encoder_layers,
            neck_common_stride=cfg.MODEL.COSINE.PIXEL_DECODER.neck_common_stride,
            #MFormer
            transformer_depth=cfg.MODEL.COSINE.Transformer.depth,
            transformer_nheads=cfg.MODEL.COSINE.Transformer.nheads,
            transformer_mlp_dim=cfg.MODEL.COSINE.Transformer.mlp_dim,
            transformer_mask_dim=cfg.MODEL.COSINE.Transformer.mask_dim,
            transformer_fusion_layer_depth=cfg.MODEL.COSINE.Transformer.fusion_layer_depth,
            transformer_num_queries=cfg.MODEL.COSINE.Transformer.num_queries,
            transformer_pre_norm=cfg.MODEL.COSINE.Transformer.pre_norm,
        )
        def dict_to_namespace(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            return d

        args = dict_to_namespace(cfg_dict)


        encoder_soup = build_encodersoup(args)
        pixel_decoder = build_pixel_decoder(args, encoder_soup.clip.output_shape())
        transformer = build_mformer(args)

        return {
            "encoder_soup": encoder_soup,
            "pixel_decoder": pixel_decoder,
            "transformer_decoder": transformer,
            "use_visual": cfg.MODEL.COSINE.use_visual,
            "use_text": cfg.MODEL.COSINE.use_text,
            "preprocess": cfg.MODEL.COSINE.preprocess,
            "sem_seg_postprocess_before_inference": cfg.MODEL.COSINE.sem_seg_postprocess_before_inference,
            "score_threshold": cfg.MODEL.COSINE.score_threshold,
            "use_id_query": cfg.MODEL.COSINE.use_id_query,
            "geometric_ensemble_alpha": cfg.MODEL.COSINE.GEOMETRIC_ENSEMBLE_ALPHA,
            "geometric_ensemble_beta": cfg.MODEL.COSINE.GEOMETRIC_ENSEMBLE_BETA,
            "test_categories": MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def get_sam_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings = self.segmenter.image_encoder(pixel_values)
        return image_embeddings

    def get_enc_embs(self, pixel_values: torch.FloatTensor):

        with torch.no_grad():
            image_embeddings = self.image_encoder.get_enc_embs(pixel_values)
        image_embeddings = self.image_encoder.neck(image_embeddings)

        return image_embeddings

    def do_preprocess(self, batched_inputs):


        batched_visual_prompt = batched_inputs
        if self.use_text:
            batched_text_prompt = [item['class_names'] for item in batched_inputs]
        else:
            batched_text_prompt = [None for item in batched_inputs]
        if not self.use_visual:
            batched_tag = ['sem' for item in batched_inputs]
        else:
            batched_tag = ['visual' for item in batched_inputs]

        prompts = self.encoder_soup.encode_prompt(batched_visual_prompt, batched_text_prompt, batched_tag)

        id_prompt_list = prompts['id_prompt_list']
        sem_prompt_list = prompts['sem_prompt_list']
        id_prompt_label_list = prompts['id_prompt_label_list']
        sem_prompt_label_list = prompts['sem_prompt_label_list']

        for id_prompts, sem_prompts, id_prompt_labels, sem_prompt_labels in zip(id_prompt_list, sem_prompt_list, id_prompt_label_list, sem_prompt_label_list):

            if not isinstance(id_prompt_labels, list):
                id_prompt_labels = id_prompt_labels.cpu().tolist()
                sem_prompt_labels = sem_prompt_labels.cpu().tolist()

            for id_prompt, sem_prompt, id_prompt_label, sem_prompt_label in zip(id_prompts, sem_prompts, id_prompt_labels, sem_prompt_labels):

                if id_prompt_label not in self.ref_id_feat_dict:
                    self.ref_id_feat_dict[id_prompt_label] = []
                self.ref_id_feat_dict[id_prompt_label].append(id_prompt.cpu())

                if sem_prompt_label not in self.ref_sem_feat_dict:
                    self.ref_sem_feat_dict[sem_prompt_label] = []
                self.ref_sem_feat_dict[sem_prompt_label].append(sem_prompt.cpu())


    def integrate_queries(self):

        id_labels = []
        id_queries = []

        for label in self.ref_id_feat_dict.keys():
            feats = self.ref_id_feat_dict[label]
            query = torch.stack(feats, dim=0).mean(0)
            id_labels.append(label)
            id_queries.append(query)

        id_labels = torch.tensor(id_labels, dtype=torch.int64, device=self.device)
        id_queries = torch.stack(id_queries, dim=0).to(self.device)


        seg_labels = []
        seg_queries = []

        for label in self.ref_sem_feat_dict.keys():
            feats = self.ref_sem_feat_dict[label]
            query = torch.stack(feats, dim=0).mean(0)
            seg_labels.append(label)
            seg_queries.append(query)

        seg_labels = torch.tensor(seg_labels, dtype=torch.int64, device=self.device)
        seg_queries = torch.stack(seg_queries, dim=0).to(self.device)


        self.register_buffer("id_labels", id_labels)
        self.register_buffer("id_queries", id_queries)
        self.register_buffer("seg_labels", seg_labels)
        self.register_buffer("seg_queries", seg_queries)

        return self.id_labels, self.id_queries, self.seg_labels, self.seg_queries

    def forward(self, batched_inputs):

        assert not self.training

        # # prepare dinov2 features
        # images = [x["image"].to(self.device) for x in batched_inputs]
        # images = ImageList.from_tensors(images, size_divisibility=self.image_encoder.patch_size)

        # dinov2 images
        dino_images = [x["image"].to(self.device) for x in batched_inputs]
        dino_images = ImageList.from_tensors(dino_images, size_divisibility=self.encoder_soup.dinov2.patch_size)
        # clip images
        clip_images = [x["clip_image"].to(self.device) for x in batched_inputs]
        clip_images = ImageList.from_tensors(clip_images, size_divisibility=self.encoder_soup.clip.patch_size)

        # features = self.get_enc_embs(images.tensor)

        if self.preprocess:
            self.do_preprocess(batched_inputs)
            return None

        features = self.encoder_soup.encode_image(dino_images.tensor, clip_images.tensor)
        clip_vis_dense_feats = features['clip_vis_dense']
        text_classifier, num_templates = self.get_text_classifier()

        if not self.use_id_query:
            id_queries = torch.randn((0,self.id_queries.shape[-1])).to(self.id_queries)
            id_labels = torch.randn((0, self.id_labels.shape[-1])).to(self.id_labels)
        else:
            id_queries = self.id_queries
            id_labels = self.id_labels

        id_prompt_list = [id_queries for i in range(len(batched_inputs))]
        sem_prompt_list = [self.seg_queries for i in range(len(batched_inputs))]
        id_prompt_label_list = [id_labels for i in range(len(batched_inputs))]
        sem_prompt_label_list = [self.seg_labels for i in range(len(batched_inputs))]

        prompts = {
            'id_prompt_list': id_prompt_list,
            'sem_prompt_list': sem_prompt_list,
            'id_prompt_label_list': id_prompt_label_list,
            'sem_prompt_label_list': sem_prompt_label_list
        }

        if not self.use_visual:
            batched_tag = ['sem' for item in batched_inputs]
        else:
            batched_tag = ['visual' for item in batched_inputs]

        # feature merge
        features = self.encoder_soup.merge_feature(features, prompts)

        # fpn, deformable attn fpn
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)


        outputs = self.transformer_decoder(
            features=multi_scale_features,
            mask_features=mask_features,
            prompts=prompts,
            tags=batched_tag
        )

        processed_results = []

        for output, input_per_image, image_size, clip_vis_dense in zip(
                outputs, batched_inputs, dino_images.image_sizes, clip_vis_dense_feats
        ):
            ref_sem_labels = output['ref_sem_labels']
            pred_ins_logits = output['pred_ins_logits']
            pred_ins_masks = output['pred_ins_masks']
            clip_feature = clip_vis_dense[None, ...]

            processed_results.append({})

            # calcul clip score
            mask_for_pooling = F.interpolate(pred_ins_masks, size=clip_feature.shape[-2:],
                                    mode='bilinear', align_corners=False)
            pooled_clip_feature = self.mask_pooling(clip_feature, mask_for_pooling)
            pooled_clip_feature = self.encoder_soup.clip.encoder.visual_prediction_forward(pooled_clip_feature)

            out_vocab_cls_results = get_classification_logits(pooled_clip_feature, text_classifier, self.encoder_soup.clip.encoder.clip_model.logit_scale, num_templates)
            in_vocab_cls_results = pred_ins_logits[..., :-1] # remove void

            out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
            in_vocab_cls_results = in_vocab_cls_results.softmax(-1)
            category_overlapping_mask = self.category_overlapping_mask.to(self.device)

            alpha = self.geometric_ensemble_alpha
            beta = self.geometric_ensemble_beta

            cls_logits_seen = (
                (in_vocab_cls_results ** (1 - alpha) * out_vocab_cls_probs**alpha).log()
                * category_overlapping_mask
            )
            cls_logits_unseen = (
                (in_vocab_cls_results ** (1 - beta) * out_vocab_cls_probs**beta).log()
                * (1 - category_overlapping_mask)
            )
            cls_results = cls_logits_seen + cls_logits_unseen


            # This is used to filtering void predictions.
            is_void_prob = F.softmax(pred_ins_logits, dim=-1)[..., -1:]
            mask_cls_probs = torch.cat([
                cls_results.softmax(-1) * (1.0 - is_void_prob),
                is_void_prob], dim=-1)
            pred_ins_logits = torch.log(mask_cls_probs + 1e-8)

            # upsample masks
            pred_ins_masks = F.interpolate(
                pred_ins_masks,
                size=image_size,
                mode="bilinear",
                align_corners=False,
            )

            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            new_size = (input_per_image['instances'].image_size[0], input_per_image['instances'].image_size[1])

            if self.sem_seg_postprocess_before_inference:

                pred_ins_masks = retry_if_cuda_oom(sem_seg_postprocess)(
                    pred_ins_masks[0], new_size, height, width
                )
                pred_ins_logits = pred_ins_logits.to(pred_ins_masks)[0]
                ref_sem_labels = ref_sem_labels.to(pred_ins_masks)[0]


            r = retry_if_cuda_oom(self.instance_inference)(pred_ins_masks, pred_ins_logits, ref_sem_labels)
            if not self.sem_seg_postprocess_before_inference:
                r = retry_if_cuda_oom(sem_seg_postprocess)(
                    r, new_size, height, width
                )
            processed_results[-1]["instances"] = r

        return processed_results

    def get_text_classifier(self):
        if self.test_text_classifier is None:
            text_classifier = []
            # this is needed to avoid oom, which may happen when num of class is large
            bs = 128
            for idx in range(0, len(self.test_class_names), bs):
                text_classifier.append(self.encoder_soup.clip.encoder.get_text_classifier(self.test_class_names[idx:idx+bs], self.device).detach())
            text_classifier = torch.cat(text_classifier, dim=0)

            # average across templates and normalization.
            text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
            text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
            text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
            self.test_text_classifier = text_classifier
        return self.test_text_classifier, self.test_num_templates

    def instance_inference(self, pred_masks, pred_logits, labels):

        # inference instances
        image_size = pred_masks.shape[-2:]
        # [Q, K]
        scores = F.softmax(pred_logits, dim=-1)[:, :-1]
        scores = scores.flatten(0, 1)
        num_classes = len(labels)
        labels = labels.unsqueeze(0).repeat(self.transformer_decoder.num_queries, 1).flatten(0, 1)

        # First, select top-k based on score
        scores_per_image, topk_indices = scores.topk(min(self.test_topk_per_image, len(scores)), sorted=False)  # select top-100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // num_classes
        pred_masks = pred_masks[topk_indices]
        pred_masks_ = (pred_masks > 0).float()

        # Second, filter instances below the threshold

        # 1. calculate average mask prob
        mask_scores_per_image = (pred_masks.sigmoid().flatten(1) * pred_masks_.flatten(1)).sum(1) / (
                pred_masks_.flatten(1).sum(1) + 1e-6)
        # 2. calculate scores
        scores = scores_per_image * mask_scores_per_image
        # 3. filter
        vaild = scores > self.score_threshold
        scores_per_image = scores[vaild]
        pred_masks_ = pred_masks_[vaild]
        labels_per_image = labels_per_image[vaild]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = pred_masks_

        result.scores = scores_per_image
        result.pred_classes = labels_per_image.to(torch.int64)

        # get bbox from mask
        pred_boxes = torch.zeros(pred_masks_.size(0), 4)
        for i in range(pred_masks_.size(0)):
           mask = pred_masks_[i].squeeze()
           ys, xs = torch.where(mask)
           try:
                pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).float()
           except:
               pred_boxes[i] = torch.tensor([0,0,1,1]).to(mask).float()

        result.pred_boxes = Boxes(pred_boxes)

        return result

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets
