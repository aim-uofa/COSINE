from typing import Tuple, List

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
from fsdet.model.image_encoder.encodersoup_single_scale_ablation import build_encodersoup
from fsdet.model.transofrmer_decoder import build_mformer


@META_ARCH_REGISTRY.register()
class COSINEABLATION(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        encoder_soup: nn.Module,
        transformer_decoder: nn.Module = None,
        use_visual: bool = True,
        use_text: bool = False,
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
        super(COSINEABLATION, self).__init__()

        self.encoder_soup = encoder_soup
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

    @classmethod
    def from_config(cls, cfg):

        from types import SimpleNamespace
        cfg_dict = dict(
            # dinov2
            dinov2_size=cfg.MODEL.DINO.SIZE,
            dinov2_weights=cfg.MODEL.DINO.WEIGHTS,
            # clip
            clip_weights=cfg.MODEL.CLIP.WEIGHTS,
            img_feats_merging_type=cfg.MODEL.COSINE.img_feats_merging_type,
            feat_chans=cfg.MODEL.COSINE.Transformer.feat_chans,
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

        transformer = build_mformer(args)

        return {
            "encoder_soup": encoder_soup,
            "transformer_decoder": transformer,
            "use_visual": cfg.MODEL.COSINE.use_visual,
            "use_text": cfg.MODEL.COSINE.use_text,
            "preprocess": cfg.MODEL.COSINE.preprocess,
            "sem_seg_postprocess_before_inference": cfg.MODEL.COSINE.sem_seg_postprocess_before_inference,
            "score_threshold": cfg.MODEL.COSINE.score_threshold,
            "use_id_query": cfg.MODEL.COSINE.use_id_query
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
        features = self.encoder_soup.merge_feature(features, prompts, batched_tag)

        outputs = self.transformer_decoder(
            features=features,
            prompts=prompts,
            tags=batched_tag
        )

        processed_results = []

        for output, input_per_image, image_size in zip(
                outputs, batched_inputs, dino_images.image_sizes
        ):
            ref_sem_labels = output['ref_sem_labels']
            pred_ins_logits = output['pred_ins_logits']
            pred_ins_masks = output['pred_ins_masks']
            processed_results.append({})


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
