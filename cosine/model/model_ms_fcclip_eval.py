from typing import Tuple, List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import Boxes, ImageList, Instances, BitMasks


# from cosine.model.image_encoder.encodersoup import build_encodersoup
# from cosine.model.image_encoder.encodersoup_ms_eval import build_encodersoup
from cosine.model.image_encoder.encodersoup_ms_fsod import build_encodersoup

from cosine.model.pixel_decoder.msdeformattn import build_pixel_decoder
from cosine.model.transformer_decoder.mformer_ms import build_mformer, MLP
from cosine.model.matcher import HungarianMatcher
from cosine.model.criterion import SetCriterion


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


class COSINE(nn.Module):

    def __init__(
        self,
        encoder_soup: nn.Module,
        pixel_decoder: nn.Module = None,
        transformer_decoder: nn.Module = None,
        criterion: nn.Module = None,
        # FC-CLIP
        geometric_ensemble_alpha: float = 0.4,
        geometric_ensemble_beta: float = 0.8,
        test_categories: list = [],
        # inference
        semantic_on: bool = True,
        instance_on: bool = False,
        identity_on: bool = False,
        sem_seg_postprocess_before_inference: bool = False,
        num_classes: int = 1,
        test_topk_per_image: int = 100,
        score_threshold: float = 0.8,

        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        super(COSINE, self).__init__()

        self.encoder_soup = encoder_soup
        self.pixel_decoder = pixel_decoder
        self.transformer_decoder = transformer_decoder
        self.criterion = criterion
        self.num_classes = num_classes
        self.test_topk_per_image = test_topk_per_image
        self.score_threshold = score_threshold
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.identity_on = identity_on

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # for n, param in self.image_encoder.encoder.named_parameters():
        #     param.requires_grad = False


        # FC-CLIP args
        self.mask_pooling = MaskPooling()
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.geometric_ensemble_beta = geometric_ensemble_beta

        COCO_CATEGORIES = MetadataCatalog.get('coco_2017_train_panoptic').stuff_classes
        coco_prompt_file = os.environ.get(
            "COSINE_COCO_PROMPT_FILE",
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "inference_fsod",
                "fsdet",
                "data",
                "coco_panoptic_with_prompt_eng.txt",
            ),
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
        self.o365_class_names = [s.lower() for s in o365_class_names]
        self.COCO_CATEGORIES = COCO_CATEGORIES

        # test_categories=MetadataCatalog.get("lvis_v0.5_val").thing_classes

        # assert len(LVIS_CATEGORIES) == len(test_categories)
        # self.test_class_names = []
        # for lvis_item, name in zip(LVIS_CATEGORIES, test_categories):
        #     assert lvis_item['name'] == name
        #     synonyms = [i.replace('_', ' ') for i in lvis_item['synonyms']]
        #     synonyms = ','.join(synonyms)
        #     self.test_class_names.append(synonyms)

        # self.test_text_classifier = None
        # self.category_overlapping_mask, self.test_num_templates, self.test_class_names = \
        #     self.prepare_class_names_from_metadata(self.test_class_names, COCO_CATEGORIES, o365_class_names)


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

        self.train_class_set = set(coco_class_names).union(set(o365_class_names))

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


    @property
    def device(self):
        return self.pixel_mean.device

    # def get_sam_embs(self, pixel_values: torch.FloatTensor):
    #     with torch.no_grad():
    #         image_embeddings = self.segmenter.image_encoder(pixel_values)
    #     return image_embeddings
    #
    # def get_enc_embs(self, pixel_values: torch.FloatTensor):
    #
    #     with torch.no_grad():
    #         image_embeddings = self.image_encoder.get_enc_embs(pixel_values)
    #     image_embeddings = self.image_encoder.neck(image_embeddings)
    #
    #     return image_embeddings

    def get_text_classifier(self):
        # if self.test_text_classifier is None:
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
        # self.test_text_classifier = text_classifier
        return test_text_classifier

    def forward(self, batched_inputs):


        batched_target_inputs = [item['target'] for item in batched_inputs]
        batched_visual_prompt = [item['visual_prompt'] for item in batched_inputs]
        batched_text_prompt = [item['text_prompt'] for item in batched_inputs]
        batched_tag = [item['tag'] for item in batched_inputs]
        # print(batched_tag)

        # dinov2 images
        dino_images = [x["image"].to(self.device) for x in batched_target_inputs]
        dino_images = ImageList.from_tensors(dino_images, size_divisibility=self.encoder_soup.dinov2.patch_size)
        # clip images
        clip_images = [x["clip_image"].to(self.device) for x in batched_target_inputs]
        clip_images = ImageList.from_tensors(clip_images, size_divisibility=self.encoder_soup.clip.patch_size)
        # sam images
        # sam_images = [x["sam_image"].to(self.device) for x in batched_target_inputs] # wo sam
        # sam_images = ImageList.from_tensors(sam_images, size_divisibility=self.encoder_soup.sam.patch_embed.proj.kernel_size[0])

        # labels and mask


        # encode image features
        # features = self.encoder_soup.encode_image(dino_images.tensor, clip_images.tensor, sam_images.tensor)
        features = self.encoder_soup.encode_image(dino_images.tensor, clip_images.tensor)
        clip_vis_dense_feats = features['clip_vis_dense']
        text_classifier, num_templates = self.get_text_classifier()


        # encode prompt
        prompts = self.encoder_soup.encode_prompt(batched_visual_prompt, batched_text_prompt, batched_tag)

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

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]['target']:
                tar_instances = [x['target']["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(tar_instances, dino_images)
            else:
                targets = None

            for out, target in zip(outputs, targets):

                if out['tag'] != 'refer':

                    ref_labels_unique = out['ref_sem_labels']
                    tar_labels_unique = torch.unique(target['labels'])
                    assert all(torch.isin(tar_labels_unique, ref_labels_unique))

                    out['ref_sem_labels_t'] = torch.tensor(list(range(out['ref_sem_labels'].shape[1]))).to(out['ref_sem_labels'])
                    ref_sem_labels_list = out['ref_sem_labels'][0].cpu().tolist()
                    tgt_ids = target["labels"]
                    target['labels_t'] = torch.tensor([ref_sem_labels_list.index(tid) for tid in tgt_ids]).to(tgt_ids)

                else:
                    assert out['ref_sem_labels'].nelement() == 0
                    out['ref_sem_labels_t'] = out['ref_sem_labels'][0]
                    target['labels_t'] = torch.empty(0, dtype=torch.int64, device=out['ref_sem_labels_t'].device)

                    # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            processed_results = []

            for output, input_per_image, image_size, clip_vis_dense in zip(
                    outputs, batched_target_inputs, dino_images.image_sizes, clip_vis_dense_feats
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

                if self.instance_on:
                    r = retry_if_cuda_oom(self.instance_inference)(pred_ins_masks, pred_ins_logits, ref_sem_labels)
                    processed_results[-1]["ins_seg"] = r

                elif self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(pred_ins_masks, pred_ins_logits, ref_sem_labels)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(
                            r, new_size, height, width
                        )
                    processed_results[-1]["sem_seg"] = r

            return processed_results[0] if len(processed_results) == 1 else processed_results


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

    def semantic_inference(self, pred_masks, pred_logits, labels):
        bs, n, h, w = pred_masks.shape
        _, _, m = pred_logits.shape

        mask_cls = F.softmax(pred_logits, dim=-1)[..., :-1]
        pred_masks = pred_masks.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, pred_masks)

        return semseg[0]


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
                    "ids": targets_per_image.ins_ids,
                }
            )
        return new_targets


    # def id_inference(self, pred_masks, pred_logits, labels):
    #
    #     # inference instances
    #     image_size = pred_masks.shape[-2:]
    #     # [Q, K]
    #     scores = F.softmax(pred_logits, dim=-1)[:, :-1]
    #     scores = scores.flatten(0, 1)
    #
    #     # First, select top-k based on score
    #     scores_per_image, topk_indices = scores.topk(min(self.test_topk_per_image, len(scores)), sorted=False)  # select top-100
    #     labels_per_image = labels[topk_indices]
    #     topk_indices = topk_indices // self.num_classes
    #     pred_masks = pred_masks[topk_indices]
    #     pred_masks_ = (pred_masks > 0).float()
    #
    #     # Second, filter instances below the threshold
    #
    #     # 1. calculate average mask prob
    #     mask_scores_per_image = (pred_masks.sigmoid().flatten(1) * pred_masks_.flatten(1)).sum(1) / (
    #             pred_masks_.flatten(1).sum(1) + 1e-6)
    #     # 2. calculate scores
    #     scores = scores_per_image * mask_scores_per_image
    #     # 3. filter
    #     vaild = scores > self.score_threshold
    #     scores_per_image = scores[vaild]
    #     pred_masks_ = pred_masks_[vaild]
    #     labels_per_image = labels_per_image[vaild]
    #
    #     result = Instances(image_size)
    #     # mask (before sigmoid)
    #     result.pred_masks = pred_masks_
    #     # result.pred_boxes = Boxes(torch.zeros(pred_masks_.size(0), 4))
    #     result.scores = scores_per_image
    #     result.pred_classes = labels_per_image
    #
    #     return result
    #
    # def instance_inference(self, pred_masks, pred_logits, labels):
    #
    #     # inference instances
    #     image_size = pred_masks.shape[-2:]
    #     # [Q, K]
    #     scores = F.softmax(pred_logits, dim=-1)[:, :-1]
    #     scores = scores.flatten(0, 1)
    #     num_classes = len(labels)
    #     labels = labels.unsqueeze(0).repeat(self.transformer_decoder.num_queries, 1).flatten(0, 1)
    #
    #     # First, select top-k based on score
    #     scores_per_image, topk_indices = scores.topk(min(self.test_topk_per_image, len(scores)), sorted=False)  # select top-100
    #     labels_per_image = labels[topk_indices]
    #     topk_indices = topk_indices // num_classes
    #     pred_masks = pred_masks[topk_indices]
    #     pred_masks_ = (pred_masks > 0).float()
    #
    #     # Second, filter instances below the threshold
    #
    #     # 1. calculate average mask prob
    #     mask_scores_per_image = (pred_masks.sigmoid().flatten(1) * pred_masks_.flatten(1)).sum(1) / (
    #             pred_masks_.flatten(1).sum(1) + 1e-6)
    #     # 2. calculate scores
    #     scores = scores_per_image * mask_scores_per_image
    #     # 3. filter
    #     vaild = scores > self.score_threshold
    #     scores_per_image = scores[vaild]
    #     pred_masks_ = pred_masks_[vaild]
    #     labels_per_image = labels_per_image[vaild]
    #
    #     result = Instances(image_size)
    #     # mask (before sigmoid)
    #     result.pred_masks = pred_masks_
    #     # result.pred_boxes = Boxes(torch.zeros(pred_masks_.size(0), 4))
    #     result.scores = scores_per_image
    #     result.pred_classes = labels_per_image
    #
    #     return result
    #
    # def semantic_inference(self, pred_masks, pred_logits, labels):
    #
    #     pred_masks = pred_masks[0]
    #     pred_logits = pred_logits[0]
    #
    #     pred_logits = F.softmax(pred_logits, dim=-1)[..., :-1]
    #     pred_masks = pred_masks.sigmoid()
    #     semseg = torch.einsum("qc,qhw->chw", pred_logits, pred_masks)
    #
    #     return semseg


def build_model(args):

    # EncoderSoup: DINOv2, SAM, and CLIP

    encoder_soup = build_encodersoup(args)
    pixel_decoder = build_pixel_decoder(args, encoder_soup.clip.output_shape())
    transformer = build_mformer(args)

    # building criterion
    matcher = HungarianMatcher(
        cost_class=args.class_weight,
        cost_mask=args.mask_weight,
        cost_dice=args.dice_weight,
        num_points=args.train_num_points,
    )

    weight_dict = {"loss_ce_ins": args.class_weight, "loss_ce_id": args.class_weight, "loss_mask": args.mask_weight,
                   "loss_dice": args.dice_weight}

    if args.deep_supervision:
        dec_layers = args.transformer_depth
        aux_weight_dict = {}
        for i in range(dec_layers):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "masks"]

    criterion = SetCriterion(
        num_classes=1,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.no_object_weight,
        losses=losses,
        num_points=args.train_num_points,
        oversample_ratio=args.oversample_ratio,
        importance_sample_ratio=args.importance_sample_ratio,
    )

    model = COSINE(
        encoder_soup=encoder_soup,
        pixel_decoder=pixel_decoder,
        transformer_decoder=transformer,
        criterion=criterion,
        score_threshold=args.score_threshold
    )

    return model


if __name__ == '__main__':

    import argparse
    from cosine.data.dataset import build_dataset

    parser = argparse.ArgumentParser('COSINE Model DEBUG')
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
    parser.add_argument('--neck_dim_feedforward', default=512, type=int)
    parser.add_argument('--neck_encoder_layers', default=3, type=int)
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
    args = parser.parse_args()

    args.steps_per_epoch = 1000
    args.world_size = 1
    args.update_freq = 1


    # dataset
    dataset = build_dataset(is_train=True, args=args)
    sampler = torch.utils.data.SequentialSampler(dataset)


    def trivial_batch_collator(batch):
        """
        A batch collator that does nothing.
        """
        return batch

    loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=trivial_batch_collator
    )

    # model

    model = build_model(args)
    model.to('cuda')

    learnable_param_list = [_ for _, p in model.named_parameters() if p.requires_grad]
    print(f"learnable params: {learnable_param_list}")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))


    for inputs in loader:
        losses = model(inputs)
        loss = sum([v for k, v in losses.items()])
        print(loss)
