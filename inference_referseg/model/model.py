from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import Boxes, ImageList, Instances, BitMasks


from inference_referseg.model.image_encoder.encodersoup import build_encodersoup

from inference_referseg.model.transformer_decoder.mformer import build_mformer, MLP

class COSINE(nn.Module):

    def __init__(
        self,
        encoder_soup: nn.Module,
        transformer_decoder: nn.Module = None,
        # inference
        sem_seg_postprocess_before_inference: bool = True,
        num_classes: int = 1,
        test_topk_per_image: int = 100,
        score_threshold: float = -1e9,

        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        super(COSINE, self).__init__()

        self.encoder_soup = encoder_soup
        self.transformer_decoder = transformer_decoder
        self.num_classes = num_classes
        self.test_topk_per_image = test_topk_per_image
        self.score_threshold = score_threshold
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        assert self.sem_seg_postprocess_before_inference


    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):

        assert not self.training

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
        sam_images = [x["sam_image"].to(self.device) for x in batched_target_inputs]
        sam_images = ImageList.from_tensors(sam_images, size_divisibility=self.encoder_soup.sam.patch_embed.proj.kernel_size[0])

        # labels and mask


        # encode image features
        features = self.encoder_soup.encode_image(dino_images.tensor, clip_images.tensor, sam_images.tensor)

        # encode prompt
        prompts = self.encoder_soup.encode_prompt(batched_visual_prompt, batched_text_prompt, batched_tag)

        # feature merge
        features = self.encoder_soup.merge_feature(features, prompts)

        outputs = self.transformer_decoder(
            features=features,
            prompts=prompts,
            tags=batched_tag
        )

        processed_results = []

        for output, input_per_image, image_size in zip(
                outputs, batched_target_inputs, dino_images.image_sizes
        ):
            ref_id_labels = output['ref_id_labels']
            pred_id_logits = output['pred_id_logits']
            pred_id_masks = output['pred_id_masks']
            pred_ids = ref_id_labels
            processed_results.append({})

            # upsample masks
            pred_id_masks = F.interpolate(
                pred_id_masks,
                size=image_size,
                mode="bilinear",
                align_corners=False,
            )

            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            new_size = input_per_image.get('resize', image_size)

            if self.sem_seg_postprocess_before_inference:
                pred_id_masks = retry_if_cuda_oom(sem_seg_postprocess)(
                    pred_id_masks[0], new_size, height, width
                )
                pred_id_logits = pred_id_logits.to(pred_id_masks)[0]
                ref_id_labels = ref_id_labels.to(pred_id_masks)[0]
                pred_ids = pred_ids.to(pred_id_masks.device)[0]

            r = retry_if_cuda_oom(self.id_inference)(pred_id_masks, pred_id_logits, ref_id_labels, pred_ids)
            if not self.sem_seg_postprocess_before_inference:
                r = retry_if_cuda_oom(sem_seg_postprocess)(
                    r, new_size, height, width
                )
            processed_results[-1]["id_seg"] = r

        return processed_results[0] if len(processed_results) == 1 else processed_results

    def id_inference(self, pred_masks, pred_logits, labels, ids):

        # inference instances
        image_size = pred_masks.shape[-2:]
        # [Q, K]
        scores = F.softmax(pred_logits, dim=-1)[:, :-1]
        scores = scores.flatten(0, 1)

        # First, select top-k based on score
        scores_per_image, topk_indices = scores.topk(min(self.test_topk_per_image, len(scores)),
                                                     sorted=False)  # select top-100
        labels_per_image = labels[topk_indices]
        ids_per_image = ids[topk_indices]
        topk_indices = topk_indices // self.num_classes
        pred_masks = pred_masks[topk_indices]
        pred_masks_ = (pred_masks > 0).float()

        # Second, filter instances below the threshold

        # 1. calculate average mask prob
        mask_scores_per_image = (pred_masks.sigmoid().flatten(1) * pred_masks_.flatten(1)).sum(1) / (
                pred_masks_.flatten(1).sum(1) + 1e-6)
        # 2. calculate scores
        scores = scores_per_image * mask_scores_per_image
        # # 3. filter
        # vaild = scores > self.score_threshold
        # scores_per_image = scores[vaild]
        # pred_masks_ = pred_masks_[vaild]
        # labels_per_image = labels_per_image[vaild]
        # ids_per_image = ids_per_image[vaild]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = pred_masks_
        # result.pred_boxes = Boxes(torch.zeros(pred_masks_.size(0), 4))
        result.scores = scores_per_image
        result.pred_classes = labels_per_image
        result.pred_ids = ids_per_image

        return result

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                       device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "ids": targets_per_image.ins_ids,
                }
            )
        return new_targets

def build_model(args):

    # EncoderSoup: DINOv2, SAM, and CLIP

    encoder_soup = build_encodersoup(args)

    transformer = build_mformer(args)


    model = COSINE(
        encoder_soup=encoder_soup,
        transformer_decoder=transformer,
        score_threshold=args.score_threshold
    )

    return model