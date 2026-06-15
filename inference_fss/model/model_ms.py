from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import Boxes, ImageList, Instances, BitMasks


# from cosine.model.image_encoder.encodersoup import build_encodersoup
from inference_fss.model.image_encoder.encodersoup_ms import build_encodersoup

from cosine.model.pixel_decoder.msdeformattn import build_pixel_decoder
from inference_fss.model.transformer_decoder.mformer_ms import build_mformer, MLP


class COSINE(nn.Module):

    def __init__(
        self,
        encoder_soup: nn.Module,
        pixel_decoder: nn.Module = None,
        transformer_decoder: nn.Module = None,
        # inference
        sem_seg_postprocess_before_inference: bool = False,
        score_threshold: float = 0.7,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        super(COSINE, self).__init__()

        self.encoder_soup = encoder_soup
        self.pixel_decoder = pixel_decoder
        self.transformer_decoder = transformer_decoder

        self.score_threshold = score_threshold
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)


    @property
    def device(self):
        return self.pixel_mean.device


    def forward(self, batched_inputs):

        assert not self.training

        batched_target_inputs = [item['target'] for item in batched_inputs if item['target'] is not None]
        batched_visual_prompt = [item['visual_prompt'] for item in batched_inputs if item['visual_prompt'] is not None]
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

        # encode prompt
        prompts = self.encoder_soup.encode_prompt(batched_visual_prompt, batched_text_prompt, batched_tag)

        # merge visual prompt

        id_prompt_list = prompts['id_prompt_list']
        sem_prompt_list = prompts['sem_prompt_list']
        id_prompt_label_list = prompts['id_prompt_label_list']
        sem_prompt_label_list = prompts['sem_prompt_label_list']

        sem_prompt_list = torch.cat(sem_prompt_list, dim=0)
        sem_prompt_list = [torch.mean(sem_prompt_list, dim=0, keepdim=True)]
        sem_prompt_label_list = [sem_prompt_label_list[0]]

        prompts = {
            'id_prompt_list': id_prompt_list,
            'sem_prompt_list': sem_prompt_list,
            'id_prompt_label_list': id_prompt_label_list,
            'sem_prompt_label_list': sem_prompt_label_list
        }
        batched_tag = [batched_tag[0]]


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

        for output, input_per_image, image_size in zip(
                outputs, batched_target_inputs, dino_images.image_sizes
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

            r = retry_if_cuda_oom(self.semantic_inference)(pred_ins_masks, pred_ins_logits, ref_sem_labels)
            if not self.sem_seg_postprocess_before_inference:
                r = retry_if_cuda_oom(sem_seg_postprocess)(
                    r, new_size, height, width
                )
            processed_results[-1]["sem_seg"] = r

        return processed_results[0] if len(processed_results) == 1 else processed_results

    def semantic_inference(self, pred_masks, pred_logits, labels):

        pred_masks = pred_masks[0]
        pred_logits = pred_logits[0]

        pred_logits = F.softmax(pred_logits, dim=-1)[..., :-1]
        pred_masks = pred_masks.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", pred_logits, pred_masks)

        return semseg

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


def build_model(args):

    # EncoderSoup: DINOv2, SAM, and CLIP

    encoder_soup = build_encodersoup(args)
    pixel_decoder = build_pixel_decoder(args, encoder_soup.clip.output_shape())
    transformer = build_mformer(args)

    model = COSINE(
        encoder_soup=encoder_soup,
        pixel_decoder=pixel_decoder,
        transformer_decoder=transformer,
        score_threshold=args.score_threshold
    )

    return model
