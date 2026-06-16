import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.structures import ImageList
from . import build_sam, build_clip, build_dinov2
from segment_anything.modeling.common import LayerNorm2d

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

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class EncoderSoup(nn.Module):
    def __init__(
            self,
            dinov2=None,
            clip=None,
            sam=None,
            merging_type='simple_merging'
    ):
        super().__init__()

        self.dinov2 = dinov2
        self.clip = clip
        self.sam = sam
        self.mask_pooling = MaskPooling()

        dim = self.dinov2.out_chans

        self.text_mlp = MLP(self.clip.encoder.dim_latent, dim, dim, 1)

        self.merging_type = merging_type
        if self.merging_type == 'simple_merging':

            self.merging_module = nn.Sequential(
                nn.Conv2d(
                    dim * 3,
                    dim,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(dim),
            )

    @property
    def device(self):
        return self.dinov2.encoder.cls_token.device

    def encode_image(
            self,
            dino_images: torch.Tensor,
            clip_images: torch.Tensor,
            sam_images: torch.Tensor
    ):

        with torch.no_grad():
            dino_embeddings = self.dinov2.get_enc_embs(dino_images)
            clip_embeddings = self.clip.get_enc_embs(clip_images)
            sam_embeddings = self.sam.get_enc_embs(sam_images)
        dino_embeddings = self.dinov2.neck(dino_embeddings)
        clip_embeddings = self.clip.neck(clip_embeddings)
        sam_embeddings = self.sam.neck(sam_embeddings)

        return {
            'dino': dino_embeddings,
            'clip': clip_embeddings,
            'sam': sam_embeddings
        }

    def encode_vision_prompt(self, ref_embedding, ref_masks, ref_labels):

        mask_for_pooling = F.interpolate(ref_masks[None, ...].to(ref_embedding.dtype), size=(256, 256),
                                         mode='nearest').to(ref_embedding.dtype)
        ref_feat_for_pooling = F.interpolate(ref_embedding, size=mask_for_pooling.shape[-2:],
                                             mode='bilinear', align_corners=False).to(ref_embedding.dtype)

        # id-level query
        id_queries = self.mask_pooling(ref_feat_for_pooling, mask_for_pooling)  # bs, nm, c

        # sem-level query
        sem_labels = torch.unique(ref_labels)
        uni_ref_label = sem_labels.cpu().tolist()
        sem_mask_for_pooling = []
        for l in uni_ref_label:
            l_ids = ref_labels == l
            sem_masks = mask_for_pooling[:, l_ids].sum(1)[:, None]
            sem_mask_for_pooling.append(sem_masks)
        sem_mask_for_pooling = torch.cat(sem_mask_for_pooling, dim=1)

        sem_queries = self.mask_pooling(ref_feat_for_pooling, sem_mask_for_pooling)  # bs, nc, c

        id_queries = id_queries / (id_queries.norm(dim=-1, keepdim=True) + 1e-6)
        sem_queries = sem_queries / (sem_queries.norm(dim=-1, keepdim=True) + 1e-6)

        return id_queries[0], sem_queries[0], sem_labels



    def encode_text_prompt(self, class_names):
        assert isinstance(class_names, list)

        def fill_all_templates_ensemble(x=''):
            res = []
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

        text_prompts = []
        bs = 128
        for idx in range(0, len(class_names), bs):
            text_prompts.append(
                self.clip.encoder.get_text_classifier(class_names[idx:min(idx + bs, len(class_names))], self.device).detach())
        text_prompts = torch.cat(text_prompts, dim=0)

        # average across templates and normalization.
        text_prompts = text_prompts / (text_prompts.norm(dim=-1, keepdim=True) + 1e-6)
        text_prompts = text_prompts.reshape(text_prompts.shape[0] // len(VILD_PROMPT), len(VILD_PROMPT),
                                                  text_prompts.shape[-1]).mean(1)
        text_prompts = self.text_mlp(text_prompts)
        text_prompts = text_prompts / (text_prompts.norm(dim=-1, keepdim=True) + 1e-6)

        return text_prompts


    def encode_prompt(self, batched_visual_prompt, batched_text_prompt, batched_tag):

        id_prompt_list = []
        sem_prompt_list = []
        id_prompt_label_list = []
        sem_prompt_label_list = []

        for vis_prompt, text_prompt, tag in zip(batched_visual_prompt, batched_text_prompt, batched_tag):
            if tag == 'refer':
                id_labels = [p['id'] for p in text_prompt]
                captions = [p['caption'] for p in text_prompt]
                id_prompts = self.encode_text_prompt(captions)
                sem_prompts, sem_labels = None, None

            elif tag == 'sem':
                sem_labels, texts = [], []
                for k in text_prompt:
                    sem_labels.append(k)
                    texts.append(text_prompt[k])
                sem_prompts = self.encode_text_prompt(texts)
                id_prompts, id_labels = None, None

            elif tag == 'visual':

                images = [vis_prompt['image'].to(self.device)] # Batched inputs used to expend N-shots
                images = ImageList.from_tensors(images, size_divisibility=self.dinov2.patch_size)
                with torch.no_grad():
                    ref_embedding = self.dinov2.get_enc_embs(images.tensor)
                ref_embedding = self.dinov2.neck(ref_embedding)

                assert "instances" in vis_prompt, "reference masks must be provided!"
                ref_instances = [vis_prompt["instances"].to(self.device)]
                ref_targets = self.prepare_targets(ref_instances, images)
                ref_masks = [ref_target['masks'] for ref_target in ref_targets][0]
                ref_labels = [ref_target['labels'] for ref_target in ref_targets][0]
                ref_ids = [ref_target['ids'] for ref_target in ref_targets][0]

                id_queries, sem_queries, sem_labels = self.encode_vision_prompt(ref_embedding, ref_masks, ref_labels)

                id_prompts = id_queries
                id_labels = ref_ids

                if text_prompt is not None:
                    text_labels, texts = [], []
                    for k in text_prompt:
                        text_labels.append(k)
                        texts.append(text_prompt[k])
                    sem_prompts = self.encode_text_prompt(texts)
                    text_labels = torch.tensor(text_labels).to(self.device).to(sem_labels)

                    sorted_indices = torch.argsort(sem_labels)
                    sem_labels = sem_labels[sorted_indices]
                    sem_queries = sem_queries[sorted_indices]

                    sorted_indices = torch.argsort(text_labels)
                    text_labels = text_labels[sorted_indices]
                    sem_prompts = sem_prompts[sorted_indices]

                    assert len(text_labels) == len(sem_labels) and torch.equal(text_labels, sem_labels)

                    sem_prompts = (sem_queries + sem_prompts) / 2.
                    sem_prompts = sem_prompts / (sem_prompts.norm(dim=-1, keepdim=True) + 1e-6)

                else:
                    sem_prompts = sem_queries
                    # sem_prompts = sem_prompts / (sem_prompts.norm(dim=-1, keepdim=True) + 1e-6)
                    sem_labels = sem_labels

            else:
                raise NotImplementedError

            if id_prompts is not None:
                assert not torch.isnan(id_prompts).any()
            if sem_prompts is not None:
                assert not torch.isnan(sem_prompts).any()

            id_prompt_list.append(id_prompts)
            sem_prompt_list.append(sem_prompts)
            id_prompt_label_list.append(id_labels)
            sem_prompt_label_list.append(sem_labels)

        return {
            'id_prompt_list': id_prompt_list,
            'sem_prompt_list': sem_prompt_list,
            'id_prompt_label_list': id_prompt_label_list,
            'sem_prompt_label_list': sem_prompt_label_list
        }

    def merge_feature(self, features, prompts):

        dino_embeddings = features['dino']
        clip_embeddings = features['clip']
        sam_embeddings = features['sam']

        if self. merging_type == 'simple_merging':

            embeddings = torch.cat([dino_embeddings, clip_embeddings, sam_embeddings], dim=1)
            embeddings = self.merging_module(embeddings)

        else:
            raise NotImplementedError

        return embeddings


    def forward(self, x):
        pass

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



def build_encodersoup(args):

    # Image Encoders: DINOv2, SAM, and CLIP-Vision
    dinov2 = build_dinov2(args)

    # SAM
    sam = build_sam(args)

    # CLIP
    clip = build_clip(args)

    return EncoderSoup(
        dinov2,
        clip,
        sam
    )
