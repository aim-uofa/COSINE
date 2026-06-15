# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / (num_masks + 1e-8)


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / (num_masks + 1e-8)


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices_id, indices_ins, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_id_logits" in outputs[0] and "pred_ins_logits" in outputs[0]
        src_id_logits = [out["pred_id_logits"].float() for out in outputs]
        src_ins_logits = [out["pred_ins_logits"].float() for out in outputs]

        # ins
        target_ins_classes_o = [t["labels_t"][J] for t, (_, J) in zip(targets, indices_ins)]
        target_ins_classes = [torch.full(
            src_ins_logit.shape[:2], out["num_classes"], dtype=torch.int64, device=src_ins_logit.device
        ) for src_ins_logit, out in zip(src_ins_logits, outputs)]

        for target_ins_class, target_ins_class_o, (src, _) in zip(target_ins_classes, target_ins_classes_o, indices_ins):
            target_ins_class[:,src] = target_ins_class_o

        empty_weights = []
        for out in outputs:
            weight = torch.ones(out["num_classes"] + 1).to(self.empty_weight)
            weight[-1] = self.eos_coef
            empty_weights.append(weight)

        loss_ce_ins = [
            F.cross_entropy(src_ins_logit.transpose(1, 2).to(self.empty_weight), target_ins_class, weight) if src_ins_logit.shape[-1] > 1 else src_ins_logit.sum() * 0.
            for src_ins_logit, target_ins_class, weight in zip(src_ins_logits, target_ins_classes, empty_weights)
        ]
        loss_ce_ins = sum(loss_ce_ins) / len(loss_ce_ins)

        # id
        target_id_classes_o = [t["ids"][J] for t, (_, J) in zip(targets, indices_id)]
        target_id_classes_o = [torch.zeros_like(tco) for tco in target_id_classes_o]

        target_id_classes = [torch.full(
            src_id_logit.shape[:2], self.num_classes, dtype=torch.int64, device=src_id_logit.device
        ) for src_id_logit in src_id_logits]

        try:
            for target_id_class, target_id_class_o, (src, _) in zip(target_id_classes, target_id_classes_o, indices_id):
                target_id_class[:,src] = target_id_class_o
        except RuntimeError:
            print()

        src_id_logits = torch.cat(src_id_logits, dim=1)
        target_id_classes = torch.cat(target_id_classes, dim=1)

        loss_ce_id = F.cross_entropy(src_id_logits.transpose(1, 2).to(self.empty_weight.dtype), target_id_classes, self.empty_weight) if not src_id_logits.numel() == 0 else src_id_logits.sum() * 0.
        losses = {
            "loss_ce_ins": loss_ce_ins,
            "loss_ce_id": loss_ce_id,
        }

        return losses


    def loss_masks(self, outputs, targets, indices_id, indices_ins, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_ins_masks" in outputs[0] and "pred_id_masks" in outputs[0]
        src_ins_masks = torch.cat([out["pred_ins_masks"][0, src] for out, (src, _) in zip(outputs, indices_ins)])
        src_id_masks = torch.cat([out["pred_id_masks"][0, src] for out, (src, _) in zip(outputs, indices_id)])

        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_ins_masks = torch.cat([target_mask[tat] for target_mask, (_, tat) in zip(target_masks, indices_ins)])
        target_ins_masks = target_ins_masks.to(src_ins_masks)
        target_id_masks = torch.cat([target_mask[tat] for target_mask, (_, tat) in zip(target_masks, indices_id)])
        target_id_masks = target_id_masks.to(src_id_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = torch.cat([src_ins_masks, src_id_masks], dim=0)[:,None]
        target_masks = torch.cat([target_ins_masks, target_id_masks], dim=0)[:,None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks.float(),
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            ).to(target_masks.dtype)
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices_id, indices_ins, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }

        new_outputs = []
        for out in outputs:

            new_outputs.append(
                {
                    "pred_id_logits": out["pred_id_logits"],
                    "pred_ins_logits": out["pred_ins_logits"],
                    "pred_id_masks": out["pred_id_masks"],
                    "pred_ins_masks": out["pred_ins_masks"],
                    "num_classes": out["ref_sem_labels_t"].shape[0]
                }
            )

        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](new_outputs, targets, indices_id, indices_ins, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = [{k: v for k, v in output.items() if k != "aux_outputs"} for output in outputs]
        # Retrieve the matching between the outputs of the last layer and the targets
        indices_id, indices_ins = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks_id = sum([len(i[0]) for i in indices_id])
        num_masks_ins = sum([len(i[0]) for i in indices_ins])
        num_masks = num_masks_id + num_masks_ins
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs[0].values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices_id, indices_ins, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs[0]:

            for i in range(len(outputs[0]["aux_outputs"])):
                aux_outputs = [
                    {
                        'ref_id_labels': out['ref_id_labels'],
                        'ref_sem_labels': out['ref_sem_labels'],
                        'pred_id_logits': out['aux_outputs'][i]['pred_id_logits'],
                        'pred_ins_logits': out['aux_outputs'][i]['pred_ins_logits'],
                        'pred_id_masks': out['aux_outputs'][i]['pred_id_masks'],
                        'pred_ins_masks': out['aux_outputs'][i]['pred_ins_masks'],
                        'ref_sem_labels_t': out['ref_sem_labels_t'],
                    } for out in outputs
                ]
                indices_id, indices_ins = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_id, indices_ins, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
