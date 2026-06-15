# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, tar_targets):
        """More memory-friendly matching"""
        bs = len(outputs)
        indices_id = []
        indices_ins = []

        # Iterate through batch size
        for b in range(bs):

            output = outputs[b]
            # ref_target = ref_targets[b]
            nid = output["pred_id_logits"].shape[1]
            nq = output["pred_ins_logits"].shape[1]
            nc = output['ref_sem_labels'].shape[1]

            if nc > 0:
                # match ins
                ref_sem_labels = output['ref_sem_labels_t']
                tgt_ids = tar_targets[b]["labels_t"]

                out_ins_logits = output["pred_ins_logits"][0]  # nq, nc+1
                out_ins_prob = out_ins_logits.softmax(-1)
                cost_ins_class = -out_ins_prob[:, tgt_ids]

                out_ins_masks = output["pred_ins_masks"][0]  # nq, h, w
                # gt masks are already padded when preparing target
                tgt_mask = tar_targets[b]["masks"].to(out_ins_masks)

                out_ins_masks = out_ins_masks[:, None]
                tgt_mask = tgt_mask[:, None]
                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.num_points, 2, device=out_ins_masks.device)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_ins_masks = point_sample(
                    out_ins_masks,
                    point_coords.repeat(out_ins_masks.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                with autocast(enabled=False):
                    out_ins_masks = out_ins_masks.float()
                    tgt_mask = tgt_mask.float()
                    # Compute the focal loss between masks
                    cost_ins_mask = batch_sigmoid_ce_loss_jit(out_ins_masks, tgt_mask)

                    # Compute the dice loss betwen masks
                    cost_ins_dice = batch_dice_loss_jit(out_ins_masks, tgt_mask)

                # Final cost matrix
                C_ins = (
                    self.cost_mask * cost_ins_mask
                    + self.cost_class * cost_ins_class
                    + self.cost_dice * cost_ins_dice
                )
                C_ins = C_ins.reshape(nq, -1).cpu().detach().numpy()

                try:
                    indices_ins.append(linear_sum_assignment(C_ins))
                except ValueError:
                    print()
            else:
                indices_ins.append((torch.empty(0, dtype=torch.int64).numpy(), (torch.empty(0, dtype=torch.int64).numpy())))

            # id
            # ref_mask_ids = ref_target['ids']
            ref_mask_ids = output['ref_id_labels'][0]
            tar_mask_ids = tar_targets[b]["ids"]

            tar_unique_ids_flag = torch.isin(tar_mask_ids, ref_mask_ids)

            intersection_ids = tar_mask_ids[tar_unique_ids_flag]
            indices_tar = torch.where(tar_unique_ids_flag)[0]
            indices_ref = torch.where(torch.isin(ref_mask_ids, intersection_ids))[0]
            indices_id.append((indices_ref.cpu().numpy(), indices_tar.cpu().numpy()))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_id
        ], [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_ins
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            # "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
