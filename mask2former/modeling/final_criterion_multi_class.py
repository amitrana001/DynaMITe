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


from typing import Optional
from torch import Tensor
import torch


def multiclass_dice_loss(input: Tensor, target: Tensor, eps: float = 1e-6,
                         check_target_validity: bool = True, ignore_zero_class: bool = False,
                         ignore_mask: Optional[Tensor] = None) -> Tensor:
    """
    Computes DICE loss for multi-class predictions. API inputs are identical to torch.nn.functional.cross_entropy()
    :param input: tensor of shape [N, C, *] with unscaled logits
    :param target: tensor of shape [N, *]
    :param eps:
    :param check_target_validity: checks if the values in the target are valid
    :param ignore_zero_class: Ignore the IoU for class ID 0
    :param ignore_mask: optional tensor of shape [N, *]
    :return: tensor
    """
    assert input.ndim >= 2
    input = input.softmax(1)
    # num_classes = input.size(1)
    return dice_loss(input, target, eps=eps, ignore_mask=ignore_mask)


def dice_loss_with_logits(input: Tensor, target: Tensor, ignore_mask: Optional[Tensor] = None,
                          eps: Optional[float] = 1e-6):
    return dice_loss(input.sigmoid(), target, ignore_mask, eps)


def dice_loss(input: Tensor, target: Tensor, ignore_mask: Optional[Tensor] = None, eps: Optional[float] = 1e-6):
    """
    Computes the DICE or soft IoU loss.
    :param input: tensor of shape [N, *]
    :param target: tensor with shape identical to input
    :param ignore_mask: tensor of same shape as input. non-zero values in this mask will be
    :param eps
    excluded from the loss calculation.
    :return: tensor
    """
    assert input.shape == target.shape, f"Shape mismatch between input ({input.shape}) and target ({target.shape})"
    # assert input.dtype == target.dtype

    if torch.is_tensor(ignore_mask):
        assert ignore_mask.dtype == torch.bool
        assert input.shape == ignore_mask.shape, f"Shape mismatch between input ({input.shape}) and " \
            f"ignore mask ({ignore_mask.shape})"
        input = torch.where(ignore_mask, torch.zeros_like(input), input)
        target = torch.where(ignore_mask, torch.zeros_like(target), target)

    input = input.flatten(1)
    target = target.detach().flatten(1)

    numerator = 2.0 * (input * target).mean(1)
    denominator = (input + target).mean(1)

    soft_iou = (numerator + eps) / (denominator + eps)

    return torch.where(numerator > eps, 1. - soft_iou, soft_iou * 0.).mean()



# dice_loss_jit = torch.jit.script(
#     multiclass_dice_loss
# )  # type: torch.jit.ScriptModule


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
    targets = torch.argmax(targets,1)
    loss = F.cross_entropy(inputs, targets)

    return loss


# sigmoid_ce_loss_jit = torch.jit.script(
#     sigmoid_ce_loss
# )  # type: torch.jit.ScriptModule


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


class SetFinalCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, weight_dict, eos_coef, losses,
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

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    

    def bg_mask_loss(self, outputs, targets):
        indices = []
        out = []
        import copy 
        t_copy = copy.deepcopy(targets)
        for i,t in enumerate(t_copy):
            # num_gt_classes = len(t['labels']) # 1 for bg_mask
            num_gt_classes = t['masks'].shape[0]
            indxs = torch.tensor(range(num_gt_classes+1))
            indices.append((indxs, indxs)) 
            # print(t["masks"].dtype)
            target_bg_mask = 1 - torch.max(t["masks"],dim=0).values
            # t['masks'] from #instxHxW -> (#inst+1)xHxW
            t_copy[i]["masks"] = torch.cat((target_bg_mask.unsqueeze(0), t["masks"]), dim=0)

            #output bg_mask
            if outputs["pred_masks"][i].shape[0] > num_gt_classes:
                out_bg_mask = torch.max(outputs["pred_masks"][i][num_gt_classes:,], dim=0).values.unsqueeze(0)
                # outputs["pred_masks"][i][num_gt_classes:,].max(dim=0)[0]
                out.append(torch.cat((out_bg_mask, outputs["pred_masks"][i][:num_gt_classes]), 0))
            else:
                out.append(outputs["pred_masks"][i])
        outputs["pred_masks"] = torch.stack(out)

        return outputs, t_copy, indices


    def loss_masks(self, outputs, targets, indices, num_masks, bg_loss = True, batched_num_scrbs_per_mask = None):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        # print("before:",outputs['pred_masks'].shape)
        # print("before targets", targets[0]['masks'].shape)
        # # if bg_loss:
        # print("bg_maks loss")
        outputs, t_copy, indices = self.bg_mask_loss(outputs,targets)
        num_masks += outputs["pred_masks"].shape[0] #batch_size == #bg_masks

        mask_pred_results = F.interpolate(
            outputs['pred_masks'],
            size=(t_copy[0]['masks'].shape[-2], t_copy[0]['masks'].shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        # print("after",mask_pred_results.shape)
        t_copy = [t["masks"] for t in t_copy]
        t_copy = torch.stack(t_copy,0)
        # print(t_copy.shape)
        # print(t_copy.dtype, mask_pred_results.dtype)
        t_copy = t_copy.to(dtype=mask_pred_results.dtype)
        # print(t_copy.dtype, mask_pred_results.dtype)
        losses = {
            "loss_mask": sigmoid_ce_loss(mask_pred_results, t_copy, num_masks),
            "loss_dice": multiclass_dice_loss(mask_pred_results, t_copy),
        }

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

    def get_loss(self, loss, outputs, targets, indices, num_masks, batched_num_scrbs_per_mask=None):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, batched_num_scrbs_per_mask)

    def forward(self, outputs, targets, batched_num_scrbs_per_mask = None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # already know the matching between the outputs of the last layer and the targets
        #use them as indices
        indices = []
        for t in targets:
            num_gt_classes = len(t['labels'])
            indxs = torch.tensor(range(num_gt_classes))
            indices.append((indxs, indxs)) 

        # indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=outputs['pred_masks'].device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
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
