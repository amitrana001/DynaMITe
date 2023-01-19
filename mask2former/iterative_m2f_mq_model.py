# Copyright (c) Facebook, Inc. and its affiliates.
from itertools import accumulate
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import random
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

# from mask2former.evaluation import iterative_evaluator
from .modeling.final_criterion import SetFinalCriterion
from .modeling.new_criterion import SetNewCriterion
from .modeling.matcher import HungarianMatcher
from mask2former.utils import get_new_scribbles, preprocess_batch_data

@META_ARCH_REGISTRY.register()
class IterativeMask2FormerMQ(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        random_bg_queries: bool,
        iterative_evaluation: bool,
        class_agnostic: bool,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # iterative
        self.iterative_evaluation = iterative_evaluation
        self.class_agnostic = class_agnostic
        self.random_bg_queries = random_bg_queries

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # #Iterative Pipeline
        # iterative_evaluation = cfg.ITERATIVE.INTERACTIVE_EVALAUTION
        class_agnostic = cfg.ITERATIVE.TRAIN.CLASS_AGNOSTIC

        # building criterion

        if class_agnostic:
            weight_dict = {"loss_mask": mask_weight, "loss_dice": dice_weight}
            losses = ["masks"]
        else:
            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
            losses = ["labels", "masks"]

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)


        criterion = SetFinalCriterion(
            sem_seg_head.num_classes,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,

            #iterative
            "iterative_evaluation": cfg.ITERATIVE.TEST.INTERACTIVE_EVALAUTION,
            "class_agnostic": cfg.ITERATIVE.TRAIN.CLASS_AGNOSTIC,
            "random_bg_queries": cfg.ITERATIVE.TRAIN.RANDOM_BG_QUERIES,

            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, images=None, scribbles=None, num_instances=None,
                features=None, mask_features=None, transformer_encoder_features=None, 
                multi_scale_features=None, prev_mask_logits=None, batched_num_scrbs_per_mask= None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        
        if (images is None) or (scribbles is None) or (num_instances is None):
            if self.training:
                images, scribbles, num_instances, batched_num_scrbs_per_mask = preprocess_batch_data(batched_inputs, self.device,
                                                                        self.pixel_mean, self.pixel_std,
                                                                        self.size_divisibility, self.random_bg_queries)
            else:
                images, scribbles, num_instances, batched_num_scrbs_per_mask = preprocess_batch_data(batched_inputs, self.device,
                                                                        self.pixel_mean, self.pixel_std,
                                                                        self.size_divisibility, self.random_bg_queries)
        if features is None:
            features = self.backbone(images.tensor)
        # mask_features, transformer_encoder_features, multi_scale_features, outputs = self.sem_seg_head(features, mask_features, transformer_encoder_features, multi_scale_features, scribbles=scribbles)
        # print(batched_num_scrbs_per_mask[0])
        if self.training:

            # outputs = self.iter_batch_data(batched_inputs, images, features, mask_features, transformer_encoder_features,
            #              multi_scale_features, accumulate_loss, num_instances,scribbles)
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            
            outputs, batched_num_scrbs_per_mask = self.sem_seg_head(batched_inputs, gt_instances, images, features, num_instances,
                                        mask=None, scribbles=scribbles, mask_features=mask_features,
                                        transformer_encoder_features=transformer_encoder_features, 
                                        multi_scale_features=multi_scale_features,
                                        prev_mask_logits=prev_mask_logits, batched_num_scrbs_per_mask=batched_num_scrbs_per_mask)
            losses = self.criterion(outputs, targets, batched_num_scrbs_per_mask)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses

            # losses = self.iter_batch_data(batched_inputs, images, num_instances, scribbles, targets)
            # return losses
        else:
            gt_instances=None
            outputs, mask_features, transformer_encoder_features, multi_scale_features, batched_num_scrbs_per_mask = self.sem_seg_head(batched_inputs, gt_instances, images, features, num_instances,
                                        mask=None, scribbles=scribbles, mask_features=mask_features,
                                        transformer_encoder_features=transformer_encoder_features, 
                                        multi_scale_features=multi_scale_features,
                                        prev_mask_logits=prev_mask_logits, batched_num_scrbs_per_mask=batched_num_scrbs_per_mask)
            processed_results = self.process_results(batched_inputs, images, outputs, num_instances, batched_num_scrbs_per_mask)
            if self.iterative_evaluation:
                return processed_results, outputs, images, scribbles, num_instances, features, mask_features, transformer_encoder_features, multi_scale_features, batched_num_scrbs_per_mask
            else:
                return processed_results

    def process_results(self, batched_inputs, images, outputs, num_instances, batched_num_scrbs_per_mask=None):
        if outputs["pred_logits"] is None:
            mask_cls_results = [None]*(outputs["pred_masks"].shape[0])
        else:
            mask_cls_results = outputs["pred_logits"]
        # mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        if batched_num_scrbs_per_mask is None:
            batched_num_scrbs_per_mask = [[1]*inst_per_image for inst_per_image in num_instances]

        processed_results = []
        for mask_cls_result, mask_pred_result, input_per_image, image_size, inst_per_image, num_scrbs_per_mask in zip(
            mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes, num_instances, batched_num_scrbs_per_mask
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                if mask_cls_result is not None:
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

            # interactive instance segmentation inference
            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.interactive_instance_inference)(mask_cls_result, mask_pred_result, inst_per_image, num_scrbs_per_mask)
                processed_results[-1]["instances"] = instance_r

        return processed_results

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

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def interactive_instance_inference(self, mask_cls, mask_pred, num_instances, num_scrbs_per_mask=None):
        # mask_pred is already processed to have the same shape as original input
        # print("interactive instance inference")
        image_size = mask_pred.shape[-2:]
        result = Instances(image_size)
        # mask (before sigmoid)
        import copy
        num_scrbs_per_mask_copy = copy.deepcopy(num_scrbs_per_mask)
        num_scrbs_per_mask_copy.append(mask_pred.shape[0]-sum(num_scrbs_per_mask))

        temp_out = []
        splited_masks = torch.split(mask_pred, num_scrbs_per_mask_copy, dim=0)
        for m in splited_masks[:-1]:
            temp_out.append(torch.max(m, dim=0).values)
        mask_pred = torch.cat([torch.stack(temp_out),splited_masks[-1]]) # can remove splited_masks[-1] all together

        mask_pred = torch.argmax(mask_pred,0)
        m = []
        for i in range(num_instances):
            m.append((mask_pred == i).float())
        
        mask_pred = torch.stack(m)
        result.pred_masks = mask_pred
        # mask_pred = mask_pred[:num_instances]
        # result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        if mask_cls is None:
            result.scores = torch.zeros((num_instances))
            result.pred_classes = torch.zeros((num_instances))
        else:
            # [Q, K+1] -> [fg, K]
            temp_out = []
            splited_scores = torch.split(mask_cls, num_scrbs_per_mask_copy, dim=0)
            for m in splited_scores[:-1]:
                temp_out.append(torch.max(m, dim=0).values)
            mask_cls = torch.cat([torch.stack(temp_out),splited_scores[-1]])

            scores = F.softmax(mask_cls, dim=-1)[:num_instances, :-1]
            labels_per_image = scores.argmax(dim=1)

            scores_per_image = scores.max(dim=1)[0]
            mask_pred = mask_pred[:num_instances]
            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
        return result
