# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F
import torch
from detectron2.modeling.postprocessing import sem_seg_postprocess

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import Boxes, ImageList, Instances, BitMasks


from ..transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder
from ..pixel_decoder.fpn import build_pixel_decoder
from mask2former.utils import get_new_scribbles, preprocess_batch_data

@SEM_SEG_HEADS_REGISTRY.register()
class IterativeMaskFormerHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
        }

    def forward(self, data, targets, images, features, num_instances, is_train, accumulate_loss, mask=None, scribbles=None):
        # print("Scribbles:",scribbles.shape)
        return self.layers(data, targets, images, features, num_instances, is_train, accumulate_loss, scribbles = scribbles)

    def layers(self, data, targets, images, features, num_instances, is_train, accumulate_loss, mask=None, scribbles=None):
        
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            # print("Scribbles:",scribbles.shape)
            # predictions = self.predictor(multi_scale_features, mask_features, mask, scribbles = scribbles)
            predictions = self.predictor(data, targets, images, num_instances, is_train, accumulate_loss,
                                                         multi_scale_features, mask_features, mask, scribbles)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        return predictions

    # def interactive_batch_forward(self, data, images, num_instances, is_train, accumulate_loss,
    #                               multi_scale_features, mask_features, mask, scribbles):
    #     if is_train:
    #         if not accumulate_loss:
                
    #             prev_output = None
    #             # mask_features, transformer_encoder_features, multi_scale_features, outputs = self.sem_seg_head(features, mask_features, transformer_encoder_features, multi_scale_features, scribbles=scribbles)
    #             for _ in range(1):
    #                 # self.predictor.eval()
    #                 prev_output = self.predictor(multi_scale_features, mask_features, mask,  scribbles=scribbles)
    #                 processed_results = self.process_results(data, images, prev_output, num_instances)
    #                 scribbles = get_new_scribbles(data, processed_results, scribbles)
    #                 # self.predictor.train()
    #             outputs = self.predictor(multi_scale_features, mask_features, mask, scribbles=scribbles)
            
    #     return outputs
    
    def process_results(self, batched_inputs, images, outputs, num_instances):
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result, input_per_image, image_size, inst_per_image in zip(
            mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes, num_instances
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            # if self.sem_seg_postprocess_before_inference:
            #     mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
            #         mask_pred_result, image_size, height, width
            #     )
            #     mask_cls_result = mask_cls_result.to(mask_pred_result)

            # # semantic segmentation inference
            # if self.semantic_on:
            #     r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
            #     if not self.sem_seg_postprocess_before_inference:
            #         r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
            #     processed_results[-1]["sem_seg"] = r

            # # panoptic segmentation inference
            # if self.panoptic_on:
            #     panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
            #     processed_results[-1]["panoptic_seg"] = panoptic_r
            
            # # # instance segmentation inference
            # # if self.instance_on:
            # #     instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
            # #     processed_results[-1]["instances"] = instance_r
            
            # # interactive instance segmentation inference
            # if self.instance_on:
            instance_r = retry_if_cuda_oom(self.interactive_instance_inference)(mask_cls_result, mask_pred_result, inst_per_image)
            processed_results[-1]["instances"] = instance_r

        return processed_results

    def interactive_instance_inference(self, mask_cls, mask_pred, num_instances):
        # mask_pred is already processed to have the same shape as original input
        # print("interactive instance inference")
        image_size = mask_pred.shape[-2:]

        # [Q, K+1] -> [fg, K]
        scores = F.softmax(mask_cls, dim=-1)[:num_instances, :-1]
        labels_per_image = scores.argmax(dim=1)

        # labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        # labels_per_image = labels[topk_indices]
        scores_per_image = scores.max(dim=1)[0]
        # topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[:num_instances]

        # if this is panoptic segmentation, we only keep the "thing" classes
        # if self.panoptic_on:
        #     keep = torch.zeros_like(scores_per_image).bool()
        #     for i, lab in enumerate(labels_per_image):
        #         keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

        #     scores_per_image = scores_per_image[keep]
        #     labels_per_image = labels_per_image[keep]
        #     mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result