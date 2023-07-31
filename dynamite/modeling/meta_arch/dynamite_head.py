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

from ..transformer_decoder import build_transformer_decoder
from ..pixel_decoder.fpn import build_pixel_decoder

@SEM_SEG_HEADS_REGISTRY.register()
class DynamiteHead(nn.Module):

    _version = 2

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

    def forward(self, data, targets, images, features, num_instances, mask=None, scribbles=None,
                mask_features=None, transformer_encoder_features=None, 
                multi_scale_features=None, prev_mask_logits=None, batched_num_scrbs_per_mask=None,
                batched_fg_coords_list = None, batched_bg_coords_list = None, batched_max_timestamp=None):
        
        return self.layers(data, targets, images, features, num_instances, mask, scribbles, mask_features,
                transformer_encoder_features, multi_scale_features, prev_mask_logits,
                batched_num_scrbs_per_mask, batched_fg_coords_list, batched_bg_coords_list,  batched_max_timestamp)

    def layers(self, data, targets, images, features, num_instances, mask=None, scribbles=None,
               mask_features=None, transformer_encoder_features=None, 
               multi_scale_features=None, prev_mask_logits=None,batched_num_scrbs_per_mask=None,
               batched_fg_coords_list = None, batched_bg_coords_list = None,  batched_max_timestamp=None):
        
        if (mask_features is None) or (multi_scale_features is None):
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)

        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions, batched_num_scrbs_per_mask = self.predictor(data, targets, images, num_instances, multi_scale_features,
                                        mask_features, mask, scribbles, prev_mask_logits, batched_num_scrbs_per_mask,
                                        batched_fg_coords_list, batched_bg_coords_list,batched_max_timestamp)
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
        if self.training:
            return predictions, batched_num_scrbs_per_mask
        else:
            return predictions, mask_features, transformer_encoder_features, multi_scale_features, batched_num_scrbs_per_mask