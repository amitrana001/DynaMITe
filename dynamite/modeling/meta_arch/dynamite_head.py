#Modified by Amit Rana from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/meta_arch/mask_former_head.py

from typing import Dict
from torch import nn
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..pixel_decoder.fpn import build_pixel_decoder
from ..interactive_transformer.utils import build_interactive_transformer

@SEM_SEG_HEADS_REGISTRY.register()
class DynamiteHead(nn.Module):

    _version = 2

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        pixel_decoder: nn.Module,
        # extra parameters
        interactive_transformer: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            pixel_decoder: the pixel decoder module
            transformer_predictor: the interactive transformer that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]

        self.common_stride = 4

        self.pixel_decoder = pixel_decoder
        self.interactive_transformer = interactive_transformer

        self.transformer_in_feature = transformer_in_feature


    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  
            interactive_transformer_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "interactive_transformer": build_interactive_transformer(
                cfg,
                interactive_transformer_in_channels,
            ),
        }

    def forward(self, data, images, features, num_instances, mask_features=None,
                multi_scale_features=None, num_clicks_per_object=None,
                fg_coords = None, bg_coords = None, 
                max_timestamp=None):
        
        if (mask_features is None) or (multi_scale_features is None):
            mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(features)

        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions, num_clicks_per_object = self.interactive_transformer(data, images, num_instances, multi_scale_features,
                                        mask_features,  num_clicks_per_object,
                                        fg_coords, bg_coords, max_timestamp)
        if self.training:
            return predictions, num_clicks_per_object
        else:
            return predictions, mask_features,  multi_scale_features, num_clicks_per_object