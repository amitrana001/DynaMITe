# Copyright (c) Facebook, Inc. and its affiliates.

from .backbone.swin import D2SwinTransformer
from .pixel_decoder.fpn import BasePixelDecoder
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .meta_arch.mask_former_head import MaskFormerHead
from .meta_arch.iterative_mask_former_head import IterativeMaskFormerHead
from .meta_arch.iterative_m2f_head import IterativeM2FHead
from .meta_arch.iterative_m2f_mq_head import IterativeM2FHeadMQ
from .meta_arch.per_pixel_baseline import PerPixelBaselineHead, PerPixelBaselinePlusHead

from .meta_arch.spatio_temp_m2f_mq_head import SpatioTempM2FHeadMQ
from .transformer_decoder.spatio_temp_m2f_mq_transformer_decoder import SpatioTempM2FTransformerDecoderMQ
# from .transformer_decoder.interactive_mask2former_transformer_decoder import InteractiveTransformerDecoder
# from .transformer_decoder.interactive_m2fclicks_transformer_decoder import InteractiveClicksTransformerDecoder
# from .transformer_decoder.iterative_m2f_transformer_decoder import IterativeM2FTransformerDecoder
# from .transformer_decoder.spatio_temp_m2f_transformer_decoder import SpatioTemporalM2FTransformerDecoder
# from .transformer_decoder.spatio_temp_V1_m2f_transformer_decoder import SpatioTemporalV1M2FTransformerDecoder
from .transformer_decoder.iterative_m2f_mq_transformer_decoder import IterativeM2FTransformerDecoderMQ