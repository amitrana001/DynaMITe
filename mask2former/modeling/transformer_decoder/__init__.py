# Copyright (c) Facebook, Inc. and its affiliates.
from .maskformer_transformer_decoder import StandardTransformerDecoder
from .mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder

#added interactive transformer decoder
from .interactive_m2fclicks_transformer_decoder import InteractiveClicksTransformerDecoder
from .interactive_mask2former_transformer_decoder import InteractiveTransformerDecoder
from .interactive_hodor import InteractiveHodor
from .iterative_m2f_transformer_decoder import IterativeM2FTransformerDecoder
# from .descriptor_initializer import AvgPoolingInitializer
# from .position_encoding import PositionEmbeddingSine