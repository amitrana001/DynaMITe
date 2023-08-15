# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.config import configurable

from IPython import embed

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@SEM_SEG_HEADS_REGISTRY.register()
class SegFormerPixelDecoder(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    @configurable
    def __init__(self, embedding_dim, feature_strides, in_channels, num_classes, in_index, dropout_ratio):
        # super(SegFormerPixelDecoder, self).__init__()
        super().__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.in_channels = in_channels
        self.in_index = in_index
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.dropout_ratio = dropout_ratio

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # decoder_params = kwargs['decoder_params']
        # embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=self.embedding_dim*4,
            out_channels=self.embedding_dim,
            kernel_size=1,
            #norm_cfg=dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        # self.linear_pred = nn.Conv2d(self.embedding_dim, self.num_classes, kernel_size=1)
        
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout = None
    
    @classmethod
    def from_config(cls, cfg, input_shape=None):
        ret = {}
    
        ret["embedding_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["feature_strides"] = cfg.MODEL.SEM_SEG_HEAD.FEATURES_STRIDES_SEGFORMER
        ret["in_channels"] = cfg.MODEL.SEM_SEG_HEAD.IN_CHANNELS_SEGFORMER
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES_SEGFORMER
        ret["in_index"] = cfg.MODEL.SEM_SEG_HEAD.IN_INDEXES_SEGFORMER
        ret["dropout_ratio"] = cfg.MODEL.SEM_SEG_HEAD.DROPOUT_RATIO_SEGFORMER
        
        return ret

    def forward_features(self, inputs):
        #x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        # x =  [inputs[i] for i in self.in_index]
        x =  [inputs[k] for k in inputs.keys()]
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        multi_scale_features = []

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        multi_scale_features.append(_c4)
        
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        multi_scale_features.append(_c3)

        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        multi_scale_features.append(_c2)

        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        feature = x
        # x = self.linear_pred(x)
        # return x, feature, multi_scale_features
        return feature, feature ,multi_scale_features


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           ):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)