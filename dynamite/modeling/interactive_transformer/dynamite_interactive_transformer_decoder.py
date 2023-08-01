# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import os
import time as timer
import random
import einops
import numpy as np
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch

from detectron2.utils.registry import Registry
import detectron2.utils.comm as comm
from torch import nn, Tensor
from torch.nn import functional as F
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.config import configurable
from detectron2.layers import Conv2d
from einops import repeat
from .position_encoding import PositionEmbeddingSine
from .descriptor_initializer import AvgClicksPoolingInitializer
from dynamite.utils.train_utils import get_next_clicks, get_pos_tensor_coords, get_spatiotemporal_embeddings

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in MaskFormer.
"""

def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, attn_map=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_map = attn_map
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, attn_map=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_map = attn_map
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
       
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
       
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
       
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class InteractiveTransformerDecoder(nn.Module):

    _version = 2

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        max_num_interactions: int,
        accumulate_loss: bool,
        class_agnostic: bool,
        concat_coord_mask_features: bool,
        concat_coord_image_features: bool,
        random_bg_queries: bool,
        query_initializer: str,
        use_pos_coords: bool,
        use_time_coords: bool,
        unique_timestamp: bool,
        concat_xyt:bool,
        use_only_time:bool,
        use_argmax: bool,
        use_rev_cross_attn: bool,
        use_mlp_rev_attn: bool,
        rev_cross_attn_num_layers: int,
        rev_cross_attn_scale: float,
        use_static_bg_queries: bool,
        num_static_bg_queries: int,
        per_obj_sampling: bool,
        use_coords_on_point_mask: bool,
        use_point_features: bool,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
         # iterative
        self.max_num_interactions = max_num_interactions
        self.accumulate_loss = accumulate_loss
        self.class_agnostic = class_agnostic
        self.concat_coord_mask_features = concat_coord_mask_features
        self.concat_coord_image_features = concat_coord_image_features
        self.random_bg_queries = random_bg_queries
        self.query_initializer = query_initializer
        self.num_static_bg_queries = num_static_bg_queries
        
        self.per_obj_sampling = per_obj_sampling
        self.use_time_coords = use_time_coords
        self.unique_timestamp = unique_timestamp
        self.concat_xyt = concat_xyt
        self.use_only_time = use_only_time

        self.use_argmax = use_argmax
        self.use_coords_on_point_mask = use_coords_on_point_mask
        self.use_point_features = use_point_features

        # Reverse Cross Attn
        self.use_rev_cross_attn = use_rev_cross_attn
        self.rev_cross_attn_num_layers = rev_cross_attn_num_layers
        self.rev_cross_attn_scale = rev_cross_attn_scale

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.use_mlp_rev_attn = use_mlp_rev_attn
        if self.use_rev_cross_attn:
            self.rev_cross_attn_layers = nn.ModuleList()
            for _ in range(self.rev_cross_attn_num_layers):
                self.rev_cross_attn_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )
            if self.use_mlp_rev_attn:
                self.rev_cross_attn_ffn_layers = nn.ModuleList()
                for _ in range(self.rev_cross_attn_num_layers):
                    self.rev_cross_attn_ffn_layers.append(
                        FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=hidden_dim*2,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )
            # positional encoding for mask features
            N_steps = hidden_dim // 2
            self.pe_mask_features = PositionEmbeddingSine(N_steps, normalize=True)

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.multi_scale = True if query_initializer == "multi_scale" else False
        self.query_descriptors_initializer = AvgClicksPoolingInitializer()
        
        self.queries_nonlinear_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.use_pos_coords = use_pos_coords
        if self.concat_xyt:
            self.ca_qpos_sine_proj = nn.Linear(3*(hidden_dim//2), hidden_dim)
        elif self.use_pos_coords:
            self.ca_qpos_sine_proj = nn.Linear(hidden_dim, hidden_dim)
        # self.descriptor_projection = nn.linear()
        # learnable query p.e.
        self.register_parameter("query_embed", nn.Parameter(torch.zeros(hidden_dim), True))
        # self.query_embed = nn.Parameter(torch.zeros(hidden_dim), True)
        # self.query_embed = nn.Embedding(1,hidden_dim)
        self.use_static_bg_queries = use_static_bg_queries
        if self.use_static_bg_queries:
            self.register_parameter("static_bg_pe", nn.Parameter(torch.zeros(self.num_static_bg_queries, hidden_dim), True))
            self.register_parameter("static_bg_query", nn.Parameter(torch.zeros(self.num_static_bg_queries,hidden_dim), True))
            self.register_parameter("bg_query", nn.Parameter(torch.zeros(hidden_dim), False))
        else:
            self.register_parameter("bg_query", nn.Parameter(torch.zeros(hidden_dim), False))

        # if self.un
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        if not self.class_agnostic:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.query_embed)
        if self.use_static_bg_queries:
            nn.init.normal_(self.static_bg_pe)
            # # # nn.init.kaiming_uniform_(self.static_bg_query, a=1)
            nn.init.xavier_uniform_(self.static_bg_query)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        ret["query_initializer"] = cfg.MODEL.MASK_FORMER.QUERY_INITIALIZER
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # Reverse Cross Attention
        ret["use_rev_cross_attn"] = cfg.REVERSE_CROSS_ATTN.USE_REVERSE_CROSS_ATTN
        ret["use_mlp_rev_attn"] = cfg.REVERSE_CROSS_ATTN.USE_MLP_REV_ATTN
        ret["rev_cross_attn_num_layers"] = cfg.REVERSE_CROSS_ATTN.NUM_LAYERS
        ret["rev_cross_attn_scale"] = cfg.REVERSE_CROSS_ATTN.SCALE_FACTOR

        ret["use_argmax"] =  cfg.ITERATIVE.TRAIN.USE_ARGMAX
        # Iterative Pipeline
        ret["max_num_interactions"] = cfg.ITERATIVE.TRAIN.MAX_NUM_INTERACTIONS
        ret["accumulate_loss"] = cfg.ITERATIVE.TRAIN.ACCUMULATE_INTERACTION_LOSS
        ret["class_agnostic"] = cfg.ITERATIVE.TRAIN.CLASS_AGNOSTIC
        ret["concat_coord_mask_features"] =cfg.ITERATIVE.TRAIN.CONCAT_COORD_MASK_FEATURES
        ret["concat_coord_image_features"]=cfg.ITERATIVE.TRAIN.CONCAT_COORD_IMAGE_FEATURES
        ret["random_bg_queries"]=cfg.ITERATIVE.TRAIN.RANDOM_BG_QUERIES
        ret["use_pos_coords"] = cfg.ITERATIVE.TRAIN.USE_POS_COORDS
        ret["use_time_coords"] = cfg.ITERATIVE.TRAIN.USE_TIME_COORDS
        ret["unique_timestamp"] =  cfg.ITERATIVE.TRAIN.UNIQUE_TIMESTAMP
        ret["concat_xyt"] = cfg.ITERATIVE.TRAIN.CONCAT_XYT
        ret["use_only_time"] = cfg.ITERATIVE.TRAIN.USE_ONLY_TIME

        ret["use_static_bg_queries"] = cfg.ITERATIVE.TRAIN.USE_STATIC_BG_QUERIES
        ret["num_static_bg_queries"] = cfg.ITERATIVE.TRAIN.NUM_STATIC_BG_QUERIES
        ret["per_obj_sampling"] = cfg.ITERATIVE.TRAIN.PER_OBJ_SAMPLING
        ret["use_coords_on_point_mask"] = cfg.ITERATIVE.TRAIN.USE_COORDS_ON_POINT_MASKS
        ret["use_point_features"] = cfg.ITERATIVE.TRAIN.USE_POINT_FEATURES
        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, data, images, num_instances, x, mask_features, batched_num_scrbs_per_mask=None,
                 batched_fg_coords_list = None, batched_bg_coords_list = None, batched_max_timestamp=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        # del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        if self.training:
            if self.accumulate_loss:
                prev_output = None
                num_iters = random.randint(0,self.max_num_interactions)
               
                batched_max_timestamp = None
                if self.unique_timestamp:
                    batched_max_timestamp = []
                    bs = len(batched_num_scrbs_per_mask)
                    for j in range(bs):
                        if batched_bg_coords_list[j]:
                            batched_max_timestamp.append(batched_bg_coords_list[j][-1][2])
                        else:
                            batched_max_timestamp.append(batched_fg_coords_list[j][-1][-1][2])

                for i in range(num_iters):
                    prev_output = self.iterative_batch_forward(x, src, pos, size_list, mask_features, batched_fg_coords_list, 
                                                               batched_bg_coords_list, batched_max_timestamp
                    )
                    # prev_mask_logits = prev_output['pred_masks']
                    processed_results = self.process_results(data, images, prev_output, num_instances, batched_num_scrbs_per_mask)
                                    
                    next_coords_info = get_next_clicks(data, processed_results,i+1, batched_num_scrbs_per_mask,batched_fg_coords_list, 
                                                       batched_bg_coords_list, batched_max_timestamp = batched_max_timestamp)
                    
                    
                    (batched_num_scrbs_per_mask,  batched_fg_coords_list, batched_bg_coords_list, batched_max_timestamp) = next_coords_info
                       
                        
            outputs = self.iterative_batch_forward(x, src, pos, size_list, mask_features, batched_fg_coords_list, 
                                                    batched_bg_coords_list, batched_max_timestamp
                    )
        else:
            outputs = self.iterative_batch_forward(x, src, pos, size_list, mask_features, batched_fg_coords_list, batched_bg_coords_list,
                                                   batched_max_timestamp)
        return outputs, batched_num_scrbs_per_mask

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        if self.class_agnostic:
            outputs_class = None
        else:
            outputs_class = self.class_embed(decoder_output)

        mask_embed = self.mask_embed(decoder_output)
        if self.concat_coord_mask_features:
            bs= mask_embed.shape[0]
            mask_embed = einops.rearrange(mask_embed, "B Q C -> (B Q) C")
            outputs_mask = torch.einsum("tc,tchw->thw", mask_embed, mask_features) #t = B*Q
            outputs_mask = einops.rearrange(outputs_mask, "(B Q) H W -> B Q H W", B=bs)    
        else:
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    def iterative_batch_forward(self, x, src, pos, size_list, mask_features, batched_fg_coords_list=None,
                                 batched_bg_coords_list=None, batched_max_timestamp=None):

        _, bs, _ = src[0].shape
        B, C, H, W = mask_features.shape
        height = 4*H
        width = 4*W
        
        descriptors = self.query_descriptors_initializer(x, batched_fg_coords_list, batched_bg_coords_list)
        max_queries_batch = max([desc.shape[1] for desc in descriptors])
        for i, desc in enumerate(descriptors):
            if self.use_static_bg_queries:
                bg_queries = repeat(self.bg_query, "C -> 1 L C", L=max_queries_batch-desc.shape[1])
            else:
                bg_queries = repeat(self.bg_query, "C -> 1 L C", L=max_queries_batch+1-desc.shape[1])
            # bg_queries = repeat(self.bg_query, "C -> 1 L C", L=self.num_static_bg_queries)
            descriptors[i] = torch.cat((descriptors[i], bg_queries), dim=1)
        output = torch.cat(descriptors, dim=0)
        
        query_embed = repeat(self.query_embed, "C -> Q N C", N=bs, Q=output.shape[1])
        if self.use_pos_coords:
            scrbs_coords = get_pos_tensor_coords(batched_fg_coords_list, batched_bg_coords_list,
                                                    output.shape[1], height, width, output.device, batched_max_timestamp=batched_max_timestamp
                            ) # bsxQx3
            pos_coord_embed = get_spatiotemporal_embeddings(scrbs_coords.permute(1,0,2), use_timestamp=self.use_time_coords, use_only_time=self.use_only_time,
                                                                concat_xyt=self.concat_xyt) # Q x bs x C
            pos_coord_embed = self.ca_qpos_sine_proj(pos_coord_embed.to(query_embed.dtype))
            
            query_embed = query_embed + pos_coord_embed
        if self.use_static_bg_queries:
            static_bg_pe = repeat(self.static_bg_pe, "Bg C -> Bg N C", N=bs)
            query_embed = torch.cat((query_embed,static_bg_pe),dim=0)
            static_bg_queries = repeat(self.static_bg_query, "Bg C -> N Bg C", N=bs)
            output = torch.cat((output,static_bg_queries), dim=1)
    
        # num_scrbs = output.shape[0]
        Bs, num_scrbs, _ = output.shape
        # NxQxC -> QxNxC
        output = self.queries_nonlinear_projection(output).permute(1,0,2)
        # query positional embedding QxNxC
        # query_embed = repeat(self.query_embed, "C -> Q N C", N=bs, Q=num_scrbs)

        # query_embed = None
        predictions_class = []
        predictions_mask = []
       
        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        if self.use_rev_cross_attn:
            if self.rev_cross_attn_scale > 1:
                scale_factor = self.rev_cross_attn_scale
                mask_features = F.interpolate(mask_features, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            
            with torch.no_grad():
                pos_encodings = self.pe_mask_features(mask_features)
                pos_encodings = einops.rearrange(pos_encodings,"B C H W -> (H W) B C")

            B, C, H, W = mask_features.shape
            mask_features = einops.rearrange(mask_features,"B C H W -> (H W) B C")

            #output is QxNxC
            if self.use_mlp_rev_attn:
                for i in range(self.rev_cross_attn_num_layers):
                    mask_features = self.rev_cross_attn_layers[i](
                        mask_features, output,
                        memory_mask=None,
                        memory_key_padding_mask=None, 
                        pos=query_embed, query_pos=pos_encodings
                    )
                    mask_features = self.rev_cross_attn_ffn_layers[i](mask_features)
            else:
                for layer in self.rev_cross_attn_layers:
                    mask_features = layer(
                        mask_features, output,
                        memory_mask=None,
                        memory_key_padding_mask=None, 
                        pos=query_embed, query_pos=pos_encodings
                    )
            mask_features = einops.rearrange(mask_features,"(H W) B C -> B C H W", H=H, W=W, B=B).contiguous()
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
    
    def process_results(self, batched_inputs, images, outputs, num_instances, batched_num_scrbs_per_mask=None):
        
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
        for mask_pred_result, input_per_image, image_size, inst_per_image, num_scrbs_per_mask in zip(
            mask_pred_results, batched_inputs, images.image_sizes, num_instances, batched_num_scrbs_per_mask
        ):
            
            processed_results.append({})

            padding_mask = torch.logical_not(input_per_image["padding_mask"]).to(mask_pred_result.device)
            mask_pred_result = mask_pred_result * padding_mask

            instance_r = retry_if_cuda_oom(self.interactive_instance_inference)(mask_pred_result, inst_per_image, num_scrbs_per_mask)
            processed_results[-1]["instances"] = instance_r

        return processed_results

    def interactive_instance_inference(self, mask_pred, num_instances, num_scrbs_per_mask=None):
        # mask_pred is already processed to have the same shape as original input
        
        import copy
        assert len(num_scrbs_per_mask) == num_instances
        image_size = mask_pred.shape[-2:]
        
        result = Instances(image_size)
        num_scrbs_per_mask_copy = copy.deepcopy(num_scrbs_per_mask)
        num_scrbs_per_mask_copy.append(mask_pred.shape[0]-sum(num_scrbs_per_mask))
        
        temp_out = []
        if num_scrbs_per_mask_copy[-1] == 0:
            splited_masks = torch.split(mask_pred, num_scrbs_per_mask_copy[:-1], dim=0)
        else:
            splited_masks = torch.split(mask_pred, num_scrbs_per_mask_copy, dim=0)
        for m in splited_masks:
            temp_out.append(torch.max(m, dim=0).values)
        
        mask_pred = torch.stack(temp_out)

        mask_pred = torch.argmax(mask_pred,0)
        m = []
        for i in range(num_instances):
            m.append((mask_pred == i).float())
        
        mask_pred = torch.stack(m)
        result.pred_masks = mask_pred
     
        return result