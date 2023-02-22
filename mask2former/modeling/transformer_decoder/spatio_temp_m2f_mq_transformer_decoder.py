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
import detectron2.utils.comm as comm
from torch import nn, Tensor
from torch.nn import functional as F
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.config import configurable
from detectron2.layers import Conv2d
from einops import repeat
from .position_encoding import PositionEmbeddingSine
from detectron2.modeling.postprocessing import sem_seg_postprocess
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from .descriptor_initializer import AvgClicksPoolingInitializer, AvgPoolingInitializer
from mask2former.utils import get_new_scribbles_opt, preprocess_batch_data, get_new_points_mq,get_new_points_mq_per_obj,get_next_clicks_mq_per_object
from mask2former.utils.train_sampling_utils import get_next_clicks_mq
import cv2
from detectron2.utils.visualizer import Visualizer

def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))
color_map = get_palette(80)[1:]
class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

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
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

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
class SpatioTempM2FTransformerDecoderMQ(nn.Module):

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
        use_rev_cross_attn: bool,
        rev_cross_attn_num_layers: int,
        rev_cross_attn_scale: float,
        num_static_bg_queries: int,
        use_point_clicks: bool,
        per_obj_sampling: bool,
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
        self.use_point_clicks = use_point_clicks
        
        self.per_obj_sampling = per_obj_sampling
        self.use_time_coords = use_time_coords

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
        self.query_descriptors_initializer = AvgClicksPoolingInitializer(multi_scale=self.multi_scale)
        
        if self.concat_coord_mask_features:
            self.coordinates_prev_mask = nn.Sequential(
                nn.Conv2d(in_channels=hidden_dim+2, out_channels=256, kernel_size=1),
                # nn.BatchNorm2d(256),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
            )
        
        if self.random_bg_queries:
            self.register_parameter("bg_query", nn.Parameter(torch.zeros(hidden_dim), False))
        # self.bg_query = nn.Parameter(torch.zeros(hidden_dim), True)
        # self.bg_query = nn.Embedding(1,hidden_dim)
        # projection layer for generate positional queries
        self.queries_nonlinear_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.use_pos_coords = use_pos_coords
        if self.use_pos_coords:
            self.ca_qpos_sine_proj = nn.Linear(hidden_dim, hidden_dim)
        # self.descriptor_projection = nn.linear()
        # learnable query p.e.
        self.register_parameter("query_embed", nn.Parameter(torch.zeros(hidden_dim), True))
        # self.query_embed = nn.Parameter(torch.zeros(hidden_dim), True)
        # self.query_embed = nn.Embedding(1,hidden_dim)
        self.use_static_bg_queries = True
        if self.use_static_bg_queries:
            self.register_parameter("static_bg_pe", nn.Parameter(torch.zeros(self.num_static_bg_queries, hidden_dim), True))
            self.register_parameter("static_bg_query", nn.Parameter(torch.zeros(self.num_static_bg_queries,hidden_dim), True))

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

        # output FFNs
        # if self.mask_classification:
        #     self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        if not self.class_agnostic:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.query_embed)
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
        ret["rev_cross_attn_num_layers"] = cfg.REVERSE_CROSS_ATTN.NUM_LAYERS
        ret["rev_cross_attn_scale"] = cfg.REVERSE_CROSS_ATTN.SCALE_FACTOR

        # Iterative Pipeline
        ret["max_num_interactions"] = cfg.ITERATIVE.TRAIN.MAX_NUM_INTERACTIONS
        ret["accumulate_loss"] = cfg.ITERATIVE.TRAIN.ACCUMULATE_INTERACTION_LOSS
        ret["class_agnostic"] = cfg.ITERATIVE.TRAIN.CLASS_AGNOSTIC
        ret["concat_coord_mask_features"] =cfg.ITERATIVE.TRAIN.CONCAT_COORD_MASK_FEATURES
        ret["concat_coord_image_features"]=cfg.ITERATIVE.TRAIN.CONCAT_COORD_IMAGE_FEATURES
        ret["random_bg_queries"]=cfg.ITERATIVE.TRAIN.RANDOM_BG_QUERIES
        ret["use_pos_coords"] = cfg.ITERATIVE.TRAIN.USE_POS_COORDS
        ret["use_time_coords"] = cfg.ITERATIVE.TRAIN.USE_TIME_COORDS
        ret["num_static_bg_queries"] = cfg.ITERATIVE.TRAIN.NUM_STATIC_BG_QUERIES
        ret["use_point_clicks"] = cfg.ITERATIVE.TRAIN.USE_POINTS
        ret["per_obj_sampling"] = cfg.ITERATIVE.TRAIN.PER_OBJ_SAMPLING
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

    def forward(self, data, targets, images, num_instances, x, mask_features, mask = None,
                 prev_mask_logits=None, batched_num_scrbs_per_mask=None,
                 batched_fg_coords_list = None, batched_bg_coords_list = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        # is_train= True
        # accumulate_loss = False
        if self.training:
            if self.accumulate_loss:
                prev_output = None
                num_iters = random.randint(0,self.max_num_interactions)
                # num_iters=3
                # if comm.is_main_process():
                #     self.visualization(data, prev_output, batched_fg_coords_list[:], batched_bg_coords_list[:],
                #     alpha_blend=0.6, num_iter = 0)
                
                for i in range(num_iters):
                    prev_output = self.iterative_batch_forward(x, src, pos, size_list, images, mask_features, prev_mask_logits,
                                                               batched_num_scrbs_per_mask, batched_fg_coords_list, 
                                                               batched_bg_coords_list
                    )
                    prev_mask_logits = prev_output['pred_masks']
                    # print("before:",batched_num_scrbs_per_mask)
                    processed_results = self.process_results(data, images, prev_output, num_instances, batched_num_scrbs_per_mask)
                    if self.use_point_clicks:
                        # scribbles, batched_num_scrbs_per_mask = get_new_points_mq_per_obj(data, processed_results, scribbles,
                        #                                                       random_bg_queries=self.random_bg_queries,
                        #                                                       batched_num_scrbs_per_mask = batched_num_scrbs_per_mask)
                        batched_num_scrbs_per_mask, batched_fg_coords_list, batched_bg_coords_list = get_next_clicks_mq(data, processed_results,i+1, src[0].device,
                                                                            batched_num_scrbs_per_mask,batched_fg_coords_list, batched_bg_coords_list,
                                                                            per_obj_sampling=self.per_obj_sampling)
                        # if comm.is_main_process():
                        #     self.visualization(data, processed_results, batched_fg_coords_list[:], batched_bg_coords_list[:],
                        #     alpha_blend=0.6, num_iter = i+1)
                    else:
                        scribbles = get_new_scribbles_opt(data, processed_results, scribbles,random_bg_queries=self.random_bg_queries)
                    # print("after:",batched_num_scrbs_per_mask)
            outputs = self.iterative_batch_forward(x, src, pos, size_list, images, mask_features, prev_mask_logits,
                                                    batched_num_scrbs_per_mask, batched_fg_coords_list, 
                                                    batched_bg_coords_list
                    )
        else:
            outputs = self.iterative_batch_forward(x, src, pos, size_list, mask_features, scribbles,prev_mask_logits,
                                                   batched_num_scrbs_per_mask, batched_fg_coords_list, batched_bg_coords_list)
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

    def iterative_batch_forward(self, x, src, pos, size_list, images, mask_features, prev_mask_logits=None,
                                batched_num_scrbs_per_mask=None, batched_fg_coords_list=None,
                                batched_bg_coords_list=None):

        _, bs, _ = src[0].shape
        B, C, H, W = mask_features.shape
        height = 4*H
        width = 4*W
        if self.use_static_bg_queries:
            # if batched_num_scrbs_per_mask is not None:
            #     new_scribbles = []
            #     for scrbs in scribbles:
            #         if scrbs[-1] is not None:
            #             new_scribbles.append(torch.cat(scrbs))
            #         else:
            #             new_scribbles.append(torch.cat(scrbs[:-1]))
            # max_scrbs_batch = max([scrbs.shape[0] for scrbs in new_scribbles])

            # _,height,width = new_scribbles[0].shape
            descriptors = self.query_descriptors_initializer(x, batched_fg_coords_list, batched_bg_coords_list, height=height, 
                                                            width=width, random_bg_queries=self.random_bg_queries)
            max_queries_batch = max([desc.shape[1] for desc in descriptors])
            for i, desc in enumerate(descriptors):
                bg_queries = repeat(self.bg_query, "C -> 1 L C", L=max_queries_batch-desc.shape[1])
                # bg_queries = repeat(self.bg_query, "C -> 1 L C", L=self.num_static_bg_queries)
                descriptors[i] = torch.cat((descriptors[i], bg_queries), dim=1)
            output = torch.cat(descriptors, dim=0)
            
            query_embed = repeat(self.query_embed, "C -> Q N C", N=bs, Q=output.shape[1])
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
        if self.use_pos_coords:
            scrbs_coords = self.get_pos_tensor_coords(batched_fg_coords_list, batched_bg_coords_list,
                                                    num_scrbs, height, width, output.device
                            ) # bsxQx3
            pos_coord_embed = self.gen_sineembed_for_position(scrbs_coords.permute(1,0,2), use_timestamp=self.use_time_coords) # Q x bs x C
            pos_coord_embed = self.ca_qpos_sine_proj(pos_coord_embed.to(query_embed.dtype))
            
            query_embed = query_embed + pos_coord_embed

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
    
    def gen_sineembed_for_position(self, pos_tensor, use_timestamp = False):
        # n_query, bs, 3 = pos_tensor.size()
        # sineembed_tensor = torch.zeros(n_query, bs, 256)
        import math
        scale = 2 * math.pi
        dim_t = torch.arange(128, dtype=torch.float, device=pos_tensor.device)
        dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / 128)
        x_embed = pos_tensor[:, :, 1] * scale
        y_embed = pos_tensor[:, :, 0] * scale
        if use_timestamp:
            t_embed = pos_tensor[:, :, 2] * scale
            y_embed += t_embed
            x_embed += x_embed
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        return pos
    
    def get_pos_tensor_coords(self, batched_fg_coords_list, batched_bg_coords_list, num_queries, height, width, device):

        #batched_fg_coords_list: batch x (list of list of fg coords) [y,x,t]

        # return
        # points: Bs x num_queries x 3 
        B = len(batched_fg_coords_list)
        
        pos_tensor = []
        
        for i, fg_coords_per_image in enumerate(batched_fg_coords_list):
            coords_per_image  = []
            for fg_coords_per_mask in fg_coords_per_image:
                for coords in fg_coords_per_mask:
                    coords_per_image.append([coords[0]/width, coords[1]/height, coords[2]])
            if batched_bg_coords_list[i] is not None:
                for coords in batched_bg_coords_list[i]:
                    coords_per_image.append([coords[0]/width, coords[1]/height, coords[2]])
            coords_per_image.extend([[0,0,0]] * (num_queries-len(coords_per_image)))
            pos_tensor.append(torch.tensor(coords_per_image,device=device))
        # pos_tensor = torch.tensor(pos_tensor,device=device)
        pos_tensor = torch.stack(pos_tensor)
        return pos_tensor
    
    @torch.no_grad()
    def create_spatial_grid(height, width, dtype=torch.float32, device="cpu"):
        # returns [tx, ty, txy, y, x]
        x_abs = max(1., width / float(height))
        y_abs = max(1., height / float(width))

        # torch.linspace does not work with float16, so create the tensors using float32 and then cast to appropriate dtype
        x = torch.linspace(-x_abs, x_abs, width, dtype=torch.float32, device=device).to(dtype=dtype)
        y = torch.linspace(-y_abs, y_abs, height, dtype=torch.float32, device=device).to(dtype=dtype)

        y, x = torch.meshgrid(y, x)
        grid = torch.stack((y, x), dim=0).to(device=device)
        return y, x, grid
    
    def process_results(self, batched_inputs, images, outputs, num_instances, batched_num_scrbs_per_mask=None):
        if outputs["pred_logits"] is None:
            mask_cls_results = [None]*(outputs["pred_masks"].shape[0])
        else:
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
        if batched_num_scrbs_per_mask is None:
            batched_num_scrbs_per_mask = [[1]*inst_per_image for inst_per_image in num_instances]
            # print("here")
            # batched_num_scrbs_per_mask

        processed_results = []
        for mask_cls_result, mask_pred_result, input_per_image, image_size, inst_per_image, num_scrbs_per_mask in zip(
            mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes, num_instances, batched_num_scrbs_per_mask
        ):
            # height = input_per_image.get("height", image_size[0])
            # width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            padding_mask = torch.logical_not(input_per_image["padding_mask"]).to(mask_pred_result.device)
            mask_pred_result = mask_pred_result * padding_mask

            instance_r = retry_if_cuda_oom(self.interactive_instance_inference)(mask_cls_result, mask_pred_result,
                                                                                inst_per_image, num_scrbs_per_mask)
            processed_results[-1]["instances"] = instance_r

        return processed_results

    def interactive_instance_inference(self, mask_cls, mask_pred, num_instances, num_scrbs_per_mask=None):
        # mask_pred is already processed to have the same shape as original input
        # print("interactive instance inference")
        import copy
        # print(num_instances)
        # print(num_scrbs_per_mask)
        assert len(num_scrbs_per_mask) == num_instances
        image_size = mask_pred.shape[-2:]
        
        result = Instances(image_size)
        num_scrbs_per_mask_copy = copy.deepcopy(num_scrbs_per_mask)
        num_scrbs_per_mask_copy.append(mask_pred.shape[0]-sum(num_scrbs_per_mask))
        # print(num_scrbs_per_mask_copy)
        # print(mask_pred.shape)
        temp_out = []
        splited_masks = torch.split(mask_pred, num_scrbs_per_mask_copy, dim=0)
        for m in splited_masks:
            temp_out.append(torch.max(m, dim=0).values)
        # mask_pred = torch.cat([torch.stack(temp_out),splited_masks[-1]])
        mask_pred = torch.stack(temp_out)

        # mask (before sigmoid)
        mask_pred = mask_pred[:num_instances]
        # print("after:",mask_pred.shape)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        if mask_cls is None:
            result.scores = torch.zeros((num_instances))
            result.pred_classes = torch.zeros((num_instances))
        else:
            # [Q, K+1] -> [fg, K]
            temp_out = []
            splited_scores = torch.split(mask_cls, num_scrbs_per_mask_copy, dim=0)
            for m in splited_scores:
                temp_out.append(torch.max(m, dim=0).values)
            # mask_cls = torch.cat([torch.stack(temp_out),splited_scores[-1]])
            mask_cls = torch.stack(temp_out)

            scores = F.softmax(mask_cls, dim=-1)[:num_instances, :-1]
            labels_per_image = scores.argmax(dim=1)

            scores_per_image = scores.max(dim=1)[0]
            mask_pred = mask_pred[:num_instances]
            # calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
            result.scores = scores_per_image * mask_scores_per_image
            result.pred_classes = labels_per_image
        return result

    def visualization(self, batched_inputs, prev_output, batched_fg_coords_list,batched_bg_coords_list,
                  alpha_blend=0.6, num_iter = 0):
        image = np.asarray(batched_inputs[0]['image'].detach().permute(1,2,0))
        import copy
        image = copy.deepcopy(image)
        visualizer = Visualizer(image, metadata=None)
        if prev_output is not None:
            import torchvision.transforms.functional as F
            pred_masks = F.resize(prev_output[0]['instances'].pred_masks.detach().to(device= 'cpu',dtype=torch.uint8), image.shape[:2])
        else:
            pred_masks = batched_inputs[0]['instances'].gt_masks.detach()
        c = []
        for i in range(pred_masks.shape[0]):
            # c.append(color_map[2*(i)+2]/255.0)
            c.append(color_map[i]/255.0)
        # pred_masks = np.asarray(pred_masks).astype(np.bool_)
        vis = visualizer.overlay_instances(masks = pred_masks, assigned_colors=c, alpha=alpha_blend)
        # [Optional] prepare labels

        image = vis.get_image()
        # # Laminate your image!
        total_colors = len(color_map)-1
        
        h,w = image.shape[:2]
        for j, fg_coords_per_mask in enumerate(batched_fg_coords_list[0]):
            for i, coords in enumerate(fg_coords_per_mask):
                color = np.array(color_map[total_colors-5*j-4], dtype=np.uint8)
                color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
                if i==0:
                    image = cv2.circle(image, (int(coords[1]), int(coords[0])), 8, tuple(color), -1)
                else:
                    image = cv2.circle(image, (int(coords[1]), int(coords[0])), 3, tuple(color), -1)
        
        if batched_bg_coords_list[0]:
            for i, coords in enumerate(batched_bg_coords_list[0]):
                color = np.array([255,0,0], dtype=np.uint8)
                color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
                image = cv2.circle(image, (int(coords[1]), int(coords[0])), 3, tuple(color), -1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = cv2.resize(image, (inputs["width"],inputs["height"]))
        save_dir = os.path.join("./train_vis/", str(batched_inputs[0]['image_id']))
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f"iter_{num_iter}.jpg"), image)