from copy import deepcopy
from torch import Tensor
from typing import Dict
from einops import rearrange, repeat
# from hodor.encoder.layer import EncoderLayer
from .layer import EncoderLayer
# from hodor.modelling.encoder.descriptor_initializer import AvgPoolingInitializer
from ..position_embeddings import SinosuidalPositionEmbeddings
from ...descriptor_initializer import AvgPoolingInitializer
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, n_heads, n_dims: int, n_layers: int, multi_scale: bool, pre_normalize: bool, pos_encodings: bool):
        super().__init__()

        self.query_initializer = AvgPoolingInitializer(multi_scale=multi_scale)

        attn_layer = EncoderLayer(
            n_dims, n_heads=n_heads, n_hidden_dims=1024, dropout=0.1, pre_normalize=pre_normalize
        )

        self.attn_layers = nn.ModuleList([deepcopy(attn_layer) for _ in range(n_layers)])

        self.pos_encoding_gen = SinosuidalPositionEmbeddings(n_dims // 2, normalize=True)
        self.use_pos_encodings = pos_encodings

        self.norm = nn.LayerNorm(n_dims) if pre_normalize else nn.Identity()

        self.register_parameter("query_embed", nn.Parameter(torch.zeros(n_dims), True))
        # self.register_parameter("fg_query_embed", nn.Parameter(torch.zeros(n_dims), True))
        # self.register_parameter("bg_query_embed", nn.Parameter(torch.zeros(n_bg_queries, n_dims), True))

        # self.num_bg_queries = n_bg_queries
        # self.mask_scale = 8
        self._reset_parameters()

    def _reset_parameters(self):
        # nn.init.normal_(self.fg_query_embed)
        # nn.init.normal_(self.bg_query_embed)
        nn.init.normal_(self.query_embed)


    def forward(self, fmaps: Tensor, scribbles: Tensor) -> Tensor:
        """
        Forward method
        :param fmaps: Tensor of shape [B, C, H, W] containing reference frame features.
        :param fg_mask: Tensor of shape [B, I, H, W] with values in range [0, 1]
        :param bg_mask: Tensor of shape [B, Qb, H, W] with values in range [0, 1]
        :return:
        """
        # assert fg_mask.dtype in (torch.float16, torch.float32)
        # assert not torch.any(fg_mask > (1. + 1e-6)) and not torch.any(fg_mask < -1e-6)
        # assert not torch.any(bg_mask > (1. + 1e-6)) and not torch.any(bg_mask < -1e-6)

        fmap = fmaps[-1]
        scribbles_resized = F.interpolate(scribbles, size=fmap.shape[-2:], mode="bilinear", align_corners=False)
        # bs, num_inst = fg_mask.shape[:2]
        # _, bs, _ = src[0].shape
        bs, num_scrbs = scribbles.shape[:2]

        with torch.no_grad():
            if self.use_pos_encodings:
                pos_encoding = self.pos_encoding_gen(fmap)
                pos_encoding = rearrange(pos_encoding, "B C H W -> B (H W) C")
            else:
                pos_encoding = None

            all_queries = self.query_initializer(fmaps, scribbles)

        fmap = rearrange(fmap, "B C H W -> B (H W) C")

        # all_queries = torch.cat((bg_queries, fg_queries), 1)  # [B, Qb+I, C]

        all_query_pos = repeat(self.query_embed, "C -> B Q C", B=bs, Q=num_scrbs)

        # bg_query_pos = repeat(self.bg_query_embed, "Qb C -> B Qb C", B=bs)
        # fg_query_pos = repeat(self.fg_query_embed, "C -> B I C", B=bs, I=num_inst)
        # all_query_pos = torch.cat((bg_query_pos, fg_query_pos), 1)  # [B, Qb+I, C]

        # bg_mask = rearrange(bg_mask, "B Qb H W -> B Qb (H W)")
        # fg_mask = rearrange(fg_mask, "B I H W -> B I (H W)")
        # all_masks = torch.cat((bg_mask, fg_mask), 1)  # [B, Qb+I, H*W]

        all_masks = rearrange(scribbles_resized, "B Q H W -> B Q (H W)")
        # print("all_queries:",all_queries.shape)
        # print("all_query_pos:",all_query_pos.shape)
        # print("fmap:",fmap.shape)
        # print("all_masks:",all_masks.shape)

        for layer in self.attn_layers:
            all_queries = layer(query=all_queries, kv=fmap, pos_key=pos_encoding, kv_mask=all_masks,
                                pos_query=all_query_pos)

        all_queries = self.norm(all_queries)

        return all_queries
       