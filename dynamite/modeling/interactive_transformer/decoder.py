import torch
import einops
from torch import nn, Tensor
from .utils import SelfAttentionLayer, CrossAttentionLayer, FFNLayer
from .position_encoding import PositionEmbeddingSine

class Decoder(nn.Module):

    def __init__(self, hidden_dim, nheads, dec_layers, pre_norm):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.pre_norm = pre_norm
        self.num_heads = nheads
        self.num_layers = dec_layers

        self.cross_attention_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=self.hidden_dim,
                    nhead=self.num_heads,
                    dropout=0.0,
                    normalize_before=self.pre_norm,
                )
            )
      
            self.ffn_layers.append(
                FFNLayer(
                    d_model=self.hidden_dim,
                    dim_feedforward=self.hidden_dim*2,
                    dropout=0.0,
                    normalize_before=self.pre_norm,
                )
            )
        # positional encoding for mask features
        N_steps = self.hidden_dim // 2
        self.pe_mask_features = PositionEmbeddingSine(N_steps, normalize=True)
    
    def forward(self, x):
        mask_features, output, query_embed = x
        with torch.no_grad():
            pos_encodings = self.pe_mask_features(mask_features)
            pos_encodings = einops.rearrange(pos_encodings,"B C H W -> (H W) B C")

        B, C, H, W = mask_features.shape
        mask_features = einops.rearrange(mask_features,"B C H W -> (H W) B C")

        #output is QxNxC
        # if self.use_mlp_rev_attn:
        for i in range(self.num_layers):
            mask_features = self.cross_attention_layers[i](
                mask_features, output,
                memory_mask=None,
                memory_key_padding_mask=None, 
                pos=query_embed, query_pos=pos_encodings
            )
            mask_features = self.ffn_layers[i](mask_features)
        return mask_features
