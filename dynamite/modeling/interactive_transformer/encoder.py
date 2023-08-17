from torch import nn, Tensor
from .utils import SelfAttentionLayer, CrossAttentionLayer, FFNLayer

class Encoder(nn.Module):

    def __init__(self, hidden_dim, dim_feedforward, nheads, enc_layers, pre_norm):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.pre_norm = pre_norm
        self.num_heads = nheads
        self.num_layers = enc_layers
        self.self_attention_layers = nn.ModuleList()
        self.cross_attention_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=self.hidden_dim,
                    nhead=self.num_heads,
                    dropout=0.0,
                    normalize_before=self.pre_norm,
                )
            )

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
                    dim_feedforward=self.dim_feedforward,
                    dropout=0.0,
                    normalize_before=self.pre_norm,
                )
            )