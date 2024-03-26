import torch
from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, d_model, do_layernorm=True, batch_first=True, full_dropout=0, **kwargs):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            d_model, batch_first=batch_first, **kwargs)
        self.layernorm = nn.LayerNorm(
            d_model) if do_layernorm else nn.Identity()
        self.full_dropout = full_dropout

    def forward(self, input, memory):
        if torch.rand(1) < self.full_dropout and self.training:
            return input

        x, _ = self.attention(input, memory, memory)

        x = input + x

        x = self.layernorm(x)

        return x
