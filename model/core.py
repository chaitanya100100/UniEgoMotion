import torch
from torch import nn
import torch.nn.functional as F
import ipdb

from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# class Attention(nn.Module):
#     def __init__(self, dim, heads, dropout):
#         super().__init__()
#         assert dim % heads == 0
#         self.heads = heads
#         self.dim = dim
#         self.dim_head = dim // heads
#         self.scale = self.dim_head**-0.5
#         self.norm = nn.LayerNorm(dim)

#         self.softmax = nn.Softmax(dim=-1)

#         self.q = nn.Linear(dim, dim, bias=False)
#         self.k = nn.Linear(dim, dim, bias=False)
#         self.v = nn.Linear(dim, dim, bias=False)
#         self.to_out = nn.Linear(dim, dim, bias=False)
#         self.attn_dropout = nn.Dropout(dropout)
#         self.out_dropout = nn.Dropout(dropout)

#     def forward(self, x, kv_x, mask=None):
#         # mask B x 1 x 1 x N denotes which elements of kv_x are valid

#         x = self.norm(x)

#         qkv = (self.q(x), self.k(kv_x), self.v(kv_x))

#         q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         if mask is not None:
#             dots.masked_fill_(mask == 0, -float("inf"))

#         attn = self.softmax(dots)
#         attn = self.attn_dropout(attn)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, "b h n d -> b n (h d)")
#         out = self.to_out(out)
#         out = self.out_dropout(out)
#         return out


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, kv_x, mask=None):
        # mask B x 1 x 1 x N denotes which elements of kv_x are valid

        x = self.norm(x)
        if mask is not None:
            mask = mask.squeeze([1, 2]).bool().logical_not()
        out = self.attn(x, kv_x, kv_x, key_padding_mask=mask)[0]
        out = self.out_dropout(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, dropout, ff_mult):
        super().__init__()
        self.attn = Attention(dim, heads, dropout)
        self.ff = FeedForward(dim, dim * ff_mult, dropout)

    def forward(self, x, mask=None):
        x = self.attn(x, kv_x=x, mask=mask) + x
        x = self.ff(x) + x
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, dropout, ff_mult):
        super().__init__()
        self.attn1 = Attention(dim, heads, dropout)
        self.attn2 = Attention(dim, heads, dropout)
        self.ff = FeedForward(dim, dim * ff_mult, dropout)

    def forward(self, x, context, mask=None, context_mask=None):
        x = self.attn1(x, kv_x=x, mask=mask) + x
        x = self.attn2(x, kv_x=context, mask=context_mask) + x
        x = self.ff(x) + x
        return x


if __name__ == "__main__":
    dim = 768
    heads = 24
    num_enc_layers = 6
    num_dec_layers = 4
    b = 3

    ipdb.set_trace()
