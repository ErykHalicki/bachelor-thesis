"""Latent -> RGB decoder, a diagnostic for what a representation keeps.

A set of learnable query tokens, one per patch of the target image, reads a latent through
several cross-attention + residual-MLP layers and is projected to pixel patches. The latent
enters only as keys/values, so the same decoder reads a full patch grid (196 V-JEPA tokens)
and a single pooled summary vector without changing shape anywhere: whatever the queries
cannot pull out of those keys is information the representation does not carry.

Nothing here is a generative model. The queries are the only spatial prior, so a decode is
the conditional mean of every image consistent with the latent -- a blurry image is a latent
that pinned down the scene loosely, not a decoder that failed to sample.
"""

import torch
import torch.nn as nn

from .attention import CrossAttention
from .layers import make_mlp

__all__ = ["PixelDecoder", "PixelDecoderBlock"]


class PixelDecoderBlock(nn.Module):
    """Pre-norm cross-attention into the latent, then a residual MLP."""

    def __init__(self, dim, num_heads, source_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.attn = CrossAttention(dim, num_heads, [source_dim])
        self.norm_mlp = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = make_mlp(dim, int(dim * mlp_ratio), dim)

    def forward(self, queries, source):
        queries = queries + self.attn(self.norm_q(queries), [source])
        return queries + self.mlp(self.norm_mlp(queries))


class PixelDecoder(nn.Module):
    """(M, N, in_dim) latent tokens -> (M, 3, H, W) images in [0, 1] units.

    `img_size` is an int for a square image or [H, W]; both must divide by `patch_size`,
    giving (H/patch)*(W/patch) query tokens. The latent's token count N is free -- it is
    the key/value length of every cross-attention layer, and each layer projects it into
    the decoder width itself, so 1 and 196 tokens cost the same query set.

    Output is unbounded (a plain linear projection of the last query states), matching the
    MSE it is trained under; clamp before viewing.
    """

    def __init__(self, in_dim, img_size=224, patch_size=16, hidden_dim=512, depth=4,
                 num_heads=8, mlp_ratio=4.0):
        super().__init__()
        size = ((int(img_size), int(img_size)) if isinstance(img_size, (int, float))
                else tuple(int(v) for v in img_size))
        patch_size = int(patch_size)
        if any(s % patch_size for s in size):
            raise ValueError(f"img_size {size} does not divide into {patch_size}px patches")
        self.img_size = size
        self.patch_size = patch_size
        self.grid = (size[0] // patch_size, size[1] // patch_size)

        self.norm_src = nn.LayerNorm(int(in_dim), eps=1e-6)
        self.query = nn.Parameter(torch.randn(1, self.grid[0] * self.grid[1], hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([
            PixelDecoderBlock(hidden_dim, int(num_heads), int(in_dim), float(mlp_ratio))
            for _ in range(int(depth))
        ])
        self.norm_out = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.head = nn.Linear(hidden_dim, patch_size * patch_size * 3)

    def forward(self, x):
        source = self.norm_src(x)
        queries = self.query.expand(x.shape[0], -1, -1).to(x.dtype)
        for block in self.blocks:
            queries = block(queries, source)
        patches = self.head(self.norm_out(queries))

        gh, gw = self.grid
        p = self.patch_size
        img = patches.reshape(-1, gh, gw, p, p, 3).permute(0, 5, 1, 3, 2, 4)
        return img.reshape(-1, 3, gh * p, gw * p)
