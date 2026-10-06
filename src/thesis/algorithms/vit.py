"""Vision Transformer backbone: conv patch-embed + spatiotemporal blocks with 3D RoPE.

A reusable core: `ViTEncoder` (encoders.py) wraps it as an observation encoder now, and a
future `ViTPredictor` backbone can reuse it by adding a time embedding and per-stream heads.
By default attention is full (non-causal) over all `T*Hp*Wp` tokens, reusing the trunk's
`SelfAttention` and `RopeND` on its three grid axes so positions carry frame index, patch
row, and patch column. With `attend_across_time=False`, frames are folded into the batch
dim instead: attention stays within each frame's own `Hp*Wp` patches (2D RoPE over row/col
only, no frame axis), matching a conv backbone's per-frame independence while still needing
explicit spatial position ids.
"""

import torch
import torch.nn as nn

from .attention import SelfAttention
from .layers import make_mlp
from .rope import RopeND, axes_from_positions, make_pos_ids

__all__ = ["ViT"]


class ViTBlock(nn.Module):
    """Pre-norm self-attention then MLP, both residual; full attention with 3D RoPE."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = make_mlp(dim, int(dim * mlp_ratio), dim)

    def forward(self, x, freqs):
        x = x + self.attn(self.norm1(x), freqs=freqs)
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """(B, T, C, H, W) float frames -> (B, T*Hp*Wp, dim) patch tokens, time-major.

    Each frame is patched by a strided conv into an (Hp, Wp) grid, and 3D RoPE attaches
    (frame index, patch row, patch col) to every token. The grid (Hp, Wp) =
    (H // patch_size, W // patch_size) is what a consuming spec must declare as its `grid`.

    `img_size` is always required to size the patch grid. `attend_across_time` (default
    True) picks the attention scope:

      True   (default) attention spans all `T*Hp*Wp` tokens; `frames` is also required, to
             fix the RoPE ladder's frame-axis range at construction. One encoder shared
             across streams of different window lengths rotates them all on the same
             frequencies, and a call may be shorter than `frames` and reuse the ladder over
             fewer positions.
      False  frames fold into the batch dim and attention stays within each frame's own
             `Hp*Wp` patches; the RoPE ladder carries row/col only (no frame axis, no
             `frames` needed), so a `rope:` spec for this mode must not declare `time`.
    """

    def __init__(self, in_channels, dim, patch_size, depth, num_heads, mlp_ratio=4.0,
                 rope=None, img_size=None, frames=None, attend_across_time=True):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.attend_across_time = attend_across_time
        assert rope, (
            "a grid-pooled ViT needs a `rope:` block naming its axes, e.g. "
            "{time: {share: 0.5, period: auto}, height: {share: 0.25, period: auto}, "
            "width: {share: 0.25, period: auto}}"
        )
        assert img_size, "a grid-pooled ViT needs `img_size` to size its patch grid"
        # a pair is (H, W): a stitched stream's frame is not square
        h, w = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.grid = (int(h) // int(patch_size), int(w) // int(patch_size))
        if attend_across_time:
            assert frames, (
                "a grid-pooled ViT with attend_across_time=True needs `frames` to size its "
                f"RoPE ladder at build time (got frames={frames})"
            )
            self.frames = int(frames)
            # no `seq` axis: every (frame, row, col) triple is already unique
            pos = make_pos_ids(torch.arange(self.frames), self.grid)
        else:
            assert "time" not in rope, (
                "attend_across_time=False folds frames into the batch dim, so no token ever "
                "sees another frame -- drop the `time` axis from `rope`"
            )
            self.frames = None
            pos = make_pos_ids(torch.zeros(1), self.grid)
        self.rope = RopeND(dim // num_heads, axes_from_positions(pos, rope))
        self.blocks = nn.ModuleList(
            [ViTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, frames):
        B, T = frames.shape[:2]
        patches = self.patch_embed(frames.flatten(0, 1))
        hp, wp = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)                # (B*T, Hp*Wp, dim)

        if self.attend_across_time:
            assert (hp, wp) == self.grid and T <= self.frames, (
                f"frames are {T}x{hp}x{wp} but the RoPE ladder was built for at most "
                f"{self.frames}x{self.grid[0]}x{self.grid[1]}; raise `img_size`/`frames` on "
                "the encoder spec rather than letting the positions rescale"
            )
            tokens = tokens.reshape(B, T, hp * wp, -1).flatten(1, 2)
            pos = make_pos_ids(torch.arange(T, device=frames.device), (hp, wp))
            freqs = self.rope(pos)
            for block in self.blocks:
                tokens = block(tokens, freqs)
            return self.norm(tokens)

        assert (hp, wp) == self.grid, (
            f"frames are {hp}x{wp} but the RoPE ladder was built for {self.grid[0]}x"
            f"{self.grid[1]}; raise `img_size` on the encoder spec rather than letting the "
            "positions rescale"
        )
        pos = make_pos_ids(torch.zeros(1, device=frames.device), (hp, wp))
        freqs = self.rope(pos)
        for block in self.blocks:
            tokens = block(tokens, freqs)                          # (B*T, Hp*Wp, dim)
        return self.norm(tokens).reshape(B, T * hp * wp, -1)
