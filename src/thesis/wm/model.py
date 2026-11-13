"""The world model: views -> one latent each, and a predictor of the next step's latents.

Training follows LeWorldModel (Maes et al., 2026). Every frame of every view is encoded
to a single latent by an unfrozen encoder and a BatchNorm projector. A causal transformer
reads, per step, each view's latent and one action token, and predicts each view's latent
at the next step. The loss is the prediction error plus SIGReg on the latents; there is no
stop-gradient and no EMA target. The one departure is the prediction head, which has no
BatchNorm (see WorldModel.__init__).

Samples have different numbers of views, so the predictor pads each step to the largest
view count in the batch and masks the empty slots as attention keys. Sequences here are
tens of tokens, where padding costs less than splitting the batch (see the timings in
docs/multi_embodiment_data.md).
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from thesis.algorithms.encoders import LeVJEPAEncoder

from .camera_embedding import CameraEmbedding
from .embodiment import ActionAdapters


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer (LeJEPA): the Epps-Pulley statistic between
    the batch's 1-D projections and a standard normal, averaged over random directions."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z):
        """z: (n, D) float32 rows."""
        directions = torch.randn(z.shape[1], self.num_proj, device=z.device)
        directions = directions / directions.norm(dim=0)
        x_t = (z @ directions).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        return ((err @ self.weights) * z.shape[0]).mean()


class BatchNormMLP(nn.Module):
    """LeWM's projector: Linear, BatchNorm, GELU, Linear, over the last dimension."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                 nn.GELU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x.flatten(0, -2)).view(*x.shape[:-1], -1)


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.out(attn.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.norm2(x))


class Predictor(nn.Module):
    """Per step: one token per view slot, then the action token. A token attends to every
    real token of its own step and of earlier steps. The output at a view's token is the
    prediction of that view at the next step."""

    def __init__(self, latent_dim, dim, depth, heads, max_steps):
        super().__init__()
        self.embed = nn.Linear(latent_dim, dim)
        self.time = nn.Embedding(max_steps, dim)
        self.action_type = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.time.weight, std=0.02)

    def forward(self, z, actions, cam, view_sample, view_slot):
        """z (N, S, latent), actions (B, S, dim), cam (N, dim) -> (N, S, dim)."""
        B, S, dim = actions.shape
        V = int(view_slot.max()) + 1
        x = actions.new_zeros(B, S, V + 1, dim)
        x[view_sample, :, view_slot] = (self.embed(z) + cam[:, None]).to(x.dtype)
        x[:, :, V] = actions + self.action_type
        x = (x + self.time.weight[:S, None]).flatten(1, 2)

        real = torch.zeros(B, V + 1, dtype=torch.bool, device=x.device)
        real[view_sample, view_slot] = True
        real[:, V] = True
        step = torch.arange(S, device=x.device).repeat_interleave(V + 1)
        mask = (step[None, :] <= step[:, None]) & real.repeat(1, S)[:, None, :]

        for block in self.blocks:
            x = block(x, mask[:, None])
        return self.norm(x).unflatten(1, (S, V + 1))[view_sample, :, view_slot]


def _checkpointed(forward, x, **kwargs):
    if not torch.is_grad_enabled():
        return forward(x, **kwargs)
    return checkpoint(functools.partial(forward, **kwargs), x, use_reentrant=False)


def build_encoder(cfg):
    """An unfrozen LeVJEPA ViT giving one [cls] per frame.

    `train_blocks` limits training to the last n transformer blocks (null trains all of
    it, 0 freezes it). Everything below them stays frozen, [cls] token included, so those blocks build
    no graph: one trainable parameter at the input would make every block store its
    activations.
    `grad_checkpoint` recomputes each block's activations in the backward pass instead of
    storing them.

    With `repeat: 1` a clip is one frame, so the block-causal mask the weights were trained
    under hides nothing. It is dropped then, which lets attention take the unmasked path.
    """
    encoder = LeVJEPAEncoder({
        "arch": cfg.arch, "checkpoint": cfg.checkpoint, "crop_size": cfg.img_size,
        "pooling": "cls", "per_frame": True, "repeat": cfg.repeat, "frozen": False,
        "attn_mode": "full" if cfg.repeat == 1 else "block_causal",
    })
    backbone = encoder.backbone
    if cfg.train_blocks is not None:
        backbone.requires_grad_(False)
        for block in backbone.blocks[len(backbone.blocks) - cfg.train_blocks:]:
            block.requires_grad_(True)
        if backbone.norm is not None and cfg.train_blocks > 0:
            backbone.norm.requires_grad_(True)
    if cfg.grad_checkpoint:
        for block in backbone.blocks:
            if any(p.requires_grad for p in block.parameters()):
                block.forward = functools.partial(_checkpointed, block.forward)
    return encoder


class WorldModel(nn.Module):
    def __init__(self, spec, cfg):
        super().__init__()
        self.spec = spec
        self.encoder = build_encoder(cfg.encoder)
        self.projector = BatchNormMLP(self.encoder.backbone.embed_dim, cfg.proj_hidden, cfg.latent_dim)
        self.actions = ActionAdapters(spec, cfg.predictor.dim)
        self.cameras = CameraEmbedding(spec, cfg.predictor.dim, cfg.camera_embedding)
        self.predictor = Predictor(cfg.latent_dim, cfg.predictor.dim, cfg.predictor.depth,
                                   cfg.predictor.heads, cfg.max_steps)
        # No BatchNorm here, unlike LeWM's pred_proj. With an unfrozen encoder the head came
        # to depend on batch statistics: one-step error on held-out windows was 0.05 with
        # them and 0.28 with running statistics, worse than copying the last latent (0.09).
        self.pred_proj = nn.Sequential(
            nn.Linear(cfg.predictor.dim, cfg.proj_hidden), nn.GELU(),
            nn.Linear(cfg.proj_hidden, cfg.latent_dim))
        self.sigreg = SIGReg(num_proj=cfg.sigreg.num_proj)
        self.sigreg_weight = cfg.sigreg.weight

    def encode(self, views):
        """uint8 (N, T, 3, H, W) -> (N, T, latent)."""
        return self.projector(self.encoder(views))

    def predict(self, z, action_tokens, batch):
        """Latents and action tokens of steps 0..S-1 -> predicted latents of steps 1..S."""
        cam = self.cameras(batch.view_emb, batch.view_cam)
        hidden = self.predictor(z, action_tokens, cam, batch.view_sample, batch.view_slot)
        return self.pred_proj(hidden)

    def forward(self, batch):
        z = self.encode(batch.views)
        pred = self.predict(z[:, :-1], self.actions(batch), batch)
        z32, pred32 = z.float(), pred.float()
        pred_loss = F.mse_loss(pred32, z32[:, 1:])
        sigreg = self.sigreg(z32.flatten(0, 1))
        return {
            "loss": pred_loss + self.sigreg_weight * sigreg,
            "pred_loss": pred_loss,
            "sigreg": sigreg,
            "z": z32,
            "pred": pred32,
        }

    @torch.no_grad()
    def rollout(self, z, batch, context):
        """Predict steps context..T-1 from the first `context` true latents and all actions,
        feeding predictions back in. Returns (N, T - context, latent)."""
        action_tokens = self.actions(batch)
        seq = z[:, :context]
        for t in range(context, z.shape[1]):
            nxt = self.predict(seq, action_tokens[:, :t], batch)[:, -1:]
            seq = torch.cat([seq, nxt.to(seq.dtype)], dim=1)
        return seq[:, context:]
