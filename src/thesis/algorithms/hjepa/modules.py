"""Building blocks of H-JEPA (Zhang et al. 2026, external/h-jepa), ported module for module.

Parameter names match the reference exactly, so a reference checkpoint's state dict loads
into these modules unchanged (see `HJEPAWorldModel.load_reference_checkpoint`) and the two
produce the same numbers on the same inputs (tests/test_hjepa_parity.py).
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def init_module_weights(m, std=0.02):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class SIGReg(nn.Module):
    """LeWM's SIGReg exactly as H-JEPA calls it: proj (T, B, D), one statistic per time step
    scaled by the batch size, averaged over T. ECF means are averaged across DDP ranks."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        cos, sin = x_t.cos().mean(-3), x_t.sin().mean(-3)
        n = proj.size(-2)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            for ecf in (cos, sin):
                torch.distributed.all_reduce(ecf)
                ecf.data.div_(world_size)
            n = n * world_size
        err = (cos - self.phi).square() + sin.square()
        statistic = (err @ self.weights) * n
        return statistic.mean()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, norm_fn=nn.LayerNorm,
                 act_fn=nn.GELU, final_ln=False):
        super().__init__()
        out_dim = output_dim or input_dim
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), norm, act_fn(), nn.Linear(hidden_dim, out_dim),
        )
        self.final_norm = nn.LayerNorm(out_dim) if final_ln else nn.Identity()

    def forward(self, x):
        return self.final_norm(self.net(x))


class ResidualLatentMLP(nn.Module):
    """The `latent_mlp` upper-level state encoder (no residual despite the name)."""

    def __init__(self, input_dim, hidden_dim=384, output_dim=None, act_fn=nn.GELU,
                 final_ln=True, trunc_normal_init=False):
        super().__init__()
        out_dim = output_dim or input_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = act_fn()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.final_norm = nn.LayerNorm(out_dim) if final_ln else nn.Identity()
        if trunc_normal_init:
            self.apply(init_module_weights)

    def forward(self, x):
        return self.final_norm(self.fc2(self.act(self.fc1(self.input_norm(x)))))


def build_projector(cfg, *, input_dim, output_dim):
    kind = str((cfg or {}).get("type", "mlp")).lower()
    if kind == "identity":
        if int(input_dim) != int(output_dim):
            raise ValueError(
                f"identity projector cannot change width: {input_dim} -> {output_dim}"
            )
        return nn.Identity()
    if kind == "layernorm":
        return nn.LayerNorm(output_dim)
    if kind == "mlp":
        return MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=2048,
                   norm_fn=nn.BatchNorm1d)
    raise ValueError(f"unsupported projector type '{kind}'")


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
                       if project_out else nn.Identity())

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads)
                   for t in self.to_qkv(x).chunk(3, dim=-1))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        return self.to_out(rearrange(out, "b h t d -> b t (h d)"))


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning on the action embedding."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        return x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, heads, dim_head, mlp_dim,
                 dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.input_proj = (nn.Linear(input_dim, hidden_dim)
                           if input_dim != hidden_dim else nn.Identity())
        self.cond_proj = (nn.Linear(input_dim, hidden_dim)
                          if input_dim != hidden_dim else nn.Identity())
        self.output_proj = (nn.Linear(hidden_dim, output_dim)
                            if hidden_dim != output_dim else nn.Identity())

    def forward(self, x, c):
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        return self.output_proj(self.norm(x))


class ARPredictor(nn.Module):
    """Causal next-embedding predictor over a `num_frames` history."""

    def __init__(self, *, num_frames, depth, heads, mlp_dim, input_dim, hidden_dim,
                 output_dim=None, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(input_dim, hidden_dim, output_dim or input_dim, depth,
                                       heads, dim_head, mlp_dim, dropout)

    def forward(self, x, c):
        x = self.dropout(x + self.pos_embedding[:, : x.size(1)])
        return self.transformer(x, c)


class Embedder(nn.Module):
    """Action embedding: optional 1x1 conv, then a two-layer SiLU MLP."""

    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=10, mlp_scale=4, patch_embed=True):
        super().__init__()
        self.patch_embed = (nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
                            if patch_embed else None)
        feature_dim = smoothed_dim if patch_embed else input_dim
        self.embed = nn.Sequential(
            nn.Linear(feature_dim, mlp_scale * emb_dim), nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.float()
        if self.patch_embed is not None:
            if x.dim() == 3:
                x = self.patch_embed(x.permute(0, 2, 1)).permute(0, 2, 1)
            elif x.dim() == 4:
                b, t, c, d = x.shape
                x = self.patch_embed(x.reshape(b, t * c, d).permute(0, 2, 1)).permute(0, 2, 1)
                x = x.reshape(b, t, c, -1)
            else:
                raise ValueError(f"expected 3D or 4D input, got shape {tuple(x.shape)}")
        return self.embed(x)


class PerStepMLP(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, d_model)

    def forward(self, x):
        return self.fc1(x)


class SequenceEncoder(nn.Module):
    """Pools each chunk (B, T, L, C) of lower-level action embeddings into one latent
    macro-action (B, T, output_dim) through a CLS token. `padding_mask` is True on padding."""

    def __init__(self, output_dim=4, input_dim=10, d_model=64, nhead=2, num_layers=1,
                 ff_mult=2, dropout=0.1, max_chunk=15, norm_first=True):
        super().__init__()
        self.max_chunk = max_chunk
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.step_mlp = PerStepMLP(input_dim=input_dim, d_model=d_model)
        self.pos = nn.Parameter(torch.zeros(max_chunk + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_mult * d_model, dropout=dropout,
            batch_first=True, norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
        )
        self.final_ln = nn.LayerNorm(output_dim, elementwise_affine=True, eps=1e-6)

    def forward(self, inputs, padding_mask=None):
        B, T, L, C = inputs.shape
        assert C == self.input_dim, f"last dim of inputs must be {self.input_dim}, got {C}"
        if padding_mask is None:
            padding_mask = torch.zeros(B, T, L, dtype=torch.bool, device=inputs.device)
        x = self.step_mlp(inputs.reshape(B * T, L, C)) + self.pos[1: L + 1].unsqueeze(0)
        cls_tok = self.pos[0].view(1, 1, -1).expand(B * T, 1, -1)
        x = torch.cat([cls_tok, x], dim=1)
        mask = torch.cat([torch.zeros(B * T, 1, dtype=torch.bool, device=x.device),
                          padding_mask.reshape(B * T, L)], dim=1)
        y = self.encoder(x, src_key_padding_mask=mask)
        return self.final_ln(self.head(y[:, 0, :])).view(B, T, -1)


class SequenceMLPEncoder(nn.Module):
    """The `type: mlp` action encoder: flatten each fixed-length chunk into one MLP."""

    def __init__(self, output_dim, hidden_dims, input_dim, temporal_stride, final_ln=True,
                 trunc_normal_init=False):
        super().__init__()
        (hidden_dim,) = hidden_dims
        self.output_dim = int(output_dim)
        self.net = MLP(input_dim=int(temporal_stride) * int(input_dim),
                       hidden_dim=int(hidden_dim), output_dim=self.output_dim, norm_fn=None,
                       act_fn=nn.ReLU, final_ln=bool(final_ln))
        if trunc_normal_init:
            self.apply(init_module_weights)

    def forward(self, inputs, padding_mask=None):
        B, T, L, C = inputs.shape
        return self.net(inputs.reshape(B * T, L * C)).reshape(B, T, self.output_dim)


class FlattenedSequenceEncoder(nn.Module):
    """SequenceEncoder as a state encoder over (B*T, K, D) windows (`type: seq_encoder`)."""

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = SequenceEncoder(**kwargs)
        self.output_dim = self.encoder.output_dim

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"expected (B*T, K, D), got {tuple(x.shape)}")
        return self.encoder(x[:, None])[:, 0]


VIT_SIZES = {
    "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
    "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
    "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
    "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
    "huge": {"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 16},
}


def create_hf_vit(size="tiny", patch_size=16, image_size=224, use_mask_token=True, **kwargs):
    from transformers import ViTConfig, ViTModel

    if size not in VIT_SIZES:
        raise ValueError(f"unknown ViT size '{size}', choose from {list(VIT_SIZES)}")
    params = dict(VIT_SIZES[size])
    params["intermediate_size"] = params["hidden_size"] * 4
    params.update(image_size=image_size, patch_size=patch_size, **kwargs)
    model = ViTModel(ViTConfig(**params), add_pooling_layer=False, use_mask_token=use_mask_token)
    model.config.interpolate_pos_encoding = True
    return model


class FusedViT(nn.Module):
    """One ViT over several cameras and the proprioceptive state, read out through one CLS.

    Every camera is patch-embedded by the same convolution and gets its own position
    embeddings (the first camera's start as the ViT's own) plus a learned per-camera
    embedding; the state vector becomes one extra token through a linear projection. A
    single transformer attends across all of them, so the CLS token is the fused state.

    forward(images (N, V, 3, H, W), proprio (N, P) or None) -> CLS (N, hidden)
    """

    def __init__(self, num_views=2, proprio_dim=0, **vit_kwargs):
        super().__init__()
        self.vit = create_hf_vit(**vit_kwargs)
        emb = self.vit.embeddings
        hidden = int(self.vit.config.hidden_size)
        n = emb.patch_embeddings.num_patches
        self.num_views = int(num_views)
        self.extra_view_pos = nn.Parameter(torch.zeros(self.num_views - 1, n, hidden))
        nn.init.trunc_normal_(self.extra_view_pos, std=0.02)
        self.view_embed = nn.Parameter(torch.zeros(self.num_views, 1, hidden))
        nn.init.trunc_normal_(self.view_embed, std=0.02)
        self.proprio_dim = int(proprio_dim)
        if self.proprio_dim:
            self.proprio_proj = nn.Linear(self.proprio_dim, hidden)
            self.proprio_embed = nn.Parameter(torch.zeros(1, 1, hidden))
            nn.init.trunc_normal_(self.proprio_embed, std=0.02)
        self.config = self.vit.config

    def forward(self, images, proprio=None):
        N, V = images.shape[:2]
        assert V == self.num_views, f"expected {self.num_views} views, got {V}"
        emb = self.vit.embeddings
        patches = emb.patch_embeddings(images.flatten(0, 1).to(emb.patch_embeddings.projection.weight.dtype))
        patches = patches.unflatten(0, (N, V))
        pos = torch.cat([emb.position_embeddings[:, 1:], self.extra_view_pos], dim=0)
        patches = patches + pos.unsqueeze(0) + self.view_embed.unsqueeze(0)
        tokens = [emb.cls_token.expand(N, -1, -1) + emb.position_embeddings[:, :1],
                  patches.flatten(1, 2)]
        if self.proprio_dim:
            assert proprio is not None, "this FusedViT reads a proprio vector"
            tokens.append(self.proprio_proj(proprio.to(patches.dtype)).unsqueeze(1) + self.proprio_embed)
        x = emb.dropout(torch.cat(tokens, dim=1))
        for layer in self.vit.layers:
            x = layer(x, None)
        return self.vit.layernorm(x)[:, 0]


def build_encoder(cfg, *, default_patch_size=None, default_image_size=None):
    """levelN.encoder -> (module, output width). Types: vit, mlp, latent_mlp, seq_encoder,
    identity."""
    cfg = dict(cfg)
    kind = str(cfg.pop("type")).lower()
    if kind == "vit":
        if "size" not in cfg and "scale" in cfg:
            cfg["size"] = cfg.pop("scale")
        if default_patch_size is not None:
            cfg.setdefault("patch_size", default_patch_size)
        if default_image_size is not None:
            cfg.setdefault("image_size", default_image_size)
        enc = create_hf_vit(**cfg)
        return enc, int(enc.config.hidden_size)
    if kind == "fused_vit":
        if "size" not in cfg and "scale" in cfg:
            cfg["size"] = cfg.pop("scale")
        if default_patch_size is not None:
            cfg.setdefault("patch_size", default_patch_size)
        if default_image_size is not None:
            cfg.setdefault("image_size", default_image_size)
        enc = FusedViT(**cfg)
        return enc, int(enc.config.hidden_size)
    if kind == "mlp":
        width = cfg.get("output_dim", cfg.get("input_dim"))
        return MLP(**cfg), int(width)
    if kind == "latent_mlp":
        return ResidualLatentMLP(**cfg), int(cfg.get("output_dim", cfg["input_dim"]))
    if kind == "seq_encoder":
        enc = FlattenedSequenceEncoder(**cfg)
        return enc, int(enc.output_dim)
    if kind == "identity":
        width = cfg.get("input_dim", cfg.get("output_dim"))
        if cfg.get("input_dim") is not None and cfg.get("output_dim") is not None:
            assert int(cfg["input_dim"]) == int(cfg["output_dim"]), (
                "identity encoder cannot change width"
            )
        return nn.Identity(), int(width)
    raise ValueError(f"unsupported encoder type '{kind}'")
