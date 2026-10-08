"""Observation encoders feeding the predictor, frozen or jointly trained.

An encoder maps one raw batch field to what its spec entry's `via` demands:
a (B, N, dim) token tensor for token streams (N = len(index) * grid area),
a (B, S, dim) tensor for a shared cross source, or a list of `depth` tensors
for a per-layer cross source.

Two spec keys compose encoders without new code:

  frozen  no grads, permanent eval mode. Pretrained types (vjepa2, smolvlm)
          default frozen; from-scratch types (vit) and heads default trainable.
  input   names another `encoders:` entry whose output this one consumes. The
          stream's raw batch field feeds the chain's root (the encoder with no
          `input`); `resolve_chains` orders the chain and rejects cycles, so a
          chain always bottoms out at real data.

`raw_steps_per_index` declares how many raw time steps one index step of the
encoded stream consumes (2 frames per VJEPA2 tubelet), so the algorithm can
slice each spec entry's window out of a shared raw field before encoding.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossAttention
from .layers import SwiGLULinear, make_mlp
from .pixel_decoder import PixelDecoder
from .vit import ViT, ViTBlock

__all__ = [
    "Encoder",
    "ViTEncoder",
    "VJEPA2Encoder",
    "ResNetEncoder",
    "AttentivePoolEncoder",
    "LinearEncoder",
    "MLPEncoder",
    "ChunkMLPEncoder",
    "ReshapeEncoder",
    "PixelDecoderEncoder",
    "SmolVLMEncoder",
    "build_encoder",
    "resolve_chains",
]


def build_encoder(spec):
    """One `encoders:` section entry -> module, keyed by its `type` field."""
    kind = spec["type"]
    if kind not in _ENCODER_TYPES:
        raise ValueError(f"unknown encoder type '{kind}'. available: {sorted(_ENCODER_TYPES)}")
    encoder = _ENCODER_TYPES[kind](spec)
    if encoder.frozen:
        encoder.requires_grad_(False)
        encoder.eval()
    return encoder


def resolve_chains(encoder_cfgs):
    """`encoders:` section -> {name: [root, ..., name]} execution order per encoder.

    Follows each entry's `input` link back to a root that consumes the raw batch
    field. Unknown names and cycles are build-time errors: a cyclic chain would
    have no raw input at all.
    """
    chains = {}

    def resolve(name, visiting):
        if name in chains:
            return chains[name]
        if name in visiting:
            cycle = " -> ".join([*visiting, name])
            raise ValueError(f"encoder chain feeds back into itself ({cycle}): no raw input")
        parent = encoder_cfgs[name].get("input")
        if parent is None:
            chain = [name]
        elif parent not in encoder_cfgs:
            raise ValueError(f"encoder '{name}' takes input from unknown encoder '{parent}'")
        else:
            chain = resolve(parent, [*visiting, name]) + [name]
        chains[name] = chain
        return chain

    for name in encoder_cfgs:
        resolve(name, [])
    return chains


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Encoder(nn.Module):
    """Shared contract: forward(raw) returns the tensor(s) the trunk expects for
    every spec entry that names this encoder. `frozen` (spec key, per-type
    default) is applied by build_encoder and pins the module in eval mode.

    `supports_encoding_cache` marks the pixel backbones whose frozen output the
    encoding cache (utils/enc_cache.py) may precompute. It is a property of the
    TYPE, not of `frozen`: a probe run freezes every encoder (freeze_world_model),
    which must not turn a cheap non-visual codec (chunk_mlp on actions) into a
    "cacheable" stream -- the cache stores visual windows, and the dataset rejects
    a non-visual cache field.
    """

    raw_steps_per_index = 1
    default_frozen = False
    supports_encoding_cache = False

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.frozen = bool(spec.get("frozen", self.default_frozen))

    def train(self, mode=True):
        # the owning algorithm flips train/eval globally; a frozen encoder must keep
        # eval-mode dropout/BN statistics regardless
        return super().train(mode and not self.frozen)

    def forward(self, raw):
        raise NotImplementedError


class ViTEncoder(Encoder):
    """From-scratch ViT over (B, T, C, H, W) uint8 frames, trained end-to-end with the rest of
    the model. `pooling` picks what a frame becomes:

      grid (default)  -> (B, T*Hp*Wp, dim) patch tokens. By default RoPE spans all three axes
                         (frame, row, col) so attention crosses frames; set
                         `attend_across_time: false` to fold frames into the batch dim
                         instead, so attention stays within each frame's own patches (2D
                         RoPE over row/col only -- see `ViT`). The patch grid
                         (H/patch_size, W/patch_size) must equal the consuming stream's
                         `grid` either way.
      cls             -> (B, T, dim), one vector per frame: a learnable CLS token is prepended,
                         learnable positional embeddings replace RoPE, and the final-LayerNorm'd
                         CLS output is returned. This is the LeWorldModel representation; an
                         anti-collapse objective (sigreg) should not read this LayerNorm'd
                         output directly -- chain a `type: mlp` projector encoder (`input:` this
                         one) and point the stream at it, so training, rollout, and the eval
                         goal encoding all share the projected space. The consuming stream's
                         `grid` is [1, 1].

    Pixels are normalized here (kept uint8 through the data pipeline): `float/255` then, when
    `center`, mapped to [-1, 1].

    Spec: `dim`, `patch_size`, `depth`, `num_heads`, `mlp_ratio` (ViT geometry), optional
    `in_channels` (default 3), `center` (default true), `pooling` (default grid). `cls` pooling
    also reads `img_size` (frames are resized to it); `grid` pooling reads `rope`, `img_size`,
    and, when `attend_across_time` (default true) is true, `frames` -- together they bound the
    geometry its RoPE ladder is built for. `attend_across_time: false` folds frames into the
    batch dim so attention stays within each frame's own patches; see `ViT`.
    """

    supports_encoding_cache = True

    def __init__(self, spec):
        super().__init__(spec)
        self.center = spec.get("center", True)
        self.pixel_norm = spec.get("pixel_norm")
        assert self.pixel_norm in (None, "imagenet"), f"unknown pixel_norm '{self.pixel_norm}'"
        if self.pixel_norm == "imagenet":
            self.register_buffer(
                "px_mean", torch.tensor(_IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
            self.register_buffer(
                "px_std", torch.tensor(_IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False)
        self.gradient_checkpointing = False
        self.pooling = spec.get("pooling", "grid")
        assert self.pooling in ("grid", "cls"), f"unknown pooling '{self.pooling}'"
        dim = int(spec["dim"])

        if self.pooling == "grid":
            self.vit = ViT(
                in_channels=int(spec.get("in_channels", 3)),
                dim=dim,
                patch_size=int(spec["patch_size"]),
                depth=int(spec["depth"]),
                num_heads=int(spec["num_heads"]),
                mlp_ratio=float(spec.get("mlp_ratio", 4.0)),
                rope=spec.get("rope"),
                img_size=spec.get("img_size"),
                frames=spec.get("frames"),
                attend_across_time=spec.get("attend_across_time", True),
            )
            return

        self.img_size = int(spec["img_size"])
        patch_size = int(spec["patch_size"])
        num_patches = (self.img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            int(spec.get("in_channels", 3)), dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            ViTBlock(dim, int(spec["num_heads"]), float(spec.get("mlp_ratio", 4.0)))
            for _ in range(int(spec["depth"]))
        ])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, frames):
        x = frames.float() / 255.0
        if self.pixel_norm == "imagenet":
            x = (x - self.px_mean) / self.px_std
        elif self.center:
            x = (x - 0.5) / 0.5
        if self.pooling == "grid":
            return self.vit(x)

        B, T = frames.shape[:2]
        x = x.flatten(0, 1)                                       # (B*T, C, H, W)
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear",
                              align_corners=False)
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B*T, num_patches, dim)
        cls = self.cls_token.expand(patches.shape[0], -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.pos_embedding
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, None, use_reentrant=False)
            else:
                tokens = block(tokens, freqs=None)                # learned pos-emb, no RoPE
        cls_out = self.norm(tokens)[:, 0]                         # (B*T, dim)
        return cls_out.reshape(B, T, -1)


def vjepa_module(name):
    """Import module `name` (e.g. "app.vjepa_2_1.models.vision_transformer") from the
    external/vjepa2 submodule.

    The vjepa2 repo is not a package: its files import each other as top-level
    `src.*` / `app.*`, assuming the repo root is on sys.path. Append (not prepend)
    the submodule root so those names resolve without shadowing real packages.

    Installing it instead is not an option: its setup.py declares no packages, so
    discovery flattens `src/` to top-level `datasets`/`models`/`utils` (colliding with
    real ones) and drops `app/` altogether -- the half this actually imports.
    """
    import importlib

    root = Path(__file__).resolve().parents[3] / "external" / "vjepa2"
    if not (root / "app").is_dir():
        raise ImportError(
            f"vjepa2 submodule not found at {root}; "
            f"run `git submodule update --init external/vjepa2`"
        )
    if str(root) not in sys.path:
        sys.path.append(str(root))
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as err:
        # the submodule is source on sys.path, not an installed distribution, so its
        # own requirements never get resolved
        raise ImportError(
            f"the vjepa2 submodule needs '{err.name}', which is not installed. "
            f"try: uv pip install 'bachelor-thesis[vjepa]'"
        ) from err


def _vjepa_vision_transformer():
    return vjepa_module("app.vjepa_2_1.models.vision_transformer")


class VJEPA2Encoder(Encoder):
    """Pretrained V-JEPA 2.1 video backbone over (B, T, C, H, W) uint8 frames.

    Frames are resized to `crop_size` (an int for a square, or [H, W] to keep a wide
    input such as two stitched camera views at its own aspect), ImageNet-normalized, and
    grouped into 2-frame tubelets; output is (B, steps * H/16 * W/16, dim) patch latents.
    `frames_per_step` sets how a tubelet is filled from the raw window:

      1 (default)  each raw frame is duplicated into its own tubelet, so one index
                   step = one frame = one latent step (works with every dataset
                   backend; temporal attention still spans the whole window)
      2            consecutive raw frame pairs form true tubelets; the batch field
                   must carry 2 raw rows per index step

    Spec: `arch` (any vjepa2 factory: vit_tiny 192 / vit_small 384 / vit_base 768 /
    vit_large 1024 / vit_huge 1280 / vit_giant 1408 / vit_gigantic 1664),
    `checkpoint` (see below; null keeps random weights, for smoke tests only),
    `crop_size`, `frames_per_step`, `backbone` (extra kwargs forwarded to the
    factory), `frozen` (default true).

    `checkpoint` accepts a local .pt path, an URL, or the alias `pretrained`, which
    resolves to the official V-JEPA 2.1 release for the arch (vit_base 80M /
    vit_large 300M / vit_giant 1B / vit_gigantic 2B). URLs are downloaded once into
    `$TORCH_HOME/hub/checkpoints` (default ~/.cache/torch; set TORCH_HOME to move
    the cache off a small home disk).
    """

    default_frozen = True
    supports_encoding_cache = True
    TUBELET_SIZE = 2

    PRETRAINED_URLS = {
        "vit_base": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        "vit_large": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
        "vit_giant": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt",
        "vit_gigantic": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt",
    }

    def __init__(self, spec):
        super().__init__(spec)
        # an int is a square crop; [H, W] keeps a wide input at its own aspect, and
        # the stream's `grid` becomes [H/16, W/16]
        crop = spec.get("crop_size", 256)
        self.crop_size = ((int(crop), int(crop)) if isinstance(crop, (int, float))
                          else tuple(int(v) for v in crop))
        self.raw_steps_per_index = int(spec.get("frames_per_step", 1))
        assert self.raw_steps_per_index in (1, self.TUBELET_SIZE), (
            f"frames_per_step must be 1 or {self.TUBELET_SIZE}"
        )
        # `per_frame: true` encodes every index step as its own SINGLE-FRAME clip in the
        # batch dimension (the backbone's `img_temporal_dim_size: 1` image path), so
        # encoder attention never crosses time; every step sits at temporal RoPE position 0
        self.per_frame = bool(spec.get("per_frame", False))
        assert not (self.per_frame and self.raw_steps_per_index != 1), (
            "per_frame feeds one raw frame per index step; set frames_per_step: 1 "
            "and un-tubeleted raw_index strings in the arm config"
        )

        vt = _vjepa_vision_transformer()
        arch = spec.get("arch", "vit_large")
        if not arch.startswith("vit_") or not hasattr(vt, arch):
            available = sorted(n for n in dir(vt) if n.startswith("vit_"))
            raise ValueError(f"unknown vjepa2 arch '{arch}'. available: {available}")
        kwargs = {
            "img_size": self.crop_size,
            "num_frames": 16,
            "tubelet_size": self.TUBELET_SIZE,
            "use_sdpa": True,
            "use_silu": False,
            "wide_silu": True,
            "uniform_power": True,
            "use_rope": True,
            "img_temporal_dim_size": 1,
            "interpolate_rope": True,
            **(spec.get("backbone") or {}),
        }
        self.backbone = getattr(vt, arch)(**kwargs)
        if spec.get("checkpoint"):
            self._load_checkpoint(self._resolve_checkpoint(spec["checkpoint"], arch))

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(3, 1, 1))

    def _resolve_checkpoint(self, ref, arch):
        """`pretrained` -> official URL for the arch; URLs -> cached local download."""
        ref = str(ref)
        if ref == "pretrained":
            if arch not in self.PRETRAINED_URLS:
                raise ValueError(
                    f"no published V-JEPA 2.1 weights for '{arch}'; "
                    f"available: {sorted(self.PRETRAINED_URLS)}"
                )
            ref = self.PRETRAINED_URLS[arch]
        if ref.startswith(("http://", "https://")):
            dest = Path(torch.hub.get_dir()) / "checkpoints" / ref.rsplit("/", 1)[-1]
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                # download_url_to_file writes a temp file and renames, so concurrent ranks
                # at worst download twice, never read a torn file
                torch.hub.download_url_to_file(ref, str(dest), progress=True)
            ref = str(dest)
        return ref

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # a V-JEPA 2.1 training checkpoint nests the weights (EMA preferred)
        state = next((ckpt[k] for k in ("ema_encoder", "target_encoder", "encoder")
                      if isinstance(ckpt, dict) and k in ckpt), ckpt)
        state = {
            k.removeprefix("module.").removeprefix("backbone."): v for k, v in state.items()
        }
        self.backbone.load_state_dict(state, strict=True)

    @property
    def gradient_checkpointing(self):
        return self.backbone.use_activation_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self.backbone.use_activation_checkpointing = bool(value)

    def forward(self, frames):
        if frames.ndim == 4:
            frames = frames.unsqueeze(1)
        if self.raw_steps_per_index == 1 and not self.per_frame:
            frames = frames.repeat_interleave(self.TUBELET_SIZE, dim=1)
        B, T = frames.shape[:2]
        if not self.per_frame:
            assert T % self.TUBELET_SIZE == 0, f"window of {T} frames does not fill tubelets"

        x = frames.float()
        if frames.dtype == torch.uint8:
            x = x / 255.0
        x = x.flatten(0, 1)                                       # (B*T, C, H, W)
        if x.shape[-2:] != self.crop_size:
            x = F.interpolate(x, size=self.crop_size, mode="bilinear", antialias=True)
        x = (x - self.mean) / self.std
        dtype = next(self.backbone.parameters()).dtype
        if self.per_frame:
            # each step its own T=1 clip; output stays time-major (B, T*P, D), the
            # same layout the clip path produces
            out = self.backbone(x.unsqueeze(2).to(dtype))         # (B*T, C, 1, H, W)
            return out.unflatten(0, (B, T)).flatten(1, 2)
        x = x.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)         # (B, C, T, H, W)
        return self.backbone(x.to(dtype))


class ResNetEncoder(Encoder):
    """torchvision ResNet over (B, T, C, H, W) uint8 frames, ImageNet-normalized.
    `pooling` picks what a frame becomes, mirroring ViTEncoder's vocabulary:

      grid (default)  -> (B, T * h * w, dim) tokens from the final conv map, where
                         (h, w) = (H/32, W/32) must equal the consuming stream's `grid`
      avg             -> (B, T, dim), one pooled vector per frame; stream `grid` [1, 1]

    `dim` is fixed by the variant (512 for resnet18/34, 2048 for resnet50+). Spec:
    `variant` (default resnet18), `weights` ("pretrained" for the torchvision ImageNet
    weights, null for random init), `img_size` (resize input frames; null keeps native
    size), `frozen` (default false: like the from-scratch ViT, a ResNet is commonly
    trained jointly even from ImageNet weights — freeze it explicitly when chaining).
    """

    supports_encoding_cache = True

    def __init__(self, spec):
        super().__init__(spec)
        try:
            from torchvision import models
        except ImportError as err:
            raise ImportError("ResNetEncoder needs torchvision (installed with lerobot)") from err

        variant = spec.get("variant", "resnet18")
        if not variant.startswith("resnet"):
            raise ValueError(f"unknown resnet variant '{variant}'")
        weights = "DEFAULT" if spec.get("weights") == "pretrained" else None
        model = models.get_model(variant, weights=weights)

        self.pooling = spec.get("pooling", "grid")
        assert self.pooling in ("grid", "avg"), f"unknown pooling '{self.pooling}'"
        self.dim = model.fc.in_features
        size = spec.get("img_size")
        self.img_size = int(size) if size else None
        self.backbone = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
        )

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(3, 1, 1))

    def forward(self, frames):
        if frames.ndim == 4:
            frames = frames.unsqueeze(1)
        B, T = frames.shape[:2]
        x = frames.float()
        if frames.dtype == torch.uint8:
            x = x / 255.0
        x = x.flatten(0, 1)                                       # (B*T, C, H, W)
        if self.img_size and x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear",
                              antialias=True)
        x = (x - self.mean) / self.std
        feat = self.backbone(x)                                   # (B*T, dim, h, w)
        if self.pooling == "avg":
            return feat.mean(dim=(-2, -1)).reshape(B, T, self.dim)
        tokens = feat.flatten(2).transpose(1, 2)                  # (B*T, h*w, dim)
        return tokens.reshape(B, -1, self.dim)


class AttentivePoolEncoder(Encoder):
    """Learned-query attention pooling over another encoder's patch tokens:
    (B, steps * tokens_per_step, in_dim) -> (B, steps * num_queries, out_dim).

    Chained after a patch-token encoder (`input: <name>`), this gives a backbone the CLS
    token it never had: `num_queries` learned summary tokens per step. The queries are
    parameters, not derived from the input, so they select what to keep by content
    instead of averaging with fixed weights, and they train while a frozen backbone
    stays frozen.

    Steps are pooled INDEPENDENTLY by a shared query: the input is reshaped to
    (B * steps, tokens_per_step, in_dim), so no query ever mixes two timesteps. That is
    what keeps one summary token per index step, which in turn is what lets a
    `statistic: per_timestep` sigreg term give each step its own sample bag, and what
    shrinks the trunk's stream from a full patch grid to `steps * num_queries` tokens.

    The consuming stream declares `dim: out_dim` and `grid: [1, num_queries]`.

    Attention is the trunk's own CrossAttention (QK-norm, per-source K/V projections), so
    `in_dim` may differ from `out_dim`: pooling and the projection into the trunk width
    happen in one step, with no separate `linear` head after it. Keys carry no positional
    encoding, because the tokens being pooled are a backbone's contextualized outputs
    whose content already reflects where they came from.

    Spec: `in_dim` (the source encoder's dim), `out_dim`, `tokens_per_step` (source tokens
    per index step, i.e. the source's grid area -- 36 for a vjepa2 6x6 grid),
    `num_queries` (default 1), `num_heads` (default 8), `mlp_ratio` (default 4.0),
    `frozen` (default false).
    """

    def __init__(self, spec):
        super().__init__(spec)
        in_dim, out_dim = int(spec["in_dim"]), int(spec["out_dim"])
        self.tokens_per_step = int(spec["tokens_per_step"])
        self.num_queries = int(spec.get("num_queries", 1))
        num_heads = int(spec.get("num_heads", 8))
        assert out_dim % num_heads == 0, (
            f"attentive pool out_dim {out_dim} must divide into {num_heads} heads"
        )

        self.query = nn.Parameter(torch.randn(1, self.num_queries, out_dim) * 0.02)
        self.norm_src = nn.LayerNorm(in_dim, eps=1e-6)
        self.attn = CrossAttention(out_dim, num_heads, [in_dim])
        # the query enters attention unnormalized: it is a free parameter, so a norm
        # in front of it only reparametrizes what the parameter already learns
        self.norm_q = nn.LayerNorm(out_dim, eps=1e-6)
        self.mlp = make_mlp(out_dim, int(out_dim * float(spec.get("mlp_ratio", 4.0))), out_dim)

    def forward(self, x):
        b, n, _ = x.shape
        span = self.tokens_per_step
        if n % span:
            raise ValueError(
                f"attentive pool got {n} tokens, not a multiple of tokens_per_step="
                f"{span}; tokens_per_step must be the source encoder's grid area"
            )
        src = self.norm_src(x.reshape(b * (n // span), span, -1))
        q = self.query.expand(src.shape[0], -1, -1)
        q = q + self.attn(q, [src])
        q = q + self.mlp(self.norm_q(q))
        return q.reshape(b, -1, q.shape[-1])


class LinearEncoder(Encoder):
    """Linear projection head: (..., in_dim) -> (..., out_dim). Chained after a
    pretrained encoder (`input: <name>`), it is the cheapest trainable adapter.
    Spec: `in_dim`, `out_dim`, `bias` (default true).
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.proj = nn.Linear(int(spec["in_dim"]), int(spec["out_dim"]),
                              bias=bool(spec.get("bias", True)))

    def forward(self, x):
        return self.proj(x)


class BatchNormEncoder(Encoder):
    """BatchNorm over the last dim: (..., dim) -> (..., dim), statistics over every other
    axis (batch, steps, tokens). Chained after a trainable pool (`input: <name>`), it pins
    the latent a flow stream targets to zero mean and unit variance per dim -- the scale of
    the flow's Gaussian noise -- which an end-to-end encoder is otherwise free to shrink
    toward, since smaller targets lower the flow loss.

    Spec: `dim`, `affine` (default false: a learned per-dim scale could shrink the latent
    all over again), `momentum` (default 0.1), `eps` (default 1e-5).
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.norm = nn.BatchNorm1d(
            int(spec["dim"]),
            affine=bool(spec.get("affine", False)),
            momentum=float(spec.get("momentum", 0.1)),
            eps=float(spec.get("eps", 1e-5)),
        )

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        if self.training and flat.shape[0] < 2:
            # one row has no batch statistics (BatchNorm1d raises); the auto-batch probe
            # starts at batch 1 and a future stream is one latent per sample. Normalize with
            # the running statistics, as eval does, and leave them untouched
            n = self.norm
            return F.batch_norm(flat, n.running_mean, n.running_var, n.weight, n.bias,
                                training=False, eps=n.eps).reshape(x.shape)
        return self.norm(flat).reshape(x.shape)


class MLPEncoder(Encoder):
    """SwiGLU MLP over the last dim: norm(w(x)) * silu(w_gate(x)) -> Linear,
    i.e. the trunk's make_mlp with an optional norm on the value path. Encodes a
    raw low-dimensional field (e.g. the stacked action block conditioning the
    trunk via AdaLN) or, chained with `input:`, adapts another encoder's output.

    Spec: `in_dim`, `out_dim`, `hidden_dim` (default 4 * out_dim), `norm`
    ("batch", "layer", or null -- default null). BatchNorm sees the flattened
    leading dims as the batch.
    """

    def __init__(self, spec):
        super().__init__(spec)
        in_dim, out_dim = int(spec["in_dim"]), int(spec["out_dim"])
        hidden = int(spec.get("hidden_dim", 4 * out_dim))
        norm = spec.get("norm")
        norms = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm}
        assert norm is None or norm in norms, f"unknown MLPEncoder norm '{norm}'"
        self.mlp = nn.Sequential(
            SwiGLULinear(in_dim, hidden, norm=norms[norm](hidden) if norm else None),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        x = x.float()
        lead, d = x.shape[:-1], x.shape[-1]
        return self.mlp(x.reshape(-1, d)).reshape(*lead, -1)


class ChunkMLPEncoder(Encoder):
    """Chunk-level MLP codec: (B, k * in_steps, in_dim) -> (B, k * out_steps, out_dim),
    each in_steps-row chunk flattened and mapped as one vector, chunks independently.

    The building block of an action (or any low-dim window) autoencoder: one instance
    with `in_steps: 36, out_steps: 1` is the encoder E_a compressing a whole action chunk
    into latent token(s), its mirror with `in_steps: 1, out_steps: 36` the decoder D_a.
    Named as a predict stream's `encoder:` it makes that stream flow in the codec's
    latent space (the stream's `raw_index` lists the chunk's raw rows and `dim`/`grid`
    its latent layout); named as the stream's `decoder:` it maps the ODE endpoint back
    to raw rows; named by an `ae_recon` loss term it closes the reconstruction loop.

    Spec: `in_dim`, `out_dim`, `in_steps` (rows consumed per chunk, default 1; becomes
    `raw_steps_per_index`), `out_steps` (rows emitted per chunk, default 1), `hidden_dim`
    (default 4 * out_dim), `depth` (hidden SwiGLU layers, default 1), `norm` ("batch",
    "layer", or null), `frozen` (default false).
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.in_dim, self.out_dim = int(spec["in_dim"]), int(spec["out_dim"])
        self.in_steps = int(spec.get("in_steps", 1))
        self.out_steps = int(spec.get("out_steps", 1))
        self.raw_steps_per_index = self.in_steps
        hidden = int(spec.get("hidden_dim", 4 * self.out_dim))
        depth = int(spec.get("depth", 1))
        norm = spec.get("norm")
        norms = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm}
        assert norm is None or norm in norms, f"unknown ChunkMLPEncoder norm '{norm}'"
        layers, width = [], self.in_steps * self.in_dim
        for _ in range(depth):
            layers.append(SwiGLULinear(width, hidden, norm=norms[norm](hidden) if norm else None))
            width = hidden
        layers.append(nn.Linear(width, self.out_steps * self.out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        b, n, d = x.shape
        if n % self.in_steps:
            raise ValueError(
                f"chunk_mlp got {n} rows, not a multiple of in_steps={self.in_steps}"
            )
        k = n // self.in_steps
        z = self.net(x.reshape(b * k, self.in_steps * d))
        return z.reshape(b, k * self.out_steps, self.out_dim)


class ReshapeEncoder(Encoder):
    """chunk_mlp's rows/features regrouping with no network in it: (B, k * in_steps, in_dim)
    -> (B, k * out_steps, out_dim), each in_steps-row chunk read as one flat vector and
    re-split, chunks independently. `in_steps * in_dim` must equal `out_steps * out_dim`, so
    nothing is learned, lost or mixed -- only where the row/feature boundary falls moves.

    What a predicted stream wants when the whole chunk should be ONE token but the flow must
    stay in raw units: `in_steps: 8, out_steps: 1` makes the flow space the concatenated
    chunk itself, the trunk's own token_in/token_out become the only projection in or out,
    and the mirror (`in_steps: 1, in_dim: 16, out_steps: 8, out_dim: 2`) is the `decoder:`.
    Unlike a chunk_mlp codec this needs no `ae_recon`/`sigreg` to hold it up: an exact
    bijection has nothing to collapse.

    Spec: `in_dim`, `out_dim`, `in_steps` (default 1; becomes `raw_steps_per_index`),
    `out_steps` (default 1).
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.in_dim, self.out_dim = int(spec["in_dim"]), int(spec["out_dim"])
        self.in_steps = int(spec.get("in_steps", 1))
        self.out_steps = int(spec.get("out_steps", 1))
        self.raw_steps_per_index = self.in_steps
        if self.in_steps * self.in_dim != self.out_steps * self.out_dim:
            raise ValueError(
                f"reshape moves rows into features and back, so it cannot change the value "
                f"count: {self.in_steps} x {self.in_dim} != {self.out_steps} x {self.out_dim}"
            )

    def forward(self, x):
        x = x.float()
        b, n, d = x.shape
        if d != self.in_dim:
            raise ValueError(f"reshape got {d}-dim rows, not in_dim={self.in_dim}")
        if n % self.in_steps:
            raise ValueError(
                f"reshape got {n} rows, not a multiple of in_steps={self.in_steps}"
            )
        k = n // self.in_steps
        return x.reshape(b, k * self.out_steps, self.out_dim)


class PixelDecoderEncoder(Encoder):
    """Latent tokens back to pixels: (B, steps * tokens_per_step, in_dim) ->
    (B, steps, 3, H, W) images in [0, 1] units (see pixel_decoder.PixelDecoder).

    Chained after the encoder whose representation is being probed (`input: <name>`), or
    named as a `decoder:` by a predict stream, this is the visualization end of the repo:
    it answers what a latent still knows about the frame it came from. Steps are decoded
    INDEPENDENTLY (folded into the batch, as AttentivePoolEncoder pools them), so one index
    step of the source stream becomes one image and the query set never mixes two frames.

    Spec: `in_dim` (the source encoder's dim), `tokens_per_step` (source tokens per index
    step -- 1 for a pooled summary token, the grid area for a patch grid), `img_size` (int
    or [H, W], default 224), `patch_size` (default 16), `hidden_dim` (default 512), `depth`
    (default 4), `num_heads` (default 8), `mlp_ratio` (default 4.0), `frozen` (default
    false).
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.tokens_per_step = int(spec.get("tokens_per_step", 1))
        self.in_dim = int(spec["in_dim"])
        self.decoder = PixelDecoder(
            in_dim=self.in_dim,
            img_size=spec.get("img_size", 224),
            patch_size=int(spec.get("patch_size", 16)),
            hidden_dim=int(spec.get("hidden_dim", 512)),
            depth=int(spec.get("depth", 4)),
            num_heads=int(spec.get("num_heads", 8)),
            mlp_ratio=float(spec.get("mlp_ratio", 4.0)),
        )

    @property
    def img_size(self):
        return self.decoder.img_size

    def forward(self, x):
        b, n, _ = x.shape
        span = self.tokens_per_step
        if n % span:
            raise ValueError(
                f"pixel decoder got {n} tokens, not a multiple of tokens_per_step={span}; "
                f"tokens_per_step must be the source stream's tokens per index step"
            )
        img = self.decoder(x.reshape(b * (n // span), span, -1))
        return img.unflatten(0, (b, n // span))


class SmolVLMEncoder(Encoder):
    """Frozen SmolVLM2 LM prefix: list of B instruction strings -> list of
    `num_layers` (B, S, dim) hidden-state tensors, one per DiT block.
    """

    default_frozen = True

    def forward(self, texts):
        raise NotImplementedError(
            "SmolVLMEncoder is a stub"
        )


_ENCODER_TYPES = {
    "vit": ViTEncoder,
    "vjepa2": VJEPA2Encoder,
    "resnet": ResNetEncoder,
    "attentive_pool": AttentivePoolEncoder,
    "linear": LinearEncoder,
    "batchnorm": BatchNormEncoder,
    "mlp": MLPEncoder,
    "chunk_mlp": ChunkMLPEncoder,
    "reshape": ReshapeEncoder,
    "pixel_decoder": PixelDecoderEncoder,
    "smolvlm": SmolVLMEncoder,
}
