"""Train a pixel decoder on frozen latents: the visualization probe, as its own run.

The spec is the same one every other algorithm reads, with the two sections playing their
narrowest roles: `conditioning:` is the latent being probed (a stream through whatever
encoder chain produced it), and `predict:` is the frame that latent came from, read raw.
Each predict entry names the conditioning stream it decodes (`input:`) and the `encoders:`
entry that decodes it (`decoder:`, a `pixel_decoder`), so one run can probe several streams
-- scene and wrist camera, patch grid and pooled summary -- side by side.

    encoders:
      vjepa:  { type: vjepa2, arch: vit_base, checkpoint: pretrained, frozen: true }
      decode: { type: pixel_decoder, in_dim: 768, tokens_per_step: 196, img_size: 224 }
    conditioning:
      scene_latent: { from: observation.images.zed_left, encoder: vjepa, dim: 768,
                      index: "0", raw_index: "-6, 0", grid: [14, 14] }
    predict:
      scene_pixels: { from: observation.images.zed_left, input: scene_latent,
                      decoder: decode, type: prediction, index: "0" }

Only the decoder is meant to train: an encoder left trainable would move the very
representation being measured, so probe configs freeze the whole chain and `summary()`
reports which encoders are frozen. Freezing also means the chain runs under no_grad
(PredictiveModel.run_encoder), making a probe run roughly as cheap as a forward pass.

The loss is plain MSE against the frame in [0, 1] units, resized to the decoder's output
size -- the same resize the encoder applies to its own input, so target and latent see one
geometry. That makes `loss/<stream>` a per-pixel MSE, which the reconstruction eval turns
straight into PSNR.

`wam_decoder.py` is the other end of the same idea: the same decoders on a frozen world
model, where a PREDICTED latent can be decoded next to the encoded one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictive_model import PredictiveModel

__all__ = ["LatentPixelDecoder", "PixelProbe", "to_uint8"]


def to_uint8(images):
    """Decoded pixels -> displayable uint8, clamping what the linear head overshot."""
    return (images.detach().float().clamp(0.0, 1.0) * 255).round().to(torch.uint8)


class PixelProbe:
    """What the two pixel probes share: turning a stream's raw window into the frames its
    latent steps are at, in the units and geometry the decoder produces.
    """

    def _frames(self, name, batch, size):
        """The frame each index step of stream `name` is at, as float [0, 1] at `size`.

        A stream whose encoder consumes several raw rows per step (a V-JEPA tubelet is two
        frames) is at the LAST of them: that is the moment the latent describes, and the
        one a rollout would be standing at.
        """
        spec = {**self.conditioning, **self.predict_spec}[name]
        raw = self._raw(name, spec, batch)
        per_step = self._raw_per_step[name]
        if per_step > 1:
            raw = raw.unflatten(1, (raw.shape[1] // per_step, per_step))[:, :, -1]
        return self._pixels(raw, size)

    @staticmethod
    def _pixels(frames, size):
        """Raw (B, T, C, H, W) frames -> float [0, 1] at the decoder's output size."""
        x = frames.float()
        if frames.dtype == torch.uint8:
            x = x / 255.0
        if tuple(x.shape[-2:]) != tuple(size):
            x = F.interpolate(
                x.flatten(0, 1), size=tuple(size), mode="bilinear", antialias=True
            ).unflatten(0, x.shape[:2])
        return x

    def _check_decoder(self, name, decoder, stream, spec):
        """A decoder and the stream it reads state the same two numbers in two files (the
        latent width and how many tokens a step is); catch a drift here rather than as a
        matmul shape error thousands of steps into a run.
        """
        dec = self.encoders[decoder]
        dim = int(spec["dim"]) if "dim" in spec else None
        if dim is not None and getattr(dec, "in_dim", dim) != dim:
            raise ValueError(
                f"'{name}' decodes stream '{stream}' ({dim}-dim) with '{decoder}', whose "
                f"in_dim is {dec.in_dim}"
            )
        grid = tuple(spec.get("grid", (1, 1)))
        area = int(grid[0]) * int(grid[1])
        if getattr(dec, "tokens_per_step", area) != area:
            raise ValueError(
                f"'{name}' decodes stream '{stream}', which carries {area} token(s) per "
                f"step (grid {list(grid)}), with '{decoder}' whose tokens_per_step is "
                f"{dec.tokens_per_step}"
            )

    @staticmethod
    def _check_shape(name, decoded, frames):
        if decoded.shape != frames.shape:
            raise ValueError(
                f"'{name}' decodes {tuple(decoded.shape)} but its target window is "
                f"{tuple(frames.shape)}; the decoded stream's index steps and the "
                f"target's must match one for one"
            )

    def summary(self):
        # a probe measures a fixed representation: a trainable encoder here silently
        # changes what the numbers describe
        frozen = sorted(n for n, e in self.encoders.items() if e.frozen)
        trainable = sorted(n for n, e in self.encoders.items() if not e.frozen)
        return {
            **super().summary(),
            "encoders/frozen": ", ".join(frozen) or "none",
            "encoders/trainable": ", ".join(trainable) or "none",
        }


class LatentPixelDecoder(PixelProbe, PredictiveModel):
    """PixelDecoders over the conditioning streams' latents, trained against raw frames."""

    def _build_predictor(self, cfg):
        """There is no predictor: this algorithm is its encoders and the decoders on top.

        The decoders are `encoders:` entries, held by NAME rather than as a second module
        reference, so each one appears in the checkpoint once -- a probe's weights then
        line up key for key with the run it was seeded from.
        """
        self._decoder_of = {}
        for name, spec in cfg.predict.items():
            dec, src = spec.get("decoder"), spec.get("input")
            if dec is None or src is None:
                raise ValueError(
                    f"predict stream '{name}' must name the latent it decodes (`input:`, a "
                    f"conditioning stream) and the `encoders:` entry that decodes it "
                    f"(`decoder:`); got input={src!r} decoder={dec!r}"
                )
            if src not in cfg.conditioning:
                raise ValueError(
                    f"predict stream '{name}' decodes '{src}', which is not a conditioning "
                    f"stream; available: {sorted(cfg.conditioning)}"
                )
            if dec not in self.encoders:
                raise ValueError(f"predict stream '{name}' names unknown decoder '{dec}'")
            self._check_decoder(name, dec, src, cfg.conditioning[src])
            self._decoder_of[name] = dec
        return nn.Identity()

    def _decode(self, name, latents):
        decoder = self.encoders[self._decoder_of[name]]
        return decoder(latents[self.predict_spec[name]["input"]])

    def _build_context(self, batch):
        # the predict streams here are raw frames, so the base's encode-the-targets pass
        # has nothing to do; this also keeps the raw window a decoded stream is scored against
        clean, sources, cond, hidden = self._condition(batch)
        latents = {**clean, **sources, **cond, **hidden}
        pred, target = {}, {}
        for name in self.predict_spec:
            # float32 on both sides: under bf16 autocast an MSE over pixel-scale values
            # loses most of its resolution
            out = self._decode(name, latents).float()
            frames = self._frames(name, batch, out.shape[-2:])
            self._check_shape(name, out, frames)
            pred[name], target[name] = out, frames
        return {"latent": latents, "pred": pred, "target": target}

    def predict(self, obs):
        self.eval()
        with torch.no_grad():
            clean, sources, cond, hidden = self._condition(obs)
            latents = {**clean, **sources, **cond, **hidden}
            return {name: self._decode(name, latents) for name in self.predict_spec}

    @torch.no_grad()
    def reconstruct(self, batch):
        """{stream: {label: uint8 (B, steps, 3, H, W)}} -- the columns of one panel row,
        real frame first. What the reconstruction eval renders; train/eval mode is
        restored, so this is safe to call mid-training.
        """
        was_training = self.training
        self.eval()
        try:
            clean, sources, cond, hidden = self._condition(batch)
            latents = {**clean, **sources, **cond, **hidden}
            out = {}
            for name in self.predict_spec:
                decoded = self._decode(name, latents)
                out[name] = {
                    "frame": to_uint8(self._frames(name, batch, decoded.shape[-2:])),
                    "decoded latent": to_uint8(decoded),
                }
        finally:
            self.train(was_training)
        return out
