"""A pixel decoder probe: what a frame's single latent still says about the frame.

The decoder is trained on detached encoder latents only, so it shapes nothing in the world
model. Decoding a *predicted* latent with it, and scoring that against the true future
frame, turns prediction quality into PSNR. It is a weak proxy: a decode is the mean of
every image consistent with the latent, so it rewards getting the static scene right.
Always read it against the copy baseline (decode of the last context frame's latent),
which is what a model that predicts no motion at all would score.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.algorithms.pixel_decoder import PixelDecoder


def psnr(pred, target):
    """Per-image PSNR in dB for (..., 3, H, W) images in [0, 1]."""
    mse = (pred.clamp(0, 1) - target).square().flatten(-3).mean(-1)
    return -10 * torch.log10(mse.clamp_min(1e-10))


class DecoderProbe(nn.Module):
    def __init__(self, latent_dim, cfg):
        super().__init__()
        self.decoder = PixelDecoder(latent_dim, img_size=cfg.img_size, patch_size=cfg.patch_size,
                                    hidden_dim=cfg.hidden_dim, depth=cfg.depth,
                                    num_heads=cfg.heads)

    def forward(self, z):
        """(..., latent) -> (..., 3, H, W) images in [0, 1] units."""
        flat = z.reshape(-1, 1, z.shape[-1])
        return self.decoder(flat).unflatten(0, z.shape[:-1])

    def loss(self, z, frames, max_images):
        """MSE on a random subset of (latent, frame) pairs. z (N, T, latent) detached by the
        caller's contract is enforced here; frames uint8 (N, T, 3, H, W)."""
        z, frames = z.detach().flatten(0, 1), frames.flatten(0, 1)
        if len(z) > max_images:
            pick = torch.randperm(len(z), device=z.device)[:max_images]
            z, frames = z[pick], frames[pick]
        target = frames.float() / 255
        if target.shape[-2:] != self.decoder.img_size:
            target = F.interpolate(target, size=self.decoder.img_size, mode="area")
        recon = self(z).float()
        return F.mse_loss(recon, target), psnr(recon.detach(), target).mean()
