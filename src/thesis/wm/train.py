"""Train the mixed-embodiment world model and its decoder probe.

    python -m thesis.wm.train --config configs/wm/oxe_small.yaml [key=value ...]

Writes to `out_dir`: `metrics.jsonl` (one line per log or validation), `last.pt`, and
`panels/step_*.png` (truth / decoded latent / decoded rollout). `resume=true` continues
from `out_dir/last.pt`.
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from thesis.datasets.mixture.batch import collate
from thesis.datasets.mixture.clock import Clock
from thesis.datasets.mixture.stream import (CameraPolicy, FixedWindows, MixtureStream, build_members,
                                mixture_spec, mixture_weights)

from .model import WorldModel
from .probe import DecoderProbe, psnr


def lr_scale(step, cfg):
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.final_lr_scale + (1 - cfg.final_lr_scale) * 0.5 * (1 + math.cos(math.pi * progress))


def to_unit(frames, size):
    x = frames.flatten(0, 1).float() / 255
    if x.shape[-2:] != tuple(size):
        x = F.interpolate(x, size=tuple(size), mode="area")
    return x.unflatten(0, frames.shape[:2])


@torch.no_grad()
def validate(model, probe, loader, names, context, device, panel_path=None):
    """Per embodiment: one-step latent error against the copy-last-latent error, and PSNR of
    decoded latents against the true frames for reconstruction, the autoregressive rollout
    and the copy baseline, at the first and last rollout step."""
    model.eval(), probe.eval()
    sums, counts, panels = defaultdict(lambda: defaultdict(float)), defaultdict(int), {}
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(batch)
            z = out["z"]
            rolled = model.rollout(z, batch, context).float()
            target = to_unit(batch.views[:, context:], probe.decoder.img_size)
            images = {
                "recon": probe(z[:, context:]).float(),
                "rollout": probe(rolled).float(),
                "copy": probe(z[:, context - 1:context]).float().expand_as(target),
            }
        per_view = {
            "pred_mse": (out["pred"] - z[:, 1:]).square().mean((1, 2)),
            "copy_mse": (z[:, :-1] - z[:, 1:]).square().mean((1, 2)),
            "rollout_mse_last": (rolled[:, -1] - z[:, -1]).square().mean(1),
            "copy_mse_last": (z[:, context - 1] - z[:, -1]).square().mean(1),
        }
        for name, img in images.items():
            score = psnr(img, target)
            per_view[f"psnr_{name}"] = score.mean(1)
            if name != "recon":
                per_view[f"psnr_{name}_first"] = score[:, 0]
                per_view[f"psnr_{name}_last"] = score[:, -1]
        for e in batch.view_emb.unique().tolist():
            rows = batch.view_emb == e
            counts[e] += int(rows.sum())
            for key, value in per_view.items():
                sums[e][key] += float(value[rows].sum())
            if e not in panels:
                i = int(rows.nonzero()[0])
                panels[e] = torch.cat([target[i], images["recon"][i], images["rollout"][i]])
    metrics = {names[e]: {k: v / counts[e] for k, v in sums[e].items()} for e in sorted(sums)}
    total = sum(counts.values())
    metrics["all"] = {k: sum(sums[e][k] for e in sums) / total for k in next(iter(sums.values()))}
    if panel_path is not None:
        from torchvision.utils import save_image

        panel_path.parent.mkdir(parents=True, exist_ok=True)
        steps = target.shape[1]
        grid = torch.cat([panels[e].clamp(0, 1) for e in sorted(panels)])
        save_image(grid, panel_path, nrow=steps, padding=1)
    model.train(), probe.train()
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    out_dir = Path(cfg.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    clock = Clock(cfg.clock.f_wm, cfg.clock.steps)
    policy = CameraPolicy(cfg.cameras.roles, cfg.cameras.max_views)
    names = list(cfg.mixture.members)
    members = build_members(cfg.data_root, cfg.embodiments_dir, names, clock, "train")
    val_members = build_members(cfg.data_root, cfg.embodiments_dir, names, clock, "val")
    weights = mixture_weights(members, cfg.mixture.temperature, cfg.mixture.overrides)
    spec = mixture_spec(members)
    for m, w in zip(members, weights):
        print(f"{m.embodiment.name:20s} {m.fps:4.0f} fps  K={clock.substeps(m.fps):2d}  "
              f"D={m.manifest['action_dim']}  {m.manifest['hours']:6.2f} h  "
              f"{m.num_anchors:8d} anchors  p={w:.3f}  "
              f"({m.dropped} episodes shorter than a window)", flush=True)

    model = WorldModel(spec, cfg.model).to(device)
    probe = DecoderProbe(cfg.model.latent_dim, cfg.probe).to(device)
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    groups = [
        {"params": encoder_params, "base_lr": cfg.train.encoder_lr},
        {"params": head_params, "base_lr": cfg.train.lr},
        {"params": list(probe.parameters()), "base_lr": cfg.train.probe_lr},
    ]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]], lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    print(f"trainable: encoder {sum(p.numel() for p in encoder_params) / 1e6:.1f}M, "
          f"heads {sum(p.numel() for p in head_params) / 1e6:.1f}M, "
          f"probe {sum(p.numel() for p in probe.parameters()) / 1e6:.1f}M", flush=True)

    step = 0
    last = out_dir / "last.pt"
    if cfg.resume:
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        probe.load_state_dict(state["probe"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step = state["step"]
        print(f"resumed from {last} at step {step}", flush=True)

    stream = MixtureStream(members, weights, policy, seed=cfg.seed + step)
    loader = DataLoader(stream, batch_size=cfg.train.batch_size, collate_fn=collate,
                        num_workers=cfg.train.workers, pin_memory=True, prefetch_factor=4,
                        persistent_workers=True)
    val_loader = DataLoader(
        FixedWindows(val_members, policy, cfg.val.windows_per_member, cfg.seed),
        batch_size=cfg.train.batch_size, collate_fn=collate, num_workers=cfg.train.workers)

    def save():
        tmp = last.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "probe": probe.state_dict(),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "spec": spec, "config": OmegaConf.to_container(cfg)}, tmp)
        os.replace(tmp, last)

    def log(record):
        with open(out_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    model.train(), probe.train()
    window, t_last, t_data = defaultdict(float), time.time(), 0.0
    batches = iter(loader)
    while step < cfg.train.steps:
        t0 = time.time()
        batch = next(batches).to(device)
        t_data += time.time() - t0
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * lr_scale(step, cfg.train)

        with torch.autocast("cuda", dtype=torch.float16):
            out = model(batch)
            probe_loss, probe_psnr = probe.loss(out["z"], batch.views, cfg.probe.images_per_step)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(out["loss"] + probe_loss).backward()
        scaler.unscale_(optimizer)
        # a non-finite fp16 gradient makes GradScaler skip the step; it is counted below
        grad_norm = torch.nn.utils.clip_grad_norm_(encoder_params + head_params, cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step += 1

        z = out["z"].detach()
        stats = {
            "loss": out["loss"], "pred_loss": out["pred_loss"], "sigreg": out["sigreg"],
            "copy_loss": F.mse_loss(z[:, :-1], z[:, 1:]),
            "z_std": z.flatten(0, 1).std(0).mean(),
            "grad_norm": grad_norm if grad_norm.isfinite() else 0.0,
            "skipped_steps": 0.0 if grad_norm.isfinite() else 1.0,
            "probe_psnr": probe_psnr, "views": batch.views.shape[0],
        }
        for key, value in stats.items():
            window[key] += float(value)
        if not math.isfinite(window["loss"]):
            raise RuntimeError(f"loss is not finite at step {step}")

        if step % cfg.train.log_every == 0:
            n = cfg.train.log_every
            took = time.time() - t_last
            record = {"step": step, **{k: v / n for k, v in window.items()},
                      "lr": cfg.train.lr * lr_scale(step - 1, cfg.train), "sec_per_step": took / n,
                      "data_wait": t_data / took}
            log(record)
            print("  ".join(f"{k} {v:.4g}" for k, v in record.items()), flush=True)
            window, t_last, t_data = defaultdict(float), time.time(), 0.0

        if step % cfg.val.every == 0 or step == cfg.train.steps:
            metrics = validate(model, probe, val_loader, names, cfg.val.context, device,
                               out_dir / "panels" / f"step_{step:07d}.png")
            log({"step": step, "val": metrics})
            for name, m in metrics.items():
                print(f"val {name:20s} " + "  ".join(f"{k} {v:.4g}" for k, v in m.items()),
                      flush=True)
            save()
            t_last = time.time()


if __name__ == "__main__":
    main()
