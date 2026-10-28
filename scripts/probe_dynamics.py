"""Latent-health and dynamics probe for a checkpoint with UWM-style future-frame streams.

    python scripts/probe_dynamics.py <ckpt.pt> [<ckpt.pt> ...] [--batches 16] [--samples 4]

Runs on the held-out episodes (the training config's own `split="val"`), no environment.
Every number is in the space the model's future streams are scored in -- the target encoder's
output (for abl_uwm_frozen: a frozen ResNet18 7x7 grid, standardized by BN) -- so the
model and every baseline below are judged on the same scale. Lower MSE is better.

Dynamics: is the predicted next frame (+25 steps, the chunk's end) any good?
    pred/marginal     futures integrated with the action stream held at noise: the model's
                      own guess at what happens, not told which actions follow (policy mode)
    pred/forward      action stream held clean at the true chunk: forward dynamics
    pred/fwd_shuffled same, with another sample's chunk -- if this matches pred/forward, the
                      model's video branch ignores the action
  each scored as `single` (mean MSE of one flow sample, which a sharp but wrong sampler pays
  for) and `mean` (MSE of the K-sample mean, closer to the conditional mean).
  against
    base/zero         the target space's mean (BN-standardized: ~1 per element)
    base/batch_mean   leave-one-out mean over the batch, per token position: "average frame"
    base/persist      the current frame, encoded the same way -- the bar a dynamics model
                      has to clear; most of a 0.5 s future IS the present
    base/t+50         the frame 25 steps too late, same episode
    base/t-25         the frame 25 steps too early, same episode
    base/other        another sample's true future -- right distribution, wrong scene
  plus, per prediction mode
    skill             1 - MSE / MSE(persist): >0 means it beats copying the present
    delta_cos         cosine between the predicted change (pred - now) and the true change
                      (future - now), per token: does it move the right way, even if short
    top1              share of samples whose prediction is closer to their own future than to
                      any other sample's in the batch (chance 1/B): discriminative, scale-free

Latent health, per encoded camera stream (the trainable context encoders AND the target):
    eff_rank          participation ratio of the token covariance's spectrum (1 = one
                      direction carries everything, D = isotropic)
    dead_dims         share of dims whose std is below 1% of the median dim's
    temporal          E||z(t) - z(t+25)||^2 / (2 * total variance): 1 if the two frames'
                      latents were independent draws, 0 if the encoder ignores the passage of
                      time. The frozen target's value is the reference point -- a trainable
                      encoder far below it is collapsing time away (abl_uwm sat at ~0.02)

Results print as a table and append as one JSON line per checkpoint to --out.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict

from thesis.algorithms import build_algorithm
from thesis.datasets import build_dataset
from thesis.utils.checkpoint import load_checkpoint
from thesis.utils.ckpt_utils import config_from_checkpoint
from thesis.utils.spec import flow_source_entries, raw_index, spec_fields, split_index

HORIZON = 25           # the future streams' raw offset; frames are stored every 25 steps
PROBE_OFFSETS = (-25, 0, 25, 50)


def image_field(field):
    return field.startswith("observation.images.")


def field_windows(entries):
    out = {}
    for name, spec in entries.items():
        if "index" not in spec:
            continue
        rel, _ = split_index(raw_index(spec))
        for field in spec_fields(name, spec):
            out.setdefault(field, set()).update(rel)
    return {f: sorted(s) for f, s in out.items()}


def load(path, device):
    cfg = config_from_checkpoint(path)
    if cfg is None:
        raise ValueError(f"{path} embeds no config_yaml; it predates config embedding")
    model = build_algorithm(cfg.algorithm)
    step = load_checkpoint(model, None, path)
    return cfg, model.to(device).eval(), step


def future_streams(model):
    """Flow predict streams reading one camera field through an encoder: name -> field."""
    out = {}
    for name, spec in model.predict_spec.items():
        fields = spec_fields(name, spec)
        if (spec.get("type") == "flow" and spec.get("encoder") and len(fields) == 1
                and image_field(fields[0])):
            out[name] = fields[0]
    return out


def context_streams(model):
    out = {}
    for name, spec in model.conditioning.items():
        fields = spec_fields(name, spec)
        if spec.get("encoder") and len(fields) == 1 and image_field(fields[0]):
            out[name] = fields[0]
    return out


def build_probe_dataset(cfg, model):
    """The held-out split with every camera field widened to PROBE_OFFSETS, through a
    spec-less extra conditioning entry per field (the loader unions windows per field)."""
    dcfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    cams = sorted({f for f in {**future_streams(model), **context_streams(model)}.values()})
    with open_dict(dcfg):
        dcfg.augment = None
        for i, field in enumerate(cams):
            dcfg.conditioning[f"_probe_{i}"] = {
                "from": field,
                "index": f"0..{len(PROBE_OFFSETS) - 1}",
                "raw_index": ", ".join(str(o) for o in PROBE_OFFSETS),
            }
    ds = build_dataset(dcfg, norm_stats_override=getattr(model, "norm_stats", None),
                       augment=False, split="val")
    return ds, cams


def model_batch(batch, model, data_windows):
    """Cut every field back to the window the model was trained on, so its row selectors
    index the same rows they did in training."""
    entries = {**dict(model.conditioning), **dict(model.predict_spec),
               **flow_source_entries(model.predict_spec)}
    out = dict(batch)
    for field, rel in field_windows(entries).items():
        have = data_windows.get(field)
        if have is None or have == rel or field not in batch:
            continue
        rows = [have.index(o) for o in rel]
        out[field] = batch[field][:, rows]
    return out


def frame(batch, field, offset, data_windows):
    i = data_windows[field].index(offset)
    return batch[field][:, i:i + 1]


class Acc:
    def __init__(self):
        self.sum, self.n = {}, {}

    def add(self, key, value, n=1):
        self.sum[key] = self.sum.get(key, 0.0) + float(value) * n
        self.n[key] = self.n.get(key, 0) + n

    def result(self):
        return {k: self.sum[k] / self.n[k] for k in self.sum}


def mse(a, b):
    return (a - b).pow(2).mean().item()


def top1(pred, target):
    """Share of rows whose prediction is nearest (MSE) its own target among the batch's."""
    p, t = pred.flatten(1), target.flatten(1)
    d = torch.cdist(p.float(), t.float())
    return (d.argmin(dim=1) == torch.arange(len(p), device=p.device)).float().mean().item()


def delta_cos(pred, now, target):
    a, b = (pred - now), (target - now)
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


class Health:
    """Streaming covariance / temporal statistics for one latent stream."""

    def __init__(self):
        self.n, self.s, self.ss, self.dt, self.npairs = 0, None, None, 0.0, 0

    def add(self, z0, z1):
        x = z0.reshape(-1, z0.shape[-1]).double()
        if self.s is None:
            self.s = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
            self.ss = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64, device=x.device)
        self.n += len(x)
        self.s += x.sum(0)
        self.ss += x.T @ x
        self.dt += (z0 - z1).double().pow(2).sum(-1).sum().item()
        self.npairs += z0.shape[0] * z0.shape[1]

    def result(self):
        mean = self.s / self.n
        cov = self.ss / self.n - torch.outer(mean, mean)
        ev = torch.linalg.eigvalsh(cov).clamp_min(0)
        std = cov.diagonal().clamp_min(0).sqrt()
        total = cov.diagonal().sum().item()
        return {
            "eff_rank": (ev.sum() ** 2 / ev.pow(2).sum()).item(),
            "dead_dims": (std < 0.01 * std.median()).double().mean().item(),
            "temporal": (self.dt / self.npairs) / (2 * total) if total > 0 else float("nan"),
        }


@torch.no_grad()
def probe(path, args, device):
    cfg, model, step = load(path, device)
    fut, ctx = future_streams(model), context_streams(model)
    ds, cams = build_probe_dataset(cfg, model)
    data_windows = field_windows(ds_cfg_entries(cfg, cams))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=args.workers,
    )
    torch.manual_seed(args.seed)
    acc = Acc()
    health = {}
    action_name = next((n for n, s in model.predict_spec.items()
                        if spec_fields(n, s) == ["action"]), None)

    for b, batch in enumerate(loader):
        if b >= args.batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        obs = model_batch(batch, model, data_windows)
        B = next(v for v in batch.values() if torch.is_tensor(v)).shape[0]
        perm = torch.roll(torch.arange(B, device=device), 1)

        # latent health: every camera stream, now vs +25, plus the targets
        for name, field in ctx.items():
            enc = model.conditioning[name]["encoder"]
            z0 = model.run_encoder(enc, frame(batch, field, 0, data_windows))
            z1 = model.run_encoder(enc, frame(batch, field, HORIZON, data_windows))
            health.setdefault(f"{name}", Health()).add(z0, z1)

        targets, enc_at = {}, {}
        for name, field in fut.items():
            enc = model.predict_spec[name]["encoder"]
            enc_at[name] = {o: model.run_encoder(enc, frame(batch, field, o, data_windows))
                            for o in PROBE_OFFSETS}
            targets[name] = enc_at[name][HORIZON]
            health.setdefault(f"{name}(target enc)", Health()).add(enc_at[name][0], targets[name])
        if not fut:
            continue

        def sample(clamp, action_obs=None):
            o = obs if action_obs is None else {**obs, "action": action_obs}
            draws = [model.predict(o, streams=list(fut), clamp=clamp) for _ in range(args.samples)]
            return {n: torch.stack([d[n] for d in draws]) for n in fut}

        modes = {"marginal": sample({action_name: 0.0} if action_name else {})}
        if action_name:
            modes["forward"] = sample({action_name: 1.0})
            modes["fwd_shuffled"] = sample({action_name: 1.0}, obs["action"][perm])

        for name in fut:
            y, now = targets[name], enc_at[name][0]
            persist = mse(now, y)
            loo = (y.sum(0, keepdim=True) - y) / (B - 1)
            base = {
                "zero": mse(torch.zeros_like(y), y),
                "batch_mean": mse(loo, y),
                "persist": persist,
                "t+50": mse(enc_at[name][50], y),
                "t-25": mse(enc_at[name][-25], y),
                "other": mse(y[perm], y),
            }
            for k, v in base.items():
                acc.add(f"{name}/base/{k}", v, B)
            acc.add(f"{name}/base/persist/top1", top1(now, y), B)
            for mode, by_stream in modes.items():
                draws = by_stream[name]
                single = sum(mse(d, y) for d in draws) / len(draws)
                mean_pred = draws.mean(0)
                acc.add(f"{name}/{mode}/single", single, B)
                acc.add(f"{name}/{mode}/mean", mse(mean_pred, y), B)
                acc.add(f"{name}/{mode}/skill", 1 - mse(mean_pred, y) / persist, B)
                acc.add(f"{name}/{mode}/delta_cos", delta_cos(mean_pred, now, y), B)
                acc.add(f"{name}/{mode}/top1", top1(mean_pred, y), B)
            if "forward" in modes:
                acc.add(f"{name}/action_effect",
                        mse(modes["forward"][name].mean(0), modes["fwd_shuffled"][name].mean(0)), B)

    out = {"checkpoint": str(path), "step": step, **acc.result(),
           **{f"health/{k}/{m}": v for k, h in health.items() for m, v in h.result().items()}}
    return out


def ds_cfg_entries(cfg, cams):
    """The entries the probe dataset was built from: the model's plus the widened probes."""
    entries = {**OmegaConf.to_container(cfg.dataset.conditioning, resolve=True),
               **OmegaConf.to_container(cfg.dataset.predict, resolve=True)}
    entries.update(flow_source_entries(entries))
    for i, field in enumerate(cams):
        entries[f"_probe_{i}"] = {
            "from": field, "index": f"0..{len(PROBE_OFFSETS) - 1}",
            "raw_index": ", ".join(str(o) for o in PROBE_OFFSETS),
        }
    return entries


def report(r):
    print(f"\n== {r['checkpoint']}  step {r['step']}")
    streams = sorted({k.split("/")[0] for k in r if "/" in k and not k.startswith("health/")})
    for s in streams:
        base = {k.split("/", 2)[2]: v for k, v in r.items() if k.startswith(f"{s}/base/")}
        print(f"  {s}")
        print("    baselines  " + "  ".join(f"{k} {v:.4f}" for k, v in base.items()))
        for mode in ("marginal", "forward", "fwd_shuffled"):
            if f"{s}/{mode}/mean" not in r:
                continue
            print(f"    {mode:12s} single {r[f'{s}/{mode}/single']:.4f}  mean {r[f'{s}/{mode}/mean']:.4f}"
                  f"  skill {r[f'{s}/{mode}/skill']:+.3f}  delta_cos {r[f'{s}/{mode}/delta_cos']:+.3f}"
                  f"  top1 {r[f'{s}/{mode}/top1']:.3f}")
        if f"{s}/action_effect" in r:
            print(f"    action_effect (fwd vs shuffled-action MSE) {r[f'{s}/action_effect']:.4f}")
    print("  latent health")
    names = sorted({k.split("/")[1] for k in r if k.startswith("health/")})
    for n in names:
        print(f"    {n:34s} eff_rank {r[f'health/{n}/eff_rank']:7.1f}"
              f"  dead {r[f'health/{n}/dead_dims']:.3f}  temporal {r[f'health/{n}/temporal']:.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--samples", type=int, default=4, help="flow samples per prediction mode")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/probe_dynamics.jsonl"))
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for path in args.checkpoints:
        r = probe(path, args, device)
        report(r)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    sys.exit(main())
