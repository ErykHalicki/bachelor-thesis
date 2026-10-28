"""How aligned are the action and dynamics gradients on the shared weights?

    python scripts/grad_alignment.py <ckpt.pt> [--batches 16] [--batch-size 32]

Per training batch (train split, train mode, no augmentation), one forward pass builds the
loss terms -- so both objectives see the same flow-time and noise draws -- and each is
backpropagated on its own:

    action    the `action` flow term, at its configured weight
    dynamics  every other weighted term (the future-frame flow terms), at their weights

Only parameters both gradients touch are compared (the trunk and the context encoders);
heads that only one objective reaches carry no conflict. Reported per module group and
overall:

    cos/batch     mean +- std over batches of cos(g_action, g_dynamics)
    cos/mean      cosine of the two gradients averaged over all batches -- the alignment
                  of the expected gradients, with most of the per-batch noise averaged out
    norm ratio    |g_dynamics| / |g_action| of the averaged gradients: who dominates
    action self   cos(g_action on one half of the batches, g_action on the other half): how
                  well the action gradient agrees with ITSELF, the yardstick for cos/mean
                  (two independent noisy estimates of one gradient score below 1 too)

cos > 0 helps, ~0 means the objectives pull on unrelated directions, < 0 means conflict.
No optimizer step is taken; the checkpoint is never modified.
"""

import argparse
import re
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from thesis.algorithms import build_algorithm
from thesis.datasets import build_dataset
from thesis.utils.checkpoint import load_checkpoint
from thesis.utils.ckpt_utils import config_from_checkpoint


def group_of(name):
    """encoders.<name>.* -> encoders.<name>; trunk blocks bucketed by depth; rest by head."""
    if name.startswith("encoders."):
        return ".".join(name.split(".")[:2])
    m = re.match(r"predictor\.blocks\.(\d+)\.", name)
    if m:
        return f"trunk.block{int(m.group(1)):02d}"
    return "trunk.other"


def flat(grads, names):
    return torch.cat([grads[n].flatten() for n in names])


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    device = torch.device("cuda")

    cfg = config_from_checkpoint(args.checkpoint)
    model = build_algorithm(cfg.algorithm)
    step = load_checkpoint(model, None, args.checkpoint)
    model.to(device).train()
    dcfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    ds = build_dataset(dcfg, norm_stats_override=model.norm_stats, augment=False, split="train")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=args.workers,
    )
    torch.manual_seed(args.seed)

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    terms = [t for t in model.loss_terms if t.weight != 0.0]
    action_terms = [t for t in terms if t.label == "action"]
    dyn_terms = [t for t in terms if t.label != "action"]
    assert action_terms and dyn_terms, f"need an action term and others, got {[t.label for t in terms]}"
    print(f"step {step}; action: {[t.label for t in action_terms]}; "
          f"dynamics: {[(t.label, t.weight) for t in dyn_terms]}")

    def grads_of(loss, retain):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=retain)
        return {n: p.grad.detach().clone() for n, p in params.items() if p.grad is not None}

    shared, per_batch, sums = None, [], {"a": None, "d": None, "a0": None, "a1": None}
    for b, batch in enumerate(loader):
        if b >= args.batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        ctx = model._build_context(batch)
        la = sum(t.weight * t(ctx) for t in action_terms)
        ld = sum(t.weight * t(ctx) for t in dyn_terms)
        ga, gd = grads_of(la, True), grads_of(ld, False)
        if shared is None:
            shared = [n for n in params if n in ga and n in gd]
            groups = {}
            for n in shared:
                groups.setdefault(group_of(n), []).append(n)
            only_a = sum(params[n].numel() for n in ga if n not in gd)
            only_d = sum(params[n].numel() for n in gd if n not in ga)
            print(f"shared params {sum(params[n].numel() for n in shared) / 1e6:.1f}M; "
                  f"action-only {only_a / 1e6:.2f}M; dynamics-only {only_d / 1e6:.2f}M")
        row = {"loss_a": la.item(), "loss_d": ld.item()}
        del ctx, la, ld
        va, vd = flat(ga, shared), flat(gd, shared)
        row["all"] = cos(va, vd)
        for g, names in groups.items():
            row[g] = cos(flat(ga, names), flat(gd, names))
        per_batch.append(row)
        del ga, gd
        va, vd = va.cpu(), vd.cpu()
        half = "a0" if b % 2 == 0 else "a1"
        for key, v in (("a", va), ("d", vd), (half, va)):
            sums[key] = v.clone() if sums[key] is None else sums[key] + v
        print(f"  batch {b:2d}  cos {row['all']:+.3f}  L_action {row['loss_a']:.4f}  L_dyn {row['loss_d']:.4f}")

    import statistics as st

    def col(k):
        xs = [r[k] for r in per_batch]
        return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)

    offsets, i = {}, 0
    for n in shared:
        k = params[n].numel()
        offsets[n] = slice(i, i + k)
        i += k

    def sub(v, names):
        return torch.cat([v[offsets[n]] for n in names])

    print(f"\n{'group':22s} {'params':>8s}  {'cos/batch':>14s}  {'cos/mean':>8s}  {'|gd|/|ga|':>9s}  {'action self':>11s}")
    for g, names in [("ALL SHARED", shared)] + sorted(groups.items()):
        m, s = col("all" if g == "ALL SHARED" else g)
        A, D = sub(sums["a"], names), sub(sums["d"], names)
        A0, A1 = sub(sums["a0"], names), sub(sums["a1"], names)
        n = sum(params[x].numel() for x in names)
        print(f"{g:22s} {n / 1e6:7.2f}M  {m:+.3f} +- {s:.3f}  {cos(A, D):+8.3f}  "
              f"{(D.norm() / A.norm()).item():9.2f}  {cos(A0, A1):+11.3f}")


if __name__ == "__main__":
    sys.exit(main())
