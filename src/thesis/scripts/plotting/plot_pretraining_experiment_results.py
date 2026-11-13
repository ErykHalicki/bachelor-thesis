#!/usr/bin/env python3
"""Best-checkpoint success rates: 8-task pretrain + fine-tune vs training from scratch.

    python src/thesis/scripts/plotting/plot_pretraining_experiment_results.py
    python src/thesis/scripts/plotting/plot_pretraining_experiment_results.py --save results.png

Where the numbers come from (all hardcoded, cross-checked against W&B project `halicki/thesis`
on 2026-10-23):

- Pretrain + fine-tune: the "best" column of `docs/pretrain8_finetune_results.md`, i.e. the
  better of the 12.5k and 25k evals of a 25k-step fine-tune, 200 EMA-weight episodes each.
  UWM-frozen fine-tunes start from pretrain `qbkz8azv` (W&B runs 7blol3rs block-single,
  wswu7hwy hanoi, ebarcrdu switch-3x3, 96it3m1p block-double; `best/success_rate`).
  attnpool fine-tunes start from pretrain `mxavewkg` and ran on Sockeye with offline W&B,
  so their numbers come from the Sockeye logs listed in that doc.
- From scratch: block-single-task1 only, the best `eval/success_rate` over training,
  500 episodes per eval. UWM-frozen is `nvi41jur` (abl_uwm_frozen, 68.0 at step 125k of
  212.5k); attnpool / no world model is `a5061ikx` (abl_attnpool, 87.6 at step 62.5k; the
  run failed at 112.5k of 212.5k).

The two sources use different episode counts (500 vs 200), and best-checkpoint selection
uses the eval episodes themselves, so every bar is somewhat optimistic.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

UWM, ATT = "#2a78d6", "#eb6834"

TASKS = ["block-single", "hanoi-single", "switch-3x3", "block-double"]
BEST_FT = {"UWM-frozen": [77.5, 74, 7, 52.5], "attnpool": [76, 72, 28.5, 62]}
BEST_SCRATCH = {"UWM-frozen": 68, "attnpool": 87.6}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", help="write the figure here instead of opening a window")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(TASKS))
    w = 0.2

    bars = [
        ("UWM-frozen, from scratch", UWM, "//", -1.5, {0: BEST_SCRATCH["UWM-frozen"]}),
        ("attnpool, from scratch", ATT, "//", -0.5, {0: BEST_SCRATCH["attnpool"]}),
        ("UWM-frozen, pretrain8 + fine-tune", UWM, None, 0.5, dict(enumerate(BEST_FT["UWM-frozen"]))),
        ("attnpool, pretrain8 + fine-tune", ATT, None, 1.5, dict(enumerate(BEST_FT["attnpool"]))),
    ]
    for label, c, hatch, off, vals in bars:
        xs = [x[i] + off * w for i in vals]
        ys = list(vals.values())
        kw = dict(facecolor="white", edgecolor=c, hatch=hatch) if hatch else dict(color=c)
        ax.bar(xs, ys, w * 0.92, label=label, **kw)
        for xi, v in zip(xs, ys):
            ax.text(xi, v + 1, f"{v:g}", ha="center", fontsize=8)

    ax.set_xticks(x, TASKS)
    ax.set_ylabel("best-checkpoint success rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
