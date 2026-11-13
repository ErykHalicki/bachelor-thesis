"""Figures for a thesis.wm.train run, from its metrics.jsonl.

    python scripts/plot_wm_run.py <run_dir> <out_dir>
    python scripts/plot_wm_run.py --compare <out_dir> unfrozen=<run_dir> frozen=<run_dir>

Needs matplotlib, which the training venv does not carry.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def smooth(y, k=5):
    if len(y) < k:
        return np.asarray(y)
    pad = np.concatenate([np.full(k // 2, y[0]), y, np.full(k // 2, y[-1])])
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def compare(out, runs):
    """Held-out curves of several runs on shared axes, one colour per run."""
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(11, 2.8))
    for (label, run), color in zip(runs, (BLUE, ORANGE, AQUA)):
        val = [json.loads(line) for line in open(Path(run) / "metrics.jsonl")]
        val = [r for r in val if "val" in r]
        step = [r["step"] for r in val]
        get = lambda k: np.array([r["val"]["all"][k] for r in val])
        style = dict(color=color, marker="o", ms=3, label=label)
        ax[0].plot(step, get("pred_mse") / get("copy_mse"), **style)
        ax[1].plot(step, get("psnr_rollout_last") - get("psnr_copy_last"), **style)
        ax[2].plot(step, get("psnr_recon"), **style)
    ax[0].axhline(1, color=MUTED, lw=0.8, ls=":")
    ax[1].axhline(0, color=MUTED, lw=0.8, ls=":")
    ax[0].set_title("One-step latent MSE / copy MSE", loc="left")
    ax[1].set_title("Last-step rollout PSNR minus copy (dB)", loc="left")
    ax[2].set_title("PSNR of decoded true latents (dB)", loc="left")
    ax[0].legend()
    for a in ax:
        a.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out / "compare.pdf")


def main():
    if sys.argv[1] == "--compare":
        return compare(Path(sys.argv[2]), [a.split("=", 1) for a in sys.argv[3:]])
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in open(run / "metrics.jsonl")]
    train = [r for r in records if "val" not in r]
    val = [r for r in records if "val" in r]
    step = np.array([r["step"] for r in train])
    col = lambda k: np.array([r[k] for r in train])

    fig, ax = plt.subplots(1, 4, figsize=(11, 2.6))
    ax[0].plot(step, smooth(col("pred_loss")), color=BLUE, label="predictor")
    ax[0].plot(step, smooth(col("copy_loss")), color=ORANGE, label="copy last latent")
    ax[0].set_yscale("log"), ax[0].set_title("Next-latent MSE (train)", loc="left"), ax[0].legend()
    ax[1].plot(step, smooth(col("sigreg")), color=BLUE)
    ax[1].set_yscale("log"), ax[1].set_title("SIGReg statistic", loc="left")
    ax[2].plot(step, col("z_std"), color=BLUE)
    ax[2].axhline(1, color=MUTED, lw=0.8, ls=":")
    ax[2].set_title("Latent std per dimension", loc="left")
    ax[3].plot(step, smooth(col("probe_psnr")), color=AQUA)
    ax[3].set_title("Decoder probe PSNR (train, dB)", loc="left")
    for a in ax:
        a.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out / "train_curves.pdf")

    vstep = np.array([r["step"] for r in val])
    names = [n for n in val[-1]["val"] if n != "all"]
    get = lambda name, k: np.array([r["val"][name][k] for r in val])
    fig, ax = plt.subplots(1, 3, figsize=(11, 2.8))
    ax[0].plot(vstep, get("all", "pred_mse"), color=BLUE, marker="o", ms=3, label="predictor, one step")
    ax[0].plot(vstep, get("all", "copy_mse"), color=ORANGE, marker="o", ms=3, label="copy last latent")
    ax[0].set_yscale("log"), ax[0].set_title("Held-out next-latent MSE", loc="left"), ax[0].legend()
    for a, which, title in ((ax[1], "first", "PSNR, first rollout step (dB)"),
                            (ax[2], "last", "PSNR, last rollout step (dB)")):
        a.plot(vstep, get("all", "psnr_recon"), color=AQUA, marker="o", ms=3, label="decode true latent")
        a.plot(vstep, get("all", f"psnr_rollout_{which}"), color=BLUE, marker="o", ms=3, label="decode rollout")
        a.plot(vstep, get("all", f"psnr_copy_{which}"), color=ORANGE, marker="o", ms=3, label="decode last context latent")
        a.set_title(title, loc="left")
    ax[2].legend()
    for a in ax:
        a.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out / "val_curves.pdf")

    final = val[-1]["val"]
    y = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(11, 0.36 * len(names) + 1.2), sharey=True)
    ratio = [final[n]["pred_mse"] / final[n]["copy_mse"] for n in names]
    ax[0].barh(y, ratio, height=0.6, color=BLUE)
    ax[0].axvline(1, color=MUTED, lw=0.8, ls=":")
    ax[0].set_title("One-step latent MSE / copy MSE (below 1 beats copying)", loc="left")
    for yi, v in zip(y, ratio):
        ax[0].text(v, yi, f" {v:.2f}", va="center", color=INK, fontsize=8)
    gain_first = [final[n]["psnr_rollout_first"] - final[n]["psnr_copy_first"] for n in names]
    gain_last = [final[n]["psnr_rollout_last"] - final[n]["psnr_copy_last"] for n in names]
    ax[1].barh(y - 0.18, gain_first, height=0.32, color=BLUE, label="first rollout step")
    ax[1].barh(y + 0.18, gain_last, height=0.32, color=AQUA, label="last rollout step")
    ax[1].axvline(0, color=MUTED, lw=0.8)
    ax[1].set_title("Rollout PSNR minus copy PSNR (dB)", loc="left"), ax[1].legend()
    ax[0].set_yticks(y, names), ax[0].invert_yaxis()
    for a in ax:
        a.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "per_embodiment.pdf")
    print(json.dumps(final["all"], indent=1))


if __name__ == "__main__":
    main()
