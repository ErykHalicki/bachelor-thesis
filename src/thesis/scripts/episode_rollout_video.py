"""Roll a WAM pixel-decoder over a whole episode and write side-by-side mp4s.

For each decoded stream, every video frame is [ground truth | encoded GT | predicted]
at the model's latent rate. Default is AUTOREGRESSIVE: after the first window the
video/state context comes from the model's own predictions (clean actions stay ground
truth), episode start to episode end. --teacher-forcing re-encodes the real context
every window instead.

    # fetch the decoder from wandb (needs WANDB_API_KEY); the dataset auto-downloads
    # from the HF hub if the checkpoint's root is not on this machine (--root overrides)
    python src/thesis/scripts/episode_rollout_video.py halicki/thesis/ple9jevk --episode 0

    # or a local checkpoint, teacher-forced
    python src/thesis/scripts/episode_rollout_video.py ckpt_ple9jevk.pt --teacher-forcing

Outputs <out>/<stream>_ep<N>_<ar|tf>.mp4 at --fps (default 2.5, the latent rate).
"""

import argparse
import os
from fractions import Fraction

import numpy as np
import torch


def load_decoder(ref, device):
    from thesis.algorithms import build_algorithm
    from thesis.utils.checkpoint import load_checkpoint
    from thesis.utils.ckpt_utils import (config_from_checkpoint,
                                        download_latest_checkpoint, split_alias)

    if os.path.exists(ref):
        path = ref
    else:
        run, alias = split_alias(ref)
        path = download_latest_checkpoint(run, os.path.expanduser("~/.cache/thesis_ckpts"),
                                          alias=alias)
    stored = config_from_checkpoint(path)
    model = build_algorithm(stored.algorithm)
    step = load_checkpoint(model, None, path)
    if not hasattr(model, "reconstruct_ar"):
        raise SystemExit(f"{ref} is not a wam_decoder checkpoint")
    print(f"loaded {ref} @ step {step}")
    return model.to(device).eval(), stored


def episode_batches(dataset, episode_pos, stride, max_windows):
    """One (B=1) batch per decision point of the chosen episode, `stride` raw frames
    apart, episode start to end. `episode_pos` indexes the episodes present in the
    split, in order."""
    from torch.utils.data import default_collate

    src = dataset
    while not hasattr(src, "_index") and hasattr(src, "dataset"):
        src = src.dataset
    episodes = sorted({e[2] for e in src._index})
    if episode_pos >= len(episodes):
        raise SystemExit(f"episode {episode_pos} out of range; split has {len(episodes)}")
    ep = episodes[episode_pos]
    by_frame = {e[0]: i for i, e in enumerate(src._index) if e[2] == ep}
    frames = sorted(by_frame)
    picks, f = [], frames[0]
    while f in by_frame:
        picks.append(by_frame[f])
        f += stride
    if max_windows:
        picks = picks[:max_windows]
    print(f"episode {ep}: {len(picks)} windows ({len(picks) * stride / 30:.1f}s)")
    return [default_collate([dataset[i]]) for i in picks]


def write_mp4(path, frames, fps, labels):
    """frames: list of (H, W, 3) uint8. 2.5 fps via an exact rational rate."""
    import av
    from PIL import Image, ImageDraw

    rate = Fraction(fps).limit_denominator(1000)
    container = av.open(path, "w")
    stream = container.add_stream("libx264", rate=rate)
    h, w = frames[0].shape[:2]
    header = 26
    stream.width, stream.height = w, h + header
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}
    n_cols = len(labels)
    for arr in frames:
        img = Image.new("RGB", (w, h + header), "white")
        img.paste(Image.fromarray(arr), (0, header))
        d = ImageDraw.Draw(img)
        for c, lab in enumerate(labels):
            d.text((c * (w // n_cols) + 8, 5), lab, fill="black")
        for packet in stream.encode(av.VideoFrame.from_image(img)):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ckpt", help="wandb run (entity/project/id[:alias]) or local .pt")
    p.add_argument("--teacher-forcing", action="store_true",
                   help="re-encode real context every window instead of feeding "
                        "predictions back")
    p.add_argument("--episode", type=int, default=0,
                   help="episode position within the split (default first)")
    p.add_argument("--split", default="val")
    p.add_argument("--root", default=None, help="local dataset root override")
    p.add_argument("--out", default="rollout_videos")
    p.add_argument("--fps", type=float, default=2.5)
    p.add_argument("--max-windows", type=int, default=0, help="cap windows (0 = all)")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"))
    model, stored = load_decoder(args.ckpt, device)
    if args.root:
        stored.dataset.root = os.path.expanduser(args.root)
    elif stored.dataset.get("root") and not os.path.exists(str(stored.dataset.root)):
        # the training machine's path; null it so lerobot resolves the repo_id via
        # HF_LEROBOT_HOME / the hub and downloads on first use
        print(f"dataset root {stored.dataset.root} not found locally; "
              f"fetching {stored.dataset.repo_id} from the HF hub")
        stored.dataset.root = None

    from thesis.datasets import build_dataset
    dataset = build_dataset(stored.dataset, norm_stats_override=model.norm_stats,
                            augment=False, split=args.split)
    batches = episode_batches(dataset, args.episode, model.ar_window_stride(),
                              args.max_windows)
    batches = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
               for b in batches]

    with torch.no_grad():
        torch.manual_seed(0)
        if args.teacher_forcing:
            per_window = [model.reconstruct(b) for b in batches]
            merged = {}
            for name in per_window[0]:
                merged[name] = {
                    lab: torch.cat([w[name][lab] for w in per_window], dim=1)
                    for lab in per_window[0][name]
                }
        else:
            merged = model.reconstruct_ar(batches)

    os.makedirs(args.out, exist_ok=True)
    labels = ["ground truth", "encoded", "predicted"]
    keys = ["frame", "decoded latent", "decoded prediction"]
    mode = "tf" if args.teacher_forcing else "ar"
    gap = 4
    for name, columns in merged.items():
        steps = columns[keys[0]].shape[1]
        frames = []
        for t in range(steps):
            cells = [columns[k][0, t].permute(1, 2, 0).cpu().numpy() for k in keys
                     if k in columns]
            spacer = np.full((cells[0].shape[0], gap, 3), 255, dtype=np.uint8)
            row = []
            for c in cells:
                row += [c, spacer]
            frames.append(np.concatenate(row[:-1], axis=1))
        write_mp4(os.path.join(args.out, f"{name}_ep{args.episode}_{mode}.mp4"),
                  frames, args.fps, labels[: len(cells)])


if __name__ == "__main__":
    main()
