"""Split a reconstruction-eval panel into one labeled image per rollout.

The reconstruction eval logs one grid per stream: a row per held-out sample, and per
future step the columns [real frame | decoded latent | decoded prediction]. This
rearranges it: one PNG per sample, rows = ground truth / encoded / predicted, columns
= the future steps. Pillow only; wandb only if you fetch by run id.

    # from a wandb run (latest panel per stream), e.g. on any machine with the API key
    python split_recon_panels.py --run halicki/thesis/didja11h

    # or from panel PNGs you already have
    python split_recon_panels.py --panel scene_8001_abc.png wrist_8002_def.png

Cell geometry defaults to the shipped 256px cells with 2px gaps; override --cell/--gap
if the run used another image_size.
"""

import argparse
import os
import re
import tempfile

from PIL import Image, ImageDraw, ImageFont


def find_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def fetch_panels(run_path, tmpdir):
    """Download the latest panel per stream from a wandb run -> [(stream, step, file)]."""
    import wandb

    api = wandb.Api()
    run = api.run(run_path)
    latest = {}
    for f in run.files():
        m = re.match(r".*media/images/.*?([A-Za-z0-9]+)_(\d+)_[0-9a-f]+\.png$", f.name)
        if not m:
            continue
        stream, step = m.group(1), int(m.group(2))
        if step >= latest.get(stream, (-1, None))[0]:
            latest[stream] = (step, f)
    out = []
    for stream, (step, f) in sorted(latest.items()):
        f.download(root=tmpdir, replace=True)
        out.append((stream, step, os.path.join(tmpdir, f.name)))
    return out


def split(path, stream, step, args):
    src = Image.open(path)
    cell, gap = args.cell, args.gap
    cols = (src.width + gap) // (cell + gap)
    rows = (src.height + gap) // (cell + gap)
    variants = len(args.labels)
    steps = cols // variants
    assert steps * variants == cols, (
        f"{path}: {cols} columns do not divide into {variants} variants; "
        f"check --cell/--gap/--labels"
    )
    margin, pad, header = args.margin, 6, 30
    font, small = find_font(20), find_font(16)
    for i in range(rows):
        W = margin + steps * cell + (steps - 1) * pad
        H = header + variants * cell + (variants - 1) * pad
        out = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(out)
        for t in range(steps):
            d.text((margin + t * (cell + pad) + cell // 2 - 25, 5), f"t = {t + 1}",
                   fill="black", font=small)
        for v in range(variants):
            y0 = header + v * (cell + pad)
            d.text((10, y0 + cell // 2 - 10), args.labels[v], fill="black", font=font)
            for t in range(steps):
                xs, ys = (t * variants + v) * (cell + gap), i * (cell + gap)
                out.paste(src.crop((xs, ys, xs + cell, ys + cell)),
                          (margin + t * (cell + pad), y0))
        dest = os.path.join(args.out, f"{stream}_rollout{i + 1}_step{step}.png")
        out.save(dest)
        print(dest)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", help="wandb run (entity/project/id) to pull panels from")
    p.add_argument("--panel", nargs="*", default=[], help="local panel PNG(s) instead")
    p.add_argument("--out", default="panels", help="output directory")
    p.add_argument("--cell", type=int, default=256)
    p.add_argument("--gap", type=int, default=2)
    p.add_argument("--margin", type=int, default=185)
    p.add_argument("--labels", nargs="*",
                   default=["ground truth", "encoded", "predicted"])
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = []
    if args.run:
        jobs += fetch_panels(args.run, tempfile.mkdtemp(prefix="panels_"))
    for f in args.panel:
        m = re.match(r"([A-Za-z0-9]+)_(\d+)", os.path.basename(f))
        jobs.append((m.group(1) if m else "stream", int(m.group(2)) if m else 0, f))
    if not jobs:
        p.error("pass --run or --panel")
    for stream, step, path in jobs:
        split(path, stream, step, args)


if __name__ == "__main__":
    main()
