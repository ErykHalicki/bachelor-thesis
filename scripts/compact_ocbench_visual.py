"""Drop the failed episodes from shards written by scripts/collect_ocbench_visual.py, in place.

    python scripts/compact_ocbench_visual.py <task dir> [--max_successes 500]

A task whose scripted policy succeeds rarely (switch-*: ~12%) leaves shards that are mostly
frames of failures the dataset backend never reads (`success_only`). Each shard is rewritten
with its successful episodes only, up to --max_successes over the directory in filename
order; shards past that are deleted. Run it only on a task whose collection has finished.
"""

import argparse
import os
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=Path)
parser.add_argument("--max_successes", type=int, default=None)
args = parser.parse_args()

total = 0
for path in sorted(p for p in args.dir.glob("shard-*.npz")):
    pixels = path.with_name(path.stem + "-pixels.npy")
    with np.load(path) as data:
        cols = {k: data[k] for k in data.files}
    interval = int(cols["observation_interval"])
    ends = np.flatnonzero(cols["terminals"]) + 1
    starts = np.r_[0, ends[:-1]]
    lengths = ends - starts
    ob_rows = (lengths + interval - 1) // interval + 1
    ob_starts = np.r_[0, np.cumsum(ob_rows)[:-1]]
    success = np.array([cols["rewards"][s:e].max() == 1 for s, e in zip(starts, ends)])
    keep = np.flatnonzero(success)
    if args.max_successes is not None:
        keep = keep[: max(0, args.max_successes - total)]
    if not len(keep):
        path.unlink()
        pixels.unlink()
        print(f"{path.name}: deleted")
        continue
    total += len(keep)
    if len(keep) == len(lengths):
        print(f"{path.name}: {len(keep)} episodes, already compact")
        continue
    steps = np.concatenate([np.arange(starts[e], ends[e]) for e in keep])
    rows = np.concatenate([np.arange(ob_starts[e], ob_starts[e] + ob_rows[e]) for e in keep])
    frames = np.load(pixels, mmap_mode="r")
    assert len(frames) == ob_rows.sum(), f"{pixels} does not match {path.name}"
    frames = frames[rows]
    out = {k: (v if v.ndim == 0 else v[steps]) for k, v in cols.items()}
    tmp_npz, tmp_pix = path.with_suffix(".tmp.npz"), pixels.with_suffix(".tmp.npy")
    np.save(tmp_pix, frames)
    np.savez(tmp_npz, **out)
    os.replace(tmp_pix, pixels)
    os.replace(tmp_npz, path)
    print(f"{path.name}: kept {len(keep)}/{len(lengths)} episodes, {total} so far")
