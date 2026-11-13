"""Convert the LeWM / H-JEPA Push-T expert HDF5 (`pusht_expert_{train,val}.h5`) into a LeRobot
v3 dataset with the camera stored inline as lossless PNG parquet images, written directly (no
LeRobotDataset writer). Inline images give true random access, which the sparse frame windows
of a world model (one frame every `frameskip` rows) need; a video would decode a GOP per frame.

One row per env step at 10 Hz. Columns: `action` (2), `proprio` (4: agent xy, agent velocity),
`state` (7: agent xy, block xy, block angle, agent velocity) and `observation.image` (224 px).
Both files hold the same keys; --val appends the val file's episodes after the train file's,
and their range is recorded in `meta/info.json` `splits.val`.

    .venv/bin/python scripts/convert_pusht_to_lerobot.py \\
        --src /mnt/hdd/pusht/pusht_expert_train.h5 --val /mnt/hdd/pusht/pusht_expert_val.h5 \\
        --root /mnt/hdd/lerobot/pusht_expert --repo_id local/pusht_expert --workers 10
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TASK = "push the T block onto the target"
FPS = 10
STATE_DIMS = {"action": 2, "proprio": 4, "state": 7}
IMAGE_KEY = "observation.image"
NAMES = {
    "action": ["dx", "dy"],
    "proprio": ["agent_x", "agent_y", "agent_vx", "agent_vy"],
    "state": ["agent_x", "agent_y", "block_x", "block_y", "block_angle", "agent_vx", "agent_vy"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="pusht_expert_train.h5")
    p.add_argument("--val", default=None, help="pusht_expert_val.h5, appended after --src")
    p.add_argument("--root", required=True, help="new LeRobot dataset directory")
    p.add_argument("--repo_id", required=True)
    p.add_argument("--rows_per_file", type=int, default=16000)
    p.add_argument("--max_episodes", type=int, default=None, help="first N of --src only")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def episode_table(path, limit=None):
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(path, "r") as f:
        lengths = f["ep_len"][:].astype(np.int64)
        offsets = f["ep_offset"][:].astype(np.int64)
    if limit is not None:
        lengths, offsets = lengths[:limit], offsets[:limit]
    return lengths, offsets


def plan(sources, rows_per_file):
    """Files of whole episodes, ~rows_per_file rows each, with global episode/row offsets."""
    jobs, ep, row = [], 0, 0
    for path, lengths, offsets in sources:
        current = None
        for L, off in zip(lengths, offsets, strict=True):
            if current is None or current["rows"] >= rows_per_file:
                current = {"file": len(jobs), "src": str(path), "episodes": [],
                           "ep_offset": ep, "row_offset": row, "rows": 0}
                jobs.append(current)
            current["episodes"].append((int(off), int(L)))
            current["rows"] += int(L)
            ep += 1
            row += int(L)
    return jobs


def features_dict(size):
    feats = {k: {"dtype": "float32", "shape": (d,), "names": NAMES[k]} for k, d in STATE_DIMS.items()}
    feats[IMAGE_KEY] = {"dtype": "image", "shape": (size, size, 3),
                        "names": ["height", "width", "channels"]}
    feats.update({
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        **{k: {"dtype": "int64", "shape": (1,), "names": None}
           for k in ("frame_index", "episode_index", "index", "task_index")},
    })
    return feats


def convert_file(job):
    import cv2
    import datasets
    import h5py
    import hdf5plugin  # noqa: F401
    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.io_utils import write_table_one_row_group_per_episode

    root = Path(job["root"])
    cols = {k: [] for k in STATE_DIMS}
    images, frame_index, episode_index, meta = [], [], [], []
    with h5py.File(job["src"], "r") as f:
        for j, (off, L) in enumerate(job["episodes"]):
            for k in STATE_DIMS:
                cols[k].append(f[k][off: off + L].astype(np.float32))
            for im in f["pixels"][off: off + L]:
                ok, buf = cv2.imencode(".png", np.ascontiguousarray(im[..., ::-1]),
                                       [cv2.IMWRITE_PNG_COMPRESSION, 1])
                assert ok
                images.append({"bytes": buf.tobytes(), "path": None})
            frame_index.append(np.arange(L))
            episode_index.append(np.full(L, job["ep_offset"] + j))
            meta.append({"episode_index": job["ep_offset"] + j, "length": L, "file": job["file"]})
        size = int(f["pixels"].shape[1])

    fi = np.concatenate(frame_index)
    n = len(fi)
    data = {
        **{k: np.concatenate(v).tolist() for k, v in cols.items()},
        IMAGE_KEY: images,
        "timestamp": (fi / FPS).astype(np.float32).tolist(),
        "frame_index": fi.astype(np.int64).tolist(),
        "episode_index": np.concatenate(episode_index).astype(np.int64).tolist(),
        "index": (job["row_offset"] + np.arange(n)).astype(np.int64).tolist(),
        "task_index": np.zeros(n, np.int64).tolist(),
    }
    ds = datasets.Dataset.from_dict(data, features=get_hf_features_from_features(features_dict(size)))
    path = root / "data" / "chunk-000" / f"file-{job['file']:03d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_table_one_row_group_per_episode(ds.with_format("arrow")[:], path)
    return meta


def write_meta(root, size, episodes, splits):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows, cursor = [], 0
    for m in episodes:
        rows.append({"episode_index": m["episode_index"], "tasks": [TASK], "length": m["length"],
                     "data/chunk_index": 0, "data/file_index": m["file"],
                     "dataset_from_index": cursor, "dataset_to_index": cursor + m["length"],
                     "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0})
        cursor += m["length"]
    path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name="task")).to_parquet(
        root / "meta" / "tasks.parquet")

    features = {k: {**v, "shape": list(v["shape"])} for k, v in features_dict(size).items()}
    info = {"codebase_version": "v3.0", "fps": FPS, "features": features,
            "total_episodes": len(episodes), "total_frames": cursor, "total_tasks": 1,
            "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 200,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": None, "robot_type": "pusht", "splits": splits}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    (root / "meta" / "stats.json").write_text("{}")


def main():
    args = parse_args()
    root = Path(args.root)
    if root.exists():
        raise SystemExit(f"{root} exists; pick a new --root")
    sources = [(args.src, *episode_table(args.src, args.max_episodes))]
    if args.val:
        sources.append((args.val, *episode_table(args.val)))
    n_train = len(sources[0][1])
    total = sum(len(s[1]) for s in sources)
    splits = {"train": f"0:{n_train}"}
    if args.val:
        splits["val"] = f"{n_train}:{total}"

    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(args.src, "r") as f:
        size = int(f["pixels"].shape[1])

    jobs = plan(sources, args.rows_per_file)
    for j in jobs:
        j["root"] = str(root)
    rows = sum(int(s[1].sum()) for s in sources)
    print(f"{total} episodes ({n_train} train), {rows} rows, {len(jobs)} files, "
          f"{args.workers} workers", flush=True)
    started, done, episodes = time.time(), 0, []
    with Pool(args.workers) as pool:
        for meta in pool.imap_unordered(convert_file, jobs):
            episodes += meta
            done += len(meta)
            el = time.time() - started
            print(f"{done}/{total} episodes, {done / el:.1f} ep/s, "
                  f"eta {(total - done) * el / done / 60:.1f} min", flush=True)
    episodes.sort(key=lambda m: m["episode_index"])
    write_meta(root, size, episodes, splits)
    print(f"done: {total} episodes in {time.time() - started:.0f}s -> {root}")


if __name__ == "__main__":
    sys.exit(main())
