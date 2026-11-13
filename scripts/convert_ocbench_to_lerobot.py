"""Convert OCBench visual shards (`NNN.npz` + `NNN-pixels.npy`) into a LeRobot v3 dataset,
written directly (no LeRobotDataset writer, no per-episode statistics) with HEVC from NVENC.

One LeRobot row per control step at the env's 50 Hz. OCBench stores a camera frame only
every `observation_interval` (K) steps, so each row's cameras hold the most recent stored
frame, which is what the ocbench eval feeds a policy between renders (`pixel_interval`).
A model that wants only fresh frames anchors its samples every K rows
(`subsample_streams: {"observation.images.*": {every: K}}`); episodes start on a stored frame.

Columns: `action` (7), `qpos` (21), `qvel` (20) -- the simulator state before each action,
named like the ocbench backend's so the same configs read either -- and one video per camera
(`observation.images.<front|side|wrist>`). The observation after an episode's last action
has no action to pair with and is dropped. Only successful episodes are kept unless --all.

Each source shard becomes one data file and one video file per camera, written by a pool of
workers in parallel. Each stored frame is piped to ffmpeg once and repeated on the GPU (see
GpuHoldEncoder), so the CPU and PCIe carry 1/K of the frames and NVENC sets the pace. `meta/stats.json` is left empty: compute statistics separately.

    .venv/bin/python scripts/convert_ocbench_to_lerobot.py \\
        --src ~/.cache/ocbench/datasets/visual/visual-block-single-task1-v0 \\
        --root ~/data/lerobot/ocbench_block_single_visual \\
        --repo_id local/ocbench_block_single_visual --workers 12
"""

import argparse
import json
import subprocess
import sys
import time
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis.datasets.ocbench import episode_table, observation_rows, pixels_path  # noqa: E402

TASK = "move the cube to the target"
FPS = 50
STATE_DIMS = {"action": 7, "qpos": 21, "qvel": 20}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="directory of OCBench NNN.npz shards")
    p.add_argument("--root", required=True, help="new LeRobot dataset directory")
    p.add_argument("--repo_id", required=True)
    p.add_argument("--cameras", default="front,side,wrist", help="names of the pixel axis, in order")
    p.add_argument("--shard_start", type=int, default=0)
    p.add_argument("--shard_end", type=int, default=None)
    p.add_argument("--all", action="store_true", help="keep failed episodes too")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--vcodec", default="hevc_nvenc", help="hevc_nvenc (GPU) or libx265/libx264 (CPU)")
    p.add_argument("--qp", type=int, default=30, help="constant quantizer (NVENC) / crf (CPU)")
    p.add_argument("--gop", type=int, default=25, help="keyframe interval in frames")
    p.add_argument("--preset", default="p1", help="NVENC preset, p1 fastest .. p7 best")
    return p.parse_args()


def plan(shards, keep_all):
    """Per shard: kept (episode, length) pairs, with global episode and row offsets."""
    out, ep_offset, row_offset = [], 0, 0
    for i, shard in enumerate(shards):
        lengths, successes, K = episode_table(shard)
        kept = [(e, int(L)) for e, (L, ok) in enumerate(zip(lengths, successes, strict=True))
                if ok or keep_all]
        out.append({"file": i, "shard": str(shard), "K": int(K), "episodes": kept,
                    "ep_offset": ep_offset, "row_offset": row_offset})
        ep_offset += len(kept)
        row_offset += sum(L for _, L in kept)
    return out


class GpuHoldEncoder:
    """One camera's video: each stored frame crosses the pipe and the PCIe bus once, as a
    variable-rate frame whose duration is its hold, and ffmpeg's `fps` filter repeats it on
    the GPU after `hwupload_cuda`, so NVENC sees a constant 50 fps stream without the CPU
    touching the copies. A sentinel frame after the end makes the filter emit the last
    frame's full hold; `-frames:v` trims it."""

    def __init__(self, path, total, size, vcodec, qp, gop, preset):
        import av

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "nut", "-i", "pipe:0"]
        if vcodec.endswith("nvenc"):
            cmd[5:5] = ["-init_hw_device", "cuda=cu:0", "-filter_hw_device", "cu"]
            cmd += ["-vf", f"format=nv12,hwupload_cuda,fps={FPS}", "-c:v", vcodec,
                    "-preset", preset, "-rc", "constqp", "-qp", str(qp), "-bf", "0"]
        else:
            cmd += ["-vf", f"fps={FPS}", "-pix_fmt", "yuv420p", "-c:v", vcodec, "-crf", str(qp)]
        cmd += ["-g", str(gop), "-frames:v", str(total), "-fps_mode", "cfr", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.container = av.open(self.proc.stdin, "w", format="nut")
        self.stream = self.container.add_stream("rawvideo", rate=FPS)
        self.stream.width, self.stream.height = size
        self.stream.pix_fmt = "rgb24"
        self.stream.codec_context.time_base = Fraction(1, FPS)
        self.pts, self.last = 0, None

    def hold(self, image, steps):
        import av

        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        frame.pts = self.pts
        for packet in self.stream.encode(frame):
            packet.duration = steps
            self.container.mux(packet)
        self.pts += steps
        self.last = image

    def close(self):
        if self.last is not None:
            self.hold(self.last, 1)
        for packet in self.stream.encode(None):
            self.container.mux(packet)
        self.container.close()
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg exited {self.proc.returncode}")


def convert_shard(job):
    import pyarrow as pa
    import pyarrow.parquet as pq

    a = job["args"]
    shard, K, cameras = Path(job["shard"]), job["K"], a["cameras"]
    root = Path(a["root"])
    lengths, _, _ = episode_table(shard)
    rows = observation_rows(lengths, K)
    step0 = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    row0 = np.concatenate([[0], np.cumsum(rows)[:-1]])
    with np.load(shard) as data:
        cols = {"action": data["actions"], "qpos": data["qpos"], "qvel": data["qvel"]}
    pixels = np.load(pixels_path(shard), mmap_mode="r")
    size = (pixels.shape[-2], pixels.shape[-3])
    total = sum(L for _, L in job["episodes"])

    encoders = []
    for c in cameras:
        path = root / "videos" / f"observation.images.{c}" / "chunk-000" / f"file-{job['file']:03d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        encoders.append(GpuHoldEncoder(path, total, size, a["vcodec"], a["qp"], a["gop"], a["preset"]))

    pts, table, meta = 0, {k: [] for k in STATE_DIMS}, []
    frame_index, episode_index = [], []
    for j, (ep, L) in enumerate(job["episodes"]):
        s, r = step0[ep], row0[ep]
        for k in STATE_DIMS:
            table[k].append(cols[k][s: s + L])
        frame_index.append(np.arange(L))
        episode_index.append(np.full(L, job["ep_offset"] + j))
        frames = np.asarray(pixels[r: r + (L + K - 1) // K])
        for f in range(frames.shape[0]):
            steps = min(K, L - f * K)
            for i, enc in enumerate(encoders):
                enc.hold(frames[f, i], steps)
        meta.append({"episode_index": job["ep_offset"] + j, "length": L,
                     "from_timestamp": pts / FPS, "to_timestamp": (pts + L) / FPS})
        pts += L
    for enc in encoders:
        enc.close()

    n = pts
    index = job["row_offset"] + np.arange(n)
    fields = {k: pa.FixedSizeListArray.from_arrays(
        pa.array(np.concatenate(table[k]).astype(np.float32).reshape(-1)), STATE_DIMS[k])
        for k in STATE_DIMS}
    fi = np.concatenate(frame_index)
    data = pa.table({
        **fields,
        "timestamp": pa.array((fi / FPS).astype(np.float32)),
        "frame_index": pa.array(fi.astype(np.int64)),
        "episode_index": pa.array(np.concatenate(episode_index).astype(np.int64)),
        "index": pa.array(index.astype(np.int64)),
        "task_index": pa.array(np.zeros(n, np.int64)),
    })
    path = root / "data" / "chunk-000" / f"file-{job['file']:03d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data, path)
    for m in meta:
        m["file"] = job["file"]
    return meta


def write_meta(args, cameras, size, episodes):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(args.root)
    rows, cursor = [], 0
    for m in episodes:
        row = {"episode_index": m["episode_index"], "tasks": [TASK], "length": m["length"],
               "data/chunk_index": 0, "data/file_index": m["file"],
               "dataset_from_index": cursor, "dataset_to_index": cursor + m["length"]}
        for c in cameras:
            key = f"videos/observation.images.{c}"
            row.update({f"{key}/chunk_index": 0, f"{key}/file_index": m["file"],
                        f"{key}/from_timestamp": m["from_timestamp"],
                        f"{key}/to_timestamp": m["to_timestamp"]})
        row.update({"meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0})
        rows.append(row)
        cursor += m["length"]
    path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name="task")).to_parquet(
        root / "meta" / "tasks.parquet")

    video_info = {"video.height": size[1], "video.width": size[0], "video.codec": "hevc",
                  "video.pix_fmt": "yuv420p", "video.fps": FPS, "video.channels": 3,
                  "has_audio": False, "video.g": args.gop, "video.crf": args.qp,
                  "video.preset": args.preset, "video.fast_decode": 0,
                  "video.video_backend": "pyav", "video.extra_options": {}, "is_depth_map": False}
    features = {k: {"dtype": "float32", "shape": [d], "names": [f"{k}.{i}" for i in range(d)]}
                for k, d in STATE_DIMS.items()}
    for c in cameras:
        features[f"observation.images.{c}"] = {
            "dtype": "video", "shape": [size[1], size[0], 3],
            "names": ["height", "width", "channels"], "info": dict(video_info)}
    features.update({
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        **{k: {"dtype": "int64", "shape": [1], "names": None}
           for k in ("frame_index", "episode_index", "index", "task_index")},
    })
    info = {"codebase_version": "v3.0", "fps": FPS, "features": features,
            "total_episodes": len(episodes), "total_frames": cursor, "total_tasks": 1,
            "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 200,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "robot_type": "ocbench_ur5e", "splits": {"train": f"0:{len(episodes)}"}}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    (root / "meta" / "stats.json").write_text("{}")


def main():
    args = parse_args()
    shards = sorted(p for p in Path(args.src).expanduser().glob("*.npz")
                    if not p.name.endswith("-val.npz"))[args.shard_start: args.shard_end]
    if not shards:
        raise SystemExit(f"no shards in {args.src}")
    root = Path(args.root)
    if root.exists():
        raise SystemExit(f"{root} exists; pick a new --root")
    cameras = args.cameras.split(",")
    first = np.load(pixels_path(shards[0]), mmap_mode="r")
    if first.shape[1] != len(cameras):
        raise SystemExit(f"shards hold {first.shape[1]} cameras, --cameras names {len(cameras)}")
    size = (first.shape[-2], first.shape[-3])

    jobs = plan(shards, args.all)
    common = {"root": str(root), "cameras": cameras, "vcodec": args.vcodec, "qp": args.qp,
              "gop": args.gop, "preset": args.preset}
    for j in jobs:
        j["args"] = common
    total = sum(len(j["episodes"]) for j in jobs)
    print(f"{len(shards)} shards, {total} episodes, "
          f"{sum(L for j in jobs for _, L in j['episodes'])} rows, {args.workers} workers, "
          f"{args.vcodec}", flush=True)
    started, done, episodes = time.time(), 0, []
    with Pool(args.workers) as pool:
        for meta in pool.imap_unordered(convert_shard, jobs):
            episodes += meta
            done += len(meta)
            el = time.time() - started
            print(f"{done}/{total} episodes, {done / el:.1f} ep/s, "
                  f"eta {(total - done) * el / done / 60:.1f} min", flush=True)
    episodes.sort(key=lambda m: m["episode_index"])
    write_meta(args, cameras, size, episodes)
    print(f"done: {total} episodes in {time.time() - started:.0f}s -> {root}")


if __name__ == "__main__":
    main()
