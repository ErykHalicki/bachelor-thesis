"""Re-encode LeRobot v3.0 datasets' videos for training: resize, short-GOP h264.

    python -m thesis.datasets.mixture.reencode --raw ~/data/oxe/raw --out ~/data/oxe/224 taco_play toto
    python -m thesis.datasets.mixture.reencode --fit 320x240 --raw ~/data/oxe/raw --out ~/data/oxe/fit320x240 ...

`--size N` squashes every camera to N x N. `--fit WxH` keeps each camera's aspect ratio
and shrinks it to fit inside W x H, never enlarging: a 640x480 camera becomes 320x240, a
256x256 one 240x240, and an 84x84 one stays 84x84 (dimensions rounded down to even, as
yuv420p needs).

The sources are left untouched. Each dataset's `meta/` and `data/` are copied, every video
is rewritten, and `meta/info.json` in the copy is updated to the new codec and shape. Frame
counts and timestamps are preserved (`-fps_mode passthrough`), so frame `i` of a file is
still frame `i`, which is what `LeRobotV3Source` indexes by. Finished files are skipped on
a re-run.

It also repairs the episode table (`repair_episode_table`). In the lerobot/* Open-X v3.0
conversions, whenever a video rolled over to a new file, the episode that crossed the size
limit was written at the start of the new file but recorded in the old one, with a
timestamp past that file's end. Read as-is those episodes point at frames that do not
exist. Five of the eight datasets first tried had this on every multi-file camera.

Pipeline choices, from the eryk-pc benchmarks (2026-11-01 and 2026-11-11, V100):
  - CPU decode + CPU scale feeding NVENC, one ffmpeg process per file. GPU decode and
    `scale_cuda` were both slower, and `scale_cuda` lost about 1.2 dB.
  - The source AV1 decode is the bottleneck, not NVENC: decode alone runs at 4,300 fps on
    a 320x240 file and the full pipeline at 3,500 fps, while the NVENC preset (p1 vs p4)
    and the scaler (bicubic vs fast_bilinear) change nothing. So all files of all datasets
    share one pool that keeps every core decoding, with two decode threads per process.
  - Splitting one file into parallel segments reached 6,200 fps (1.8x) on a lone file, but
    a pool that already has a file per core gains nothing from it, so it is not done here.
  - `h264_nvenc` refuses to open at `-g 1` on this ffmpeg (6.1) and driver, which is why
    an earlier test wrote it off. From `-g 2` up it works, 3.5x faster than `libx264 fast`.
  - h264 is decoded on CPU at train time: GPU decode of h264 through torchcodec was
    unusably slow, and CPU workers kept pace with a training loop on the same GPU.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def target_shape(shape, size=None, fit=None):
    """A video feature's (height, width) after resizing: `size` squashes to a square, `fit`
    (width, height) shrinks into that box keeping the aspect ratio, never enlarging."""
    if fit is None:
        return size, size
    h, w = int(shape[0]), int(shape[1])
    k = min(1.0, fit[0] / w, fit[1] / h)
    return max(2, int(h * k) // 2 * 2), max(2, int(w * k) // 2 * 2)


def encode_cmd(src, dst, shape, gop, cq, encoder):
    h, w = shape
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-threads", "2", "-i", str(src),
           "-vf", f"scale={w}:{h}:flags=bicubic", "-c:v", encoder]
    if encoder.endswith("_nvenc"):
        cmd += ["-preset", "p4", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"]
    else:
        cmd += ["-preset", "fast", "-crf", str(cq), "-keyint_min", str(gop)]
    return cmd + ["-bf", "0", "-g", str(gop), "-pix_fmt", "yuv420p", "-fps_mode",
                  "passthrough", "-an", "-f", "mp4", str(dst)]


def count_frames(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True).stdout
    return int(out.strip().split(",")[0])


def reencode_file(src, dst, shape, gop, cq, encoder):
    """Returns (frames in the target, whether it was encoded now)."""
    if dst.exists():
        return count_frames(dst), False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial")
    subprocess.run(encode_cmd(src, tmp, shape, gop, cq, encoder), check=True)
    n_src, n_dst = count_frames(src), count_frames(tmp)
    if n_src != n_dst:
        tmp.unlink()
        raise RuntimeError(f"{src}: {n_src} frames in, {n_dst} out")
    tmp.rename(dst)
    return n_dst, True


def repair_episode_table(root, frame_counts):
    """Point every episode at the file and frame its video really starts at.

    `frame_counts` maps a video path relative to `root` to its true frame count. An episode
    recorded as starting at or past the end of its file is moved to the following file,
    with the overshoot as its offset. The result must tile every file exactly; anything
    else is a different kind of damage and raises.
    """
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    tables = [pq.read_table(f) for f in files]
    table = pa.concat_tables(tables)
    lengths = table["length"].to_numpy()
    moved = 0
    for key in keys:
        col = {c: table[f"videos/{key}/{c}"].to_numpy(zero_copy_only=False).copy()
               for c in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")}
        paths = sorted(p for p in frame_counts if p.startswith(f"videos/{key}/"))
        position = {p: i for i, p in enumerate(paths)}
        covered = dict.fromkeys(paths, 0)
        for e in range(len(lengths)):
            path = info["video_path"].format(video_key=key, chunk_index=int(col["chunk_index"][e]),
                                             file_index=int(col["file_index"][e]))
            start = int(round(col["from_timestamp"][e] * fps))
            while start >= frame_counts[path]:
                start -= frame_counts[path]
                path = paths[position[path] + 1]
            if start + lengths[e] > frame_counts[path]:
                raise RuntimeError(f"{root.name}: episode {e} of {key} straddles the end of {path}")
            chunk, file = (int(part.split("-")[1].split(".")[0]) for part in path.split("/")[-2:])
            moved += file != col["file_index"][e] or chunk != col["chunk_index"][e]
            col["chunk_index"][e], col["file_index"][e] = chunk, file
            col["from_timestamp"][e] = start / fps
            col["to_timestamp"][e] = (start + lengths[e]) / fps
            covered[path] += int(lengths[e])
        if covered != {p: frame_counts[p] for p in paths}:
            raise RuntimeError(f"{root.name}: episodes do not tile the files of {key}")
        for c, values in col.items():
            name = f"videos/{key}/{c}"
            table = table.set_column(table.column_names.index(name), name,
                                     pa.array(values, table.schema.field(name).type))
    offset = 0
    for f, original in zip(files, tables):
        pq.write_table(table.slice(offset, len(original)), f)
        offset += len(original)
    return moved


def update_info(dst, shapes):
    path = dst / "meta" / "info.json"
    info = json.loads(path.read_text())
    features = {}
    for key, feat in info["features"].items():
        if feat["dtype"] == "video":
            if key not in shapes:
                continue
            h, w = shapes[key]
            feat["shape"] = [h, w, 3]
            video = feat.get("info") or feat.get("video_info") or {}
            video.update({"video.codec": "h264", "video.height": h, "video.width": w,
                          "video.pix_fmt": "yuv420p"})
        features[key] = feat
    info["features"] = features
    path.write_text(json.dumps(info, indent=4))


def prepare(src, dst, size=None, fit=None):
    """Copy the tables, rewrite info.json, and return (source, target, (h, w)) per video."""
    info = json.loads((src / "meta" / "info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise SystemExit(f"{src} is {info['codebase_version']}, expected v3.0")
    keep = [k for k, f in info["features"].items() if f["dtype"] == "video"
            and not (f.get("info") or f.get("video_info") or {}).get("video.is_depth_map")]
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("meta", "data"):
        shutil.copytree(src / sub, dst / sub, dirs_exist_ok=True)
    # info.json's shape is height, width, channels (taco_play's static camera: 150, 200, 3)
    shapes = {k: target_shape(info["features"][k]["shape"], size, fit) for k in keep}
    update_info(dst, shapes)
    return [(f, dst / f.relative_to(src), shapes[key])
            for key in keep for f in sorted((src / "videos" / key).rglob("*.mp4"))]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("datasets", nargs="+", help="directory names under --raw")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--fit", default=None, help="WxH box to shrink into, keeping aspect")
    p.add_argument("--gop", type=int, default=2)
    p.add_argument("--cq", type=int, default=20, help="NVENC constant quality / x264 CRF")
    p.add_argument("--encoder", default="h264_nvenc")
    p.add_argument("--jobs", type=int, default=len(os.sched_getaffinity(0)))
    args = p.parse_args()
    raw, out = args.raw.expanduser(), args.out.expanduser()

    fit = tuple(int(v) for v in args.fit.lower().split("x")) if args.fit else None
    pairs = [pair for name in args.datasets
             for pair in prepare(raw / name, out / name, args.size, fit)]
    pairs.sort(key=lambda pair: -pair[0].stat().st_size)
    start = time.time()
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(
            lambda pair: reencode_file(*pair, args.gop, args.cq, args.encoder), pairs))
    took = time.time() - start
    frames = sum(n for n, encoded in results if encoded)
    print(f"{len(args.datasets)} datasets, {len(pairs)} files: {frames} frames encoded in "
          f"{took:.0f}s ({frames / max(took, 1e-9):.0f} fps)")
    for name in args.datasets:
        root = out / name
        counts = {str(dst.relative_to(root)): n for (_, dst, _), (n, _) in zip(pairs, results)
                  if dst.is_relative_to(root)}
        moved = repair_episode_table(root, counts)
        size_gb = sum(f.stat().st_size for f in (root / "videos").rglob("*.mp4")) / 1e9
        print(f"  {name}: {size_gb:.2f} GB, {moved} episode videos re-pointed to the next file")


if __name__ == "__main__":
    main()
