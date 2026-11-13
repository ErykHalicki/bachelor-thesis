"""Convert an Open-X RLDS dataset, read straight from its source, into LeRobot v3.0.

    python -m thesis.datasets.mixture.rlds jaco_play --fps 10 \
        --out ~/data/oxe/rlds --tmp ~/data/oxe/_rlds_tmp

Source: the public bucket gs://gresearch/robotics/<name>/<version>/, read over HTTPS, newest
version that has shards. Shards are downloaded one at a time to `--tmp` (the next one
prefetched while the current one converts) and deleted once converted, so temporary disk
stays at about two shards.

What is kept, decided from the dataset's own features.json:
  - every encoded RGB image under steps/observation -> a video, `observation.images.<key>`,
    shrunk to fit inside `--fit` (default 320x240) keeping its aspect ratio, never enlarged,
    h264 at GOP 2 (as `reencode.py`; frames are sized from the decoded images, not the
    spec, which can be wrong: bc_z declares 640x512 and stores 213x171)
  - every numeric per-step field of at most `--max-dim` values (state, action, joint and
    pose fields, reward, is_first/is_last/is_terminal), as recorded, under its RLDS path
    with `/` -> `.` and the `steps/` prefix dropped (steps/observation/state ->
    observation.state, steps/action/world_vector -> action.world_vector)
  - the first non-empty per-step language field as the episode's task
  - scalar and text episode_metadata fields as `meta.<name>` columns of the episode table
What is dropped, and listed in info.json under `rlds.dropped`: depth images (one channel or
16-bit), raw-tensor images, language embeddings, and any field larger than `--max-dim`
(depth arrays, optical flow, point clouds).

RLDS stores no frame rate, so `--fps` is required: the control frequency from the Open-X
dataset spreadsheet. Episodes keep the source's split, in order (train first), in
info.json `splits` and the episode table's `rlds.split` column.

A run that stops part-way is restarted from scratch for that dataset (`--overwrite`).
"""

import argparse
import io
import json
import os
import shutil
import struct
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

GCS = "https://storage.googleapis.com"
BUCKET = "gresearch"
SPLIT_ORDER = {"train": 0, "val": 1, "validation": 1, "test": 2}


# ---- GCS ---------------------------------------------------------------------------------

def _get(url, rng=None, timeout=120):
    headers = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return r.read()


def _listing(prefix, delimiter=None):
    items, prefixes, token = [], [], None
    while True:
        q = {"prefix": prefix, "fields": "items(name,size),prefixes,nextPageToken",
             "maxResults": 1000}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["pageToken"] = token
        d = json.loads(_get(f"{GCS}/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(q)))
        items += d.get("items", [])
        prefixes += d.get("prefixes", [])
        token = d.get("nextPageToken")
        if not token:
            return items, prefixes


def find_version(name):
    """(version prefix, shard objects) of the newest version of `name` that has shards."""
    _, versions = _listing(f"robotics/{name}/", "/")
    key = lambda p: [int(x) if x.isdigit() else 0 for x in p.rstrip("/").split("/")[-1].split(".")]
    for ver in sorted(versions, key=key, reverse=True):
        items, _ = _listing(ver)
        shards = [it for it in items if ".tfrecord-" in it["name"]]
        if shards:
            return ver, shards
    raise FileNotFoundError(f"gs://{BUCKET}/robotics/{name} has no version with shards")


def shard_split(name):
    """`bridge-train.tfrecord-00003-of-01024` -> 'train'."""
    return name.rsplit("/", 1)[-1].split(".tfrecord-")[0].rsplit("-", 1)[-1]


def download(obj, dst, retries=5):
    """Stream one object to `dst`, resuming with a Range request after a dropped connection."""
    url = f"{GCS}/{BUCKET}/{urllib.parse.quote(obj['name'])}"
    size = int(obj["size"])
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(retries):
        have = tmp.stat().st_size if tmp.exists() else 0
        if have >= size:
            break
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "ab") as f:
                shutil.copyfileobj(r, f, 1 << 22)
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    if tmp.stat().st_size != size:
        raise IOError(f"{obj['name']}: got {tmp.stat().st_size} bytes, expected {size}")
    tmp.rename(dst)
    return dst


# ---- TFRecord and tf.train.Example, without TensorFlow ------------------------------------

def tfrecords(path):
    """Yield the records of a TFRecord file (length-prefixed, CRCs not checked)."""
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if not head:
                return
            if len(head) < 12:
                raise IOError(f"{path}: truncated record header")
            (n,) = struct.unpack("<Q", head[:8])
            rec = f.read(n)
            if len(rec) != n or len(f.read(4)) != 4:
                raise IOError(f"{path}: truncated record")
            yield rec


def _varint(b, i):
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return v, i


def _fields(b):
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
            yield f, wt, v
        elif wt == 2:
            ln, i = _varint(b, i)
            yield f, wt, b[i:i + ln]
            i += ln
        elif wt == 5:
            yield f, wt, b[i:i + 4]
            i += 4
        elif wt == 1:
            yield f, wt, b[i:i + 8]
            i += 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")


def example_features(record):
    """tf.train.Example bytes -> {key: raw Feature bytes}, decoded lazily by `decode`."""
    out = {}
    for f, _, feats in _fields(memoryview(record)):
        if f != 1:
            continue
        for _, _, entry in _fields(feats):
            key = feat = None
            for k, _, v in _fields(entry):
                if k == 1:
                    key = bytes(v).decode()
                elif k == 2:
                    feat = v
            out[key] = feat
    return out


def decode(feat):
    """Feature bytes -> ('bytes', [bytes]) | ('float', float32 array) | ('int', int64 array)."""
    for f, _, v in _fields(feat):
        if f == 1:
            return "bytes", [bytes(x) for ff, _, x in _fields(v) if ff == 1]
        if f == 2:
            parts = [bytes(x) for ff, _, x in _fields(v) if ff == 1]
            return "float", np.frombuffer(b"".join(parts), "<f4").copy()
        if f == 3:
            ints = []
            for ff, w, x in _fields(v):
                if ff != 1:
                    continue
                if w == 2:
                    j = 0
                    while j < len(x):
                        val, j = _varint(x, j)
                        ints.append(val)
                else:
                    ints.append(x)
            a = np.array(ints, dtype=np.uint64).view(np.int64)
            return "int", a
    return "bytes", []


# ---- the dataset's own feature spec --------------------------------------------------------

def spec_leaves(node, path=""):
    """(path, class, dtype, shape, encoding) for every leaf of a features.json tree."""
    if not isinstance(node, dict):
        return
    cls = node.get("pythonClassName", "").split(".")[-1]
    # only a feature node (one with a class) carries its payload under image/tensor/text; a
    # features dict may itself have a camera named "image"
    for k in ("image", "tensor", "text") if cls else ():
        if isinstance(node.get(k), dict):
            t = node[k]
            yield (path.strip("/"), cls, t.get("dtype"), tuple(
                int(x) for x in (t.get("shape") or {}).get("dimensions", [])), t.get("encodingFormat"))
            return
    for k, v in node.items():
        if k in ("featuresDict", "features", "feature", "sequence"):
            yield from spec_leaves(v, path)
        elif isinstance(v, dict) and not (cls and k in ("image", "tensor", "text")):
            yield from spec_leaves(v, path + "/" + k)


def plan(features_json, max_dim):
    """Decide what to keep: {'cameras', 'numeric', 'text', 'meta', 'dropped'}."""
    cameras, numeric, text, meta, dropped = [], [], [], [], []
    for path, cls, dtype, shape, _ in spec_leaves(features_json):
        name = path.split("/")[-1]
        step = path.startswith("steps/")
        if cls == "Image":
            if step and path.startswith("steps/observation/") and len(shape) == 3 \
                    and shape[-1] == 3 and dtype == "uint8" and "depth" not in name.lower():
                cameras.append(path)
            else:
                dropped.append(f"{path} (image {list(shape)} {dtype})")
        elif cls == "Text" or dtype == "string":
            if step and "instruction" in name:
                text.append(path)
            elif not step:
                meta.append(path)
            else:
                dropped.append(f"{path} (text)")
        elif dtype in ("float32", "float64", "int32", "int64", "uint8", "int8", "uint16", "bool"):
            size = int(np.prod(shape)) if shape else 1
            if "embedding" in name:
                dropped.append(f"{path} (language embedding)")
            elif not step:
                (meta if size == 1 else dropped).append(path if size == 1 else f"{path} ({size} values)")
            elif size > max_dim:
                dropped.append(f"{path} ({list(shape)} {dtype})")
            else:
                numeric.append((path, dtype, shape))
        else:
            dropped.append(f"{path} ({cls} {dtype})")
    return {"cameras": cameras, "numeric": numeric, "text": text, "meta": meta, "dropped": dropped}


def column_name(path):
    p = path[len("steps/"):] if path.startswith("steps/") else path
    return p.replace("/", ".")


def camera_name(path):
    return "observation.images." + path[len("steps/observation/"):].replace("/", ".")


def fit_size(w, h, box):
    k = min(1.0, box[0] / w, box[1] / h)
    return max(2, int(w * k) // 2 * 2), max(2, int(h * k) // 2 * 2)


# ---- writers ---------------------------------------------------------------------------------

class VideoWriter:
    """One camera's sequence of mp4 files; an episode never straddles two files."""

    def __init__(self, root, key, fps, encoder, cq, gop, max_mb, chunks_size):
        self.root, self.key, self.fps = root, key, fps
        self.encoder, self.cq, self.gop = encoder, cq, gop
        self.max_bytes, self.chunks_size = max_mb * 1e6, chunks_size
        self.size = None
        self.file_index = -1
        self.proc = None
        self.frames_in_file = 0
        self.path = None
        self.files = []

    def _cmd(self, path):
        w, h = self.size
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-",
               "-c:v", self.encoder]
        if self.encoder.endswith("_nvenc"):
            cmd += ["-preset", "p4", "-rc", "vbr", "-cq", str(self.cq), "-b:v", "0"]
        else:
            cmd += ["-preset", "fast", "-crf", str(self.cq), "-keyint_min", str(self.gop)]
        return cmd + ["-bf", "0", "-g", str(self.gop), "-pix_fmt", "yuv420p", "-an",
                      "-f", "mp4", str(path)]

    def _open(self):
        self.file_index += 1
        c, f = divmod(self.file_index, self.chunks_size)
        self.path = self.root / f"videos/{self.key}/chunk-{c:03d}/file-{f:03d}.mp4"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(self._cmd(self.path), stdin=subprocess.PIPE)
        self.frames_in_file = 0

    def close(self):
        if self.proc is None:
            return
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")
        self.files.append((self.path, self.frames_in_file))
        self.proc = None

    def write_episode(self, frames):
        """uint8 (T, h, w, 3) at this camera's size -> (chunk, file, from_ts, to_ts)."""
        if self.proc is not None and self.path.exists() and \
                self.path.stat().st_size > self.max_bytes:
            self.close()
        if self.proc is None:
            self._open()
        start = self.frames_in_file
        self.proc.stdin.write(np.ascontiguousarray(frames).tobytes())
        self.frames_in_file += len(frames)
        c, f = divmod(self.file_index, self.chunks_size)
        return c, f, start / self.fps, self.frames_in_file / self.fps


class DataWriter:
    """Per-frame parquet files of about `max_mb` each; an episode never straddles two."""

    def __init__(self, root, max_mb, chunks_size):
        self.root, self.max_bytes, self.chunks_size = root, max_mb * 1e6, chunks_size
        self.file_index, self.parts, self.nbytes = -1, [], 0

    def flush(self):
        if not self.parts:
            return
        c, f = divmod(self.file_index, self.chunks_size)
        path = self.root / f"data/chunk-{c:03d}/file-{f:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(self.parts), path)
        self.parts, self.nbytes = [], 0

    def write_episode(self, table):
        if self.file_index < 0 or self.nbytes > self.max_bytes:
            self.flush()
            self.file_index += 1
        self.parts.append(table)
        self.nbytes += table.nbytes
        return divmod(self.file_index, self.chunks_size)


def _array_column(values, dtype):
    a = np.asarray(values)
    if dtype == "bool":
        a = a.astype(bool)
    elif dtype.startswith("float"):
        a = a.astype(np.float32)
    else:
        a = a.astype(np.int64)
    if a.ndim == 1:
        return pa.array(a)
    flat = a.reshape(len(a), -1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat.reshape(-1)), flat.shape[1])


def _feature_entry(dtype, shape, fps):
    d = "bool" if dtype == "bool" else ("float32" if dtype.startswith("float") else "int64")
    return {"dtype": d, "shape": list(shape) or [1], "names": None, "fps": float(fps)}


# ---- conversion --------------------------------------------------------------------------

class Converter:
    def __init__(self, name, out, tmp, fps, fit=(320, 240), encoder="h264_nvenc", cq=20,
                 gop=2, max_dim=256, max_episodes=None, decode_threads=8,
                 video_mb=500, data_mb=100, chunks_size=1000, version=None, shards=None):
        self.name, self.fps, self.fit = name, float(fps), fit
        short = name.removesuffix("_converted_externally_to_rlds")
        self.out, self.tmp = Path(out).expanduser() / short, Path(tmp).expanduser() / short
        self.encoder, self.cq, self.gop = encoder, cq, gop
        self.max_dim, self.max_episodes = max_dim, max_episodes
        self.pool = ThreadPoolExecutor(decode_threads)
        self.video_mb, self.data_mb, self.chunks_size = video_mb, data_mb, chunks_size
        self._version, self._shards = version, shards

    # one camera's frames of one episode, decoded and resized in parallel
    def _frames(self, writer, blobs):
        from PIL import Image

        def one(blob):
            im = Image.open(io.BytesIO(blob)).convert("RGB")
            if writer.size is None:
                writer.size = fit_size(*im.size, self.fit)
            if im.size != writer.size:
                im = im.resize(writer.size, Image.BICUBIC)
            return np.asarray(im)

        if writer.size is None and blobs:
            one(blobs[0])
        return np.stack(list(self.pool.map(one, blobs)))

    def run(self, overwrite=False):
        if self.out.exists():
            if not overwrite:
                raise SystemExit(f"{self.out} exists; pass --overwrite to redo it")
            shutil.rmtree(self.out)
        self.tmp.mkdir(parents=True, exist_ok=True)
        ver, shards = (self._version, self._shards) if self._version else find_version(self.name)
        base = f"{GCS}/{BUCKET}/{ver}"
        features = json.loads(_get(base + "features.json"))
        info_src = json.loads(_get(base + "dataset_info.json"))
        p = plan(features, self.max_dim)
        if not p["cameras"]:
            raise SystemExit(f"{self.name}: no RGB camera in the spec")
        shards = sorted(shards, key=lambda s: (SPLIT_ORDER.get(shard_split(s["name"]), 9), s["name"]))

        videos = {c: VideoWriter(self.out, camera_name(c), self.fps, self.encoder, self.cq,
                                 self.gop, self.video_mb, self.chunks_size) for c in p["cameras"]}
        data = DataWriter(self.out, self.data_mb, self.chunks_size)
        episodes, tasks, splits = [], {}, {}
        index = 0
        t0, nbytes = time.time(), 0

        prefetch = {}

        def fetch(i):
            obj = shards[i]
            dst = self.tmp / obj["name"].rsplit("/", 1)[-1]
            return download(obj, dst)

        dl = ThreadPoolExecutor(1)
        if shards:
            prefetch[0] = dl.submit(fetch, 0)
        done = False
        for si, obj in enumerate(shards):
            path = prefetch.pop(si).result()
            if si + 1 < len(shards) and not done:
                prefetch[si + 1] = dl.submit(fetch, si + 1)
            split = shard_split(obj["name"])
            for ri, rec in enumerate(tfrecords(path)):
                ep = self._episode(rec, p, videos, data, len(episodes), index, tasks)
                if ep is None:
                    continue
                ep.update({"rlds.split": split, "rlds.shard": obj["name"].rsplit("/", 1)[-1],
                           "rlds.record": ri})
                episodes.append(ep)
                splits.setdefault(split, [ep["episode_index"], ep["episode_index"]])[1] = \
                    ep["episode_index"]
                index += ep["length"]
                if self.max_episodes and len(episodes) >= self.max_episodes:
                    done = True
                    break
            nbytes += int(obj["size"])
            path.unlink()
            rate = nbytes / max(time.time() - t0, 1e-9) / 1e6
            print(f"{self.name}: shard {si + 1}/{len(shards)} ({split}) -> {len(episodes)} "
                  f"episodes, {index} frames, {rate:.1f} MB/s", flush=True)
            if done:
                break
        for fut in prefetch.values():
            try:
                fut.result().unlink()
            except Exception:
                pass
        dl.shutdown()
        for w in videos.values():
            w.close()
        data.flush()
        self._check_videos(videos, episodes)
        self._write_meta(p, videos, episodes, tasks, splits, index, ver, info_src)
        shutil.rmtree(self.tmp, ignore_errors=True)
        return self.out

    def _episode(self, rec, p, videos, data, ep_index, start, tasks):
        feats = example_features(rec)
        first = feats.get("steps/is_first")
        if first is None:
            raise ValueError(f"{self.name}: record without steps/is_first")
        T = len(decode(first)[1])
        if T == 0:
            return None
        cols = {}
        for path, dtype, shape in p["numeric"]:
            if path not in feats:
                continue
            kind, vals = decode(feats[path])
            size = int(np.prod(shape)) if shape else 1
            if len(vals) != T * size:
                raise ValueError(f"{self.name}: {path} has {len(vals)} values for {T} steps")
            cols[column_name(path)] = (np.asarray(vals).reshape((T,) + tuple(shape)), dtype)
        task = ""
        for path in p["text"]:
            if path in feats:
                kind, vals = decode(feats[path])
                s = next((v.decode("utf-8", "replace").strip() for v in vals if v.strip()), "")
                if s:
                    task = s
                    break
        task_index = tasks.setdefault(task, len(tasks))

        ep = {"episode_index": ep_index, "length": T, "tasks": [task],
              "dataset_from_index": start, "dataset_to_index": start + T}
        for cam, writer in videos.items():
            kind, blobs = decode(feats[cam])
            if len(blobs) != T:
                raise ValueError(f"{self.name}: {cam} has {len(blobs)} frames for {T} steps")
            c, f, a, b = writer.write_episode(self._frames(writer, blobs))
            key = camera_name(cam)
            ep.update({f"videos/{key}/chunk_index": c, f"videos/{key}/file_index": f,
                       f"videos/{key}/from_timestamp": a, f"videos/{key}/to_timestamp": b})
        for path in p["meta"]:
            if path in feats:
                kind, vals = decode(feats[path])
                v = vals[0] if len(vals) else None
                ep["meta." + path.replace("/", ".")] = (
                    v.decode("utf-8", "replace") if isinstance(v, bytes) else
                    None if v is None else float(v))

        arrays = {k: _array_column(v, dt) for k, (v, dt) in cols.items()}
        arrays["timestamp"] = pa.array(np.arange(T, dtype=np.float32) / np.float32(self.fps))
        arrays["frame_index"] = pa.array(np.arange(T, dtype=np.int64))
        arrays["episode_index"] = pa.array(np.full(T, ep_index, np.int64))
        arrays["index"] = pa.array(np.arange(start, start + T, dtype=np.int64))
        arrays["task_index"] = pa.array(np.full(T, task_index, np.int64))
        if "is_last" in cols:
            arrays["next.done"] = pa.array(cols["is_last"][0].astype(bool))
        c, f = data.write_episode(pa.table(arrays))
        ep.update({"data/chunk_index": c, "data/file_index": f})
        return ep

    def _check_videos(self, videos, episodes):
        for cam, w in videos.items():
            for path, expected in w.files:
                out = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                     "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                    check=True, capture_output=True, text=True).stdout
                if int(out.strip().split(",")[0]) != expected:
                    raise RuntimeError(f"{path}: {out.strip()} frames written, {expected} expected")

    def _write_meta(self, p, videos, episodes, tasks, splits, total, ver, info_src):
        meta = self.out / "meta"
        (meta / "episodes/chunk-000").mkdir(parents=True, exist_ok=True)
        keys = sorted({k for e in episodes for k in e})
        table = {}
        for k in keys:
            vals = [e.get(k) for e in episodes]
            table[k] = pa.array(vals, type=pa.list_(pa.string())) if k == "tasks" else pa.array(vals)
        pq.write_table(pa.table(table), meta / "episodes/chunk-000/file-000.parquet")
        pq.write_table(pa.table({"task_index": pa.array(list(tasks.values()), pa.int64()),
                                 "task": pa.array(list(tasks.keys()))}), meta / "tasks.parquet")
        features = {}
        for cam, w in videos.items():
            wdt, hgt = w.size
            features[camera_name(cam)] = {
                "dtype": "video", "shape": [hgt, wdt, 3], "names": ["height", "width", "channel"],
                "video_info": {"video.fps": self.fps, "video.codec": "h264",
                               "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                               "has_audio": False}}
        for path, dtype, shape in p["numeric"]:
            features[column_name(path)] = _feature_entry(dtype, shape, self.fps)
        for k, d in (("timestamp", "float32"), ("frame_index", "int64"),
                     ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")):
            features[k] = _feature_entry(d, (1,), self.fps)
        if any(path.endswith("/is_last") or path == "steps/is_last" for path, _, _ in p["numeric"]):
            features["next.done"] = _feature_entry("bool", (1,), self.fps)
        info = {
            "codebase_version": "v3.0", "robot_type": "unknown",
            "total_episodes": len(episodes), "total_frames": total, "total_tasks": len(tasks),
            "chunks_size": self.chunks_size, "fps": self.fps,
            "splits": {s: f"{a}:{b + 1}" for s, (a, b) in splits.items()},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "data_files_size_in_mb": self.data_mb, "video_files_size_in_mb": self.video_mb,
            "features": features,
            "rlds": {"source": f"gs://{BUCKET}/{ver}", "description":
                     info_src.get("description", "")[:2000], "fit": list(self.fit),
                     "encoder": self.encoder, "cq": self.cq, "gop": self.gop,
                     "dropped": p["dropped"], "task_fields": p["text"],
                     "partial": bool(self.max_episodes)},
        }
        (meta / "info.json").write_text(json.dumps(info, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", help="registered Open-X dataset name, e.g. jaco_play")
    ap.add_argument("--fps", type=float, required=True, help="control frequency (RLDS has none)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tmp", type=Path, required=True, help="where shards are staged")
    ap.add_argument("--fit", default="320x240")
    ap.add_argument("--encoder", default="h264_nvenc")
    ap.add_argument("--cq", type=int, default=20)
    ap.add_argument("--gop", type=int, default=2)
    ap.add_argument("--max-dim", type=int, default=256)
    ap.add_argument("--max-episodes", type=int, default=None, help="stop early (testing)")
    ap.add_argument("--decode-threads", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    fit = tuple(int(v) for v in a.fit.lower().split("x"))
    out = Converter(a.name, a.out, a.tmp, a.fps, fit, a.encoder, a.cq, a.gop, a.max_dim,
                    a.max_episodes, a.decode_threads).run(a.overwrite)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
