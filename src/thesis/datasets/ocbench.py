"""OCBench scripted demonstrations (github.com/seohongpark/ocbench) as a dataset backend.

OCBench ships its data as `.npz` files of flat transition arrays: `actions`, `rewards`,
`masks`, `terminals` (and, from its MJWarp collector, `qpos`/`qvel`) hold one row per
control step, and `terminals` marks every episode's last step. `observations` holds one
row per stored observation: every step plus each episode's final one for a dense dataset,
or -- with `observation_interval` K > 1 -- only steps 0, K, 2K, ... plus the final one.
This backend serves them in the same windowed batch format as the lerobot backend, so a
model cannot tell which one it is reading.

Columns, named like the lerobot ones so the same algorithm configs read both:

    observation.state            the state observation vector (dense state datasets)
    observation.images.<camera>  one camera of a visual dataset, (C, H, W) uint8 per frame
    qpos, qvel                   the simulator state before each action, when recorded
    action                       the 7-d joint-delta action OCBench envs take

A per-step column's dimensions are named `<column>.<i>` (`qpos.0` ... `qpos.20`) -- the
names the eval backend gives the same columns -- so `slices:` keeps some of them with the
lerobot backend's glob patterns, e.g. the robot's joints without the cube's free joint.

A visual shard may keep its frames beside it in `<shard>-pixels.npy` instead of inside the
`.npz` (scripts/collect_ocbench_visual.py writes them that way). That file is memory-mapped
rather than loaded, since tens of GB of frames do not fit in RAM.
"""

import glob
import os

import numpy as np
import torch

from ..utils.spec import flow_source_entries, raw_index, spec_fields, split_index
from .base import BaseSource
from .lerobot import resolve_slice, split_episodes

STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"
IMAGE_PREFIX = "observation.images"
# per-step columns: one row per action, none for the observation after the last one
STEP_COLUMNS = {ACTION_COLUMN: "actions", "qpos": "qpos", "qvel": "qvel"}
# the camera order of a visual env's `pixel_cameras`, with `ur5e/wrist` shortened
DEFAULT_CAMERAS = ("front", "side", "wrist")


def resolve_paths(cfg):
    """The training `.npz` files `cfg` names, downloading them from the OCBench hub
    release when it names none.

    `dataset_path` is one file, a directory, or a glob. A directory or glob skips the
    `-val.npz` files beside each shard: OCBench's validation draws are its own
    distribution, and the held-out split here comes from `validation_split` instead.
    """
    path = cfg.get("dataset_path")
    if path is None:
        import ocbench

        paths, _ = ocbench.download_datasets(
            cfg.env_name, cfg.get("dataset_root"), cfg.get("num_shards")
        )
        return [str(p) for p in paths]
    path = os.path.expanduser(str(path))
    if os.path.isdir(path):
        path = os.path.join(path, "*.npz")
    paths = sorted(p for p in glob.glob(path) if not p.endswith("-val.npz"))
    if not paths:
        raise FileNotFoundError(f"no OCBench .npz files at '{cfg.get('dataset_path')}'")
    num_shards = cfg.get("num_shards")
    return paths[: int(num_shards)] if num_shards is not None else paths


def pixels_path(path):
    return str(path)[: -len(".npz")] + "-pixels.npy"


def episode_table(path):
    """(lengths, successes, observation_interval) of one file, read off its small columns.

    An episode succeeded when any of its rewards is 1: OCBench pays exactly 1 on the step
    the task is solved and 0 everywhere else, so a time-limit failure is all zeros.
    """
    with np.load(path) as data:
        interval = int(data["observation_interval"])
        terminals = data["terminals"]
        rewards = data["rewards"]
    ends = np.flatnonzero(terminals) + 1
    starts = np.concatenate([[0], ends[:-1]])
    successes = np.array([np.any(rewards[s:e] == 1) for s, e in zip(starts, ends)], dtype=bool)
    return ends - starts, successes, interval


def observation_rows(lengths, interval):
    """Stored observation rows per episode: steps 0, K, 2K, ... before the last action,
    plus the observation the last action produced."""
    return (lengths + interval - 1) // interval + 1


class OCBenchSource(BaseSource):
    """OCBench demonstrations, windowed by the temporal index spec exactly like
    LeRobotSource: each spec entry reads the field its `from` names over a window of step
    offsets around a sampled decision point t=0, a field read by several entries carries
    the sorted union of their windows, and `"last"` appends the episode's final row.

    A decision point is a step t of an episode of L actions. A per-step column (action,
    qpos, qvel) clamps its offsets into [0, L-1] and its `"last"` is step L-1; an
    observation column clamps into [0, L] and its `"last"` is the observation the last
    action produced. A sparse visual dataset stores frames every K steps only, so when any
    field reads a camera, decision points are the multiples of K and every camera offset
    must be one too -- a window then lands on stored frames exactly.

    Config knobs:
      env_name       OCBench environment the data comes from (and the eval rolls out in).
      dataset_path   a `.npz` file, a directory of them, or a glob; null downloads the
                     env's published datasets into `dataset_root`.
      dataset_root   download directory (default: OCBench's own cache).
      num_shards     keep the first N training shards, in filename order.
      success_only   keep only episodes that reach the goal (default true, as OCBench
                     trains). Its scripted policy makes deliberate mistakes, and about
                     12% of block-single episodes run out the clock correcting them.
      episodes       episode indices to keep, counted over every file in order before the
                     success filter; null keeps all.
      max_episodes   keep only the first N episodes left after those filters (and before
                     the split), so a dataset collected past a target size trains at it.
      validation_split / validation_split_seed
                     as on the lerobot backend: a seeded shuffle of the kept episodes.
      columns        `{from_key: column}` remap; default identity.
      slices         `{from_key: patterns}` keeping only some dimensions of a per-step
                     column, matched against its `<column>.<i>` names (`"qpos.[0-9]"`).
      cameras        camera names for a visual dataset's camera axis, in env order.
      image_size     `[H, W]` to resize camera frames to; null keeps native size.
      return_uint8   uint8 [0,255] frames (default); false gives float [0,1].
      drop_boundary  drop decision points whose window runs past the episode (default
                     true). Offsets before an episode clamp onto its first step.
    """

    def __init__(self, cfg, split=None):
        self._columns = dict(cfg.get("columns") or {})
        self._return_uint8 = cfg.get("return_uint8", True)
        size = cfg.get("image_size")
        self._image_size = tuple(int(v) for v in size) if size else None
        cameras = [str(c) for c in (cfg.get("cameras") or DEFAULT_CAMERAS)]

        paths = resolve_paths(cfg)
        tables = [episode_table(p) for p in paths]
        intervals = {interval for _, _, interval in tables}
        if len(intervals) != 1:
            raise ValueError(f"{paths} mix observation intervals {sorted(intervals)}")
        self._interval = intervals.pop()
        lengths = np.concatenate([lengths for lengths, _, _ in tables])
        successes = np.concatenate([successes for _, successes, _ in tables])
        keep = np.arange(len(lengths))
        if cfg.get("success_only", True):
            keep = keep[successes[keep]]
        requested = cfg.get("episodes")
        if requested is not None:
            keep = np.array(sorted(set(keep.tolist()) & {int(e) for e in requested}), dtype=int)
        if cfg.get("max_episodes") is not None:
            if len(keep) < int(cfg.max_episodes):
                raise ValueError(
                    f"max_episodes={cfg.max_episodes} but only {len(keep)} episodes of "
                    f"{paths} pass the success/episode filters"
                )
            keep = keep[: int(cfg.max_episodes)]
        if not len(keep):
            raise ValueError(f"no episodes left of {paths} after the success/episode filters")
        if split is not None:
            keep = np.asarray(split_episodes(
                keep.tolist(),
                cfg.get("validation_split", 0.0),
                split,
                seed=int(cfg.get("validation_split_seed", 0)),
            ), dtype=int)
        self.episodes = keep
        self.num_successes = int(successes[keep].sum())

        available = self._columns_of(paths[0], cameras)
        windows, has_last, field_to_col = {}, {}, {}
        entries = {**cfg.conditioning, **cfg.predict, **flow_source_entries(cfg.predict)}
        for key, spec in entries.items():
            for field in spec_fields(key, spec):
                col = self._columns.get(field, field)
                if col in available and "index" in spec:
                    relative, last = split_index(raw_index(spec))
                    windows.setdefault(field, set()).update(relative)
                    has_last[field] = has_last.get(field, False) or last
                    field_to_col[field] = col
        if not windows:
            raise ValueError(
                f"none of the requested modalities are OCBench columns. "
                f"available: {sorted(available)}"
            )
        self._windows = {f: np.asarray(sorted(steps), dtype=np.int64) for f, steps in windows.items()}
        self._has_last = {f for f, last in has_last.items() if last}
        self._field_to_col = field_to_col
        self.provided_modalities = set(field_to_col) | set(field_to_col.values())

        visual = {f for f, c in field_to_col.items() if c in self._camera_of}
        self._stride = self._interval if visual else 1
        for field in visual:
            off = [int(o) for o in self._windows[field] if o % self._interval]
            if off:
                raise ValueError(
                    f"field '{field}' reads camera offsets {off}, but frames are stored "
                    f"every {self._interval} steps: every camera offset must be a multiple"
                )

        self._load(paths, tables, set(field_to_col.values()))

        self._slices = {}
        for field, patterns in dict(cfg.get("slices") or {}).items():
            if field not in field_to_col:
                raise ValueError(
                    f"slice given for field '{field}', which no spec entry reads. "
                    f"sliceable fields: {sorted(f for f, c in field_to_col.items() if c in STEP_COLUMNS)}"
                )
            col = field_to_col[field]
            if col not in STEP_COLUMNS:
                raise ValueError(
                    f"field '{field}' reads '{col}', which has no per-dimension names to slice"
                )
            names = [f"{col}.{i}" for i in range(self._step[col].shape[-1])]
            self._slices[field] = np.asarray(resolve_slice(patterns, names, field), dtype=np.int64)

        # a per-step column reaches step L-1 at most; an observation reaches L
        ahead = max(
            [0] + [int(w.max()) - (self._field_to_col[f] not in STEP_COLUMNS)
                   for f, w in self._windows.items() if len(w)]
        )
        drop_boundary = cfg.get("drop_boundary", True)
        self._index = np.array(
            [
                (ordinal, t)
                for ordinal, length in enumerate(self._lengths)
                for t in range(0, length - ahead if drop_boundary else length, self._stride)
            ],
            dtype=np.int64,
        ).reshape(-1, 2)

    def _columns_of(self, path, cameras):
        """The columns a file serves, and (as a side effect) the camera -> axis map."""
        frames = pixels_path(path)
        with np.load(path) as data:
            keys = set(data.files)
            ob = None
            if "observations" in keys:
                # the header alone: a visual file's frames are too big to load to look
                with data.zip.open("observations.npy") as header:
                    version = np.lib.format.read_magic(header)
                    read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                            else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = read(header)
                ob = (shape, dtype)
        if os.path.exists(frames):
            arr = np.load(frames, mmap_mode="r")
            ob = (arr.shape, arr.dtype)
        available = {col for col, key in STEP_COLUMNS.items() if key in keys}
        self._camera_of = {}
        if ob is not None and ob[1] == np.uint8:
            n_cams = ob[0][1] if len(ob[0]) == 5 else 1
            if len(cameras) < n_cams:
                raise ValueError(f"the dataset holds {n_cams} cameras but `cameras:` names {cameras}")
            self._camera_of = {f"{IMAGE_PREFIX}.{n}": i for i, n in enumerate(cameras[:n_cams])}
            available |= set(self._camera_of)
        elif ob is not None:
            if self._interval != 1:
                raise ValueError(f"{path}: sparse state observations are not supported")
            available.add(STATE_COLUMN)
        return available

    def _load(self, paths, tables, cols):
        """Pull the kept episodes' rows out of every file, one array at a time.

        Per-step columns and state observations are copied into RAM; camera frames stay
        where they are -- a memory-mapped `-pixels.npy`, or the in-RAM array of a file that
        keeps them inline -- addressed per episode by (file, first row).
        """
        kept = set(self.episodes.tolist())
        step = {col: [] for col in cols if col in STEP_COLUMNS}
        states, lengths, self._frames, frame_refs = [], [], [], []
        need_frames = any(c in self._camera_of for c in cols)
        first = 0
        for path, (file_lengths, _, _) in zip(paths, tables):
            ends = np.cumsum(file_lengths)
            starts = ends - file_lengths
            ob_rows = observation_rows(file_lengths, self._interval)
            ob_starts = np.concatenate([[0], np.cumsum(ob_rows)[:-1]])
            local = [e for e in range(len(file_lengths)) if first + e in kept]
            first += len(file_lengths)
            if not local:
                continue
            with np.load(path) as data:
                for col in step:
                    values = data[STEP_COLUMNS[col]]
                    step[col].extend(values[starts[e]:ends[e]].astype(np.float32) for e in local)
                    del values
                if STATE_COLUMN in cols:
                    obs = data["observations"]
                    states.extend(obs[ob_starts[e]:ob_starts[e] + ob_rows[e]].astype(np.float32)
                                  for e in local)
                    del obs
                if need_frames:
                    frames = (np.load(pixels_path(path), mmap_mode="r")
                              if os.path.exists(pixels_path(path)) else data["observations"])
                    if frames.ndim == 4:
                        frames = frames[:, None]
                    frame_refs.extend((len(self._frames), int(ob_starts[e])) for e in local)
                    self._frames.append(frames)
            lengths.extend(int(file_lengths[e]) for e in local)
        self._lengths = np.asarray(lengths, dtype=np.int64)
        self._act_starts = np.concatenate([[0], np.cumsum(self._lengths)[:-1]])
        self._obs_starts = self._act_starts + np.arange(len(lengths))
        self._step = {col: np.concatenate(v) for col, v in step.items()}
        self._states = np.concatenate(states) if states else None
        self._frame_refs = np.asarray(frame_refs, dtype=np.int64).reshape(-1, 2)

    def __len__(self):
        return len(self._index)

    def _offsets(self, field, ordinal, t):
        """Episode-local steps of one field's window (plus its `"last"` step)."""
        length = int(self._lengths[ordinal])
        hi = length - 1 if self._field_to_col[field] in STEP_COLUMNS else length
        steps = np.clip(t + self._windows[field], 0, hi)
        if field in self._has_last:
            steps = np.append(steps, hi)
        return steps

    def __getitem__(self, idx):
        ordinal, t = (int(v) for v in self._index[idx])
        batch = {}
        for field, col in self._field_to_col.items():
            steps = self._offsets(field, ordinal, t)
            if col in STEP_COLUMNS:
                rows = self._step[col][self._act_starts[ordinal] + steps]
                if field in self._slices:
                    rows = rows[:, self._slices[field]]
                batch[field] = torch.from_numpy(np.ascontiguousarray(rows))
            elif col == STATE_COLUMN:
                batch[field] = torch.from_numpy(self._states[self._obs_starts[ordinal] + steps])
            else:
                batch[field] = self._read_frames(col, ordinal, steps)
        return batch

    def _read_frames(self, col, ordinal, steps):
        """Frames at episode-local steps: step s < L is stored row s // K, and step L --
        the observation after the last action -- is the episode's final row."""
        length = int(self._lengths[ordinal])
        rows = np.where(steps >= length, observation_rows(length, self._interval) - 1,
                        steps // self._interval)
        file, first = self._frame_refs[ordinal]
        frames = np.ascontiguousarray(self._frames[file][first + rows, self._camera_of[col]])
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        if self._image_size:
            frames = self._resize(frames)
        return frames if self._return_uint8 else frames.float() / 255.0

    def _resize(self, tensor):
        if tuple(tensor.shape[-2:]) == self._image_size:
            return tensor
        resized = torch.nn.functional.interpolate(tensor.float(), size=self._image_size, mode="area")
        return resized.round().clamp(0, 255).to(torch.uint8)

    def stats_columns(self, keys):
        """`keys` as flat per-step arrays over this split's episodes -- one row per action
        step, so an episode's extra final observation does not count twice. None when a
        key is a camera, which is not normalized."""
        out = {}
        for key in keys:
            col = self._field_to_col.get(key, self._columns.get(key, key))
            if col in self._step:
                out[key] = (self._step[col][:, self._slices[key]] if key in self._slices
                            else self._step[col])
            elif col == STATE_COLUMN and self._states is not None:
                steps = np.concatenate([
                    np.arange(s, s + n) for s, n in zip(self._obs_starts, self._lengths)
                ])
                out[key] = self._states[steps]
            else:
                return None
        return out or None
