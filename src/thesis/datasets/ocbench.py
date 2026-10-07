"""OCBench scripted demonstrations (github.com/seohongpark/ocbench) as a dataset backend.

OCBench ships its data as `.npz` files of flat transition arrays: `actions`, `rewards`,
`masks`, `terminals` hold one row per control step, `observations` one row per step plus
each episode's final observation, and `terminals` marks every episode's last step. This
backend serves them in the same windowed batch format as the lerobot backend, so a model
cannot tell which one it is reading.

Columns, named like the lerobot ones so the same algorithm configs read both:

    observation.state            the state observation vector (state envs)
    observation.images.<camera>  one camera of a visual env, (C, H, W) uint8 per step
    action                       the 7-d joint-delta action OCBench envs take
"""

import glob
import os

import numpy as np
import torch

from ..utils.spec import flow_source_entries, raw_index, spec_fields, split_index
from .base import BaseSource
from .lerobot import split_episodes

STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"
IMAGE_PREFIX = "observation.images"
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


def episode_table(path):
    """(lengths, successes) of every episode in one file, read off its small columns.

    An episode succeeded when any of its rewards is 1: OCBench pays exactly 1 on the step
    the task is solved and 0 everywhere else, so a time-limit failure is all zeros.
    """
    with np.load(path) as data:
        interval = int(data["observation_interval"])
        if interval != 1:
            raise ValueError(
                f"{path} stores sparse observations (observation_interval={interval}); "
                f"only dense datasets are supported"
            )
        terminals = data["terminals"]
        rewards = data["rewards"]
    ends = np.flatnonzero(terminals) + 1
    starts = np.concatenate([[0], ends[:-1]])
    successes = np.array([np.any(rewards[s:e] == 1) for s, e in zip(starts, ends)], dtype=bool)
    return ends - starts, successes


class OCBenchSource(BaseSource):
    """OCBench demonstrations, windowed by the temporal index spec exactly like
    LeRobotSource: each spec entry reads the field its `from` names over a window of step
    offsets around a sampled decision point t=0, a field read by several entries carries
    the sorted union of their windows, and `"last"` appends the episode's final frame.

    A decision point is a step t of an episode of L actions. An action offset clamps into
    [0, L-1]; an observation offset into [0, L], since every episode stores the observation
    its last action produced. `"last"` is that final observation for observation fields and
    the last action for the action field.

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
      validation_split / validation_split_seed
                     as on the lerobot backend: a seeded shuffle of the kept episodes.
      columns        `{from_key: column}` remap; default identity.
      cameras        camera names for a visual dataset's camera axis, in env order.
      image_size     `[H, W]` to resize camera frames to; null keeps native size.
      return_uint8   uint8 [0,255] frames (default); false gives float [0,1].
      drop_boundary  drop decision points whose window runs past the last action
                     (default true). Offsets before an episode clamp onto its first step.
    """

    def __init__(self, cfg, split=None):
        self._columns = dict(cfg.get("columns") or {})
        if cfg.get("slices"):
            raise ValueError(
                "the ocbench backend serves no per-dimension feature names, so `slices:` "
                "has nothing to match; remap with `columns:` instead"
            )
        self._return_uint8 = cfg.get("return_uint8", True)
        size = cfg.get("image_size")
        self._image_size = tuple(int(v) for v in size) if size else None
        cameras = [str(c) for c in (cfg.get("cameras") or DEFAULT_CAMERAS)]

        paths = resolve_paths(cfg)
        tables = [episode_table(p) for p in paths]
        lengths = np.concatenate([lengths for lengths, _ in tables])
        successes = np.concatenate([successes for _, successes in tables])
        keep = np.arange(len(lengths))
        if cfg.get("success_only", True):
            keep = keep[successes[keep]]
        requested = cfg.get("episodes")
        if requested is not None:
            keep = np.array(sorted(set(keep.tolist()) & {int(e) for e in requested}), dtype=int)
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

        with np.load(paths[0]) as data:
            ob_shape = data["observations"].shape
            ob_dtype = data["observations"].dtype
        visual = ob_dtype == np.uint8
        if visual and len(ob_shape) == 4:
            ob_shape = (ob_shape[0], 1, *ob_shape[1:])
        if visual and len(cameras) < ob_shape[1]:
            raise ValueError(
                f"the dataset holds {ob_shape[1]} cameras but `cameras:` names {cameras}"
            )
        self._camera_of = (
            {f"{IMAGE_PREFIX}.{name}": i for i, name in enumerate(cameras[: ob_shape[1]])}
            if visual else {}
        )
        available = {ACTION_COLUMN, *(self._camera_of or (STATE_COLUMN,))}

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

        self._load(paths, tables, need_obs=any(c != ACTION_COLUMN for c in field_to_col.values()))

        # an observation window may reach one row further than an action window: the
        # observation the last action produced
        ahead = max(
            [0] + [int(w.max()) - (self._field_to_col[f] != ACTION_COLUMN)
                   for f, w in self._windows.items() if len(w)]
        )
        drop_boundary = cfg.get("drop_boundary", True)
        self._index = np.array(
            [
                (ordinal, t)
                for ordinal, length in enumerate(self._lengths)
                for t in range(length - ahead if drop_boundary else length)
            ],
            dtype=np.int64,
        ).reshape(-1, 2)

    def _load(self, paths, tables, need_obs):
        """Pull the kept episodes' rows out of every file, one array at a time.

        Observations are skipped entirely when no field reads them, which is most of the
        memory of a visual dataset.
        """
        kept = set(self.episodes.tolist())
        actions, observations, lengths = [], [], []
        first = 0
        for path, (file_lengths, _) in zip(paths, tables):
            ends = np.cumsum(file_lengths)
            starts = ends - file_lengths
            local = [e for e in range(len(file_lengths)) if first + e in kept]
            first += len(file_lengths)
            if not local:
                continue
            with np.load(path) as data:
                file_actions = data["actions"]
                actions.extend(file_actions[starts[e]:ends[e]].astype(np.float32) for e in local)
                del file_actions
                if need_obs:
                    file_obs = data["observations"]
                    # one extra observation per episode: the one its last action produced
                    for e in local:
                        rows = file_obs[starts[e] + e : ends[e] + e + 1]
                        observations.append(rows if rows.dtype == np.uint8 else rows.astype(np.float32))
                    del file_obs
            lengths.extend(int(file_lengths[e]) for e in local)
        self._lengths = np.asarray(lengths, dtype=np.int64)
        self._act_starts = np.concatenate([[0], np.cumsum(self._lengths)[:-1]])
        self._obs_starts = self._act_starts + np.arange(len(lengths))
        self._actions = np.concatenate(actions)
        self._observations = np.concatenate(observations) if need_obs else None
        if self._observations is not None and self._observations.dtype == np.uint8 \
                and self._observations.ndim == 4:
            self._observations = self._observations[:, None]

    def __len__(self):
        return len(self._index)

    def _rows(self, field, ordinal, t):
        """Absolute row indices of one field's window (plus its `"last"` row)."""
        length = int(self._lengths[ordinal])
        is_action = self._field_to_col[field] == ACTION_COLUMN
        hi = length - 1 if is_action else length
        start = self._act_starts[ordinal] if is_action else self._obs_starts[ordinal]
        rows = np.clip(t + self._windows[field], 0, hi)
        if field in self._has_last:
            rows = np.append(rows, hi)
        return start + rows

    def __getitem__(self, idx):
        ordinal, t = (int(v) for v in self._index[idx])
        batch = {}
        for field, col in self._field_to_col.items():
            rows = self._rows(field, ordinal, t)
            if col == ACTION_COLUMN:
                batch[field] = torch.from_numpy(self._actions[rows])
            elif col == STATE_COLUMN:
                batch[field] = torch.from_numpy(self._observations[rows])
            else:
                frames = torch.from_numpy(self._observations[rows, self._camera_of[col]])
                frames = frames.permute(0, 3, 1, 2).contiguous()
                if self._image_size:
                    frames = self._resize(frames)
                batch[field] = frames if self._return_uint8 else frames.float() / 255.0
        return batch

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
            if col == ACTION_COLUMN:
                out[key] = self._actions
            elif col == STATE_COLUMN and self._observations is not None:
                steps = np.concatenate([
                    np.arange(s, s + n) for s, n in zip(self._obs_starts, self._lengths)
                ])
                out[key] = self._observations[steps]
            else:
                return None
        return out or None
