"""RoboJEPA clips over a mixture of LeRobot v3.0 embodiments (`backend: mixture`).

One sample is a fixed-length clip from one embodiment, sampled the way RoboJEPA's
RobotVideoDataset samples one (arXiv 2610.10515, App. B; facebookresearch/robo_jepa):

  * a view group first -- V cameras with probability `view_groups[V]` -- then a member by
    weight among the embodiments with enough cameras for it, an episode uniformly, and a
    start frame uniformly in [0, max(L - (T + 1) * s, s)), where s = ceil(fps / wm_fps)
    is the embodiment's frame stride. RoboJEPA partitions its ranks into view groups;
    here a batch mixes them and the empty slot is masked (`view_valid`).
  * T frames at that stride, T + 1 proprioceptive states, and the T actions between them
    as STATE DELTAS: positions and other dims differenced, an euler rotation composed as
    R_{t+1} R_t^T (embodiment `proprio:` layout). Actions and states are zero-padded to
    `pad_size` dims; a clip that runs past its episode end is zero-padded in time and
    `step_valid` marks the real frames.
  * cameras drawn without replacement, in random order: a one-view clip from the
    exterior cameras, a two-view clip from all of them. Each view gets its own
    Inception-style random resized crop (area `scale`, aspect `ratio`) resized to
    `image_size`; held-out clips are the full frame resized.

Fields, each T rows on one time axis (the spec's windows are checked against it):

    observation.images  uint8 (T, V, 3, H, W)     V = largest view group
    action              float (T, pad_size)       action t moves state t to t + 1
    observation.state   float (T, pad_size)
    step_valid          bool  (T,)                False past the episode end
    view_valid          bool  (V,)                False for an empty camera slot
    view_id             long  (V,)                semantic camera id (embodiment.VIEW_IDS)
    embodiment_id       long  ()                  index in `members`

A field the spec reads over fewer rows (actions stop one step short of the frames) is
served its first rows only, so the batch contract -- a field carries the union of its
readers' windows -- holds.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ...utils.spec import raw_index, spec_fields, split_index
from ..base import BaseSource
from .embodiment import load_embodiment
from .manifest import load_manifest
from .source import LeRobotV3Source

REPO_ROOT = Path(__file__).resolve().parents[4]
FIELDS = ("observation.images", "action", "observation.state", "step_valid", "view_valid",
          "view_id", "embodiment_id")


def state_delta(states, proprio):
    """(n, S) states in `proprio` order -> (n - 1, S) RoboJEPA state-delta actions."""
    out = np.diff(states, axis=0)
    if proprio.rotation:
        from scipy.spatial.transform import Rotation

        lo = len(proprio.position)
        rot = Rotation.from_euler("xyz", states[:, lo:lo + 3].astype(np.float64))
        out[:, lo:lo + 3] = (rot[1:] * rot[:-1].inv()).as_euler("xyz")
    return out.astype(np.float32)


def resized_crop_box(rng, height, width, scale, ratio):
    """torchvision's RandomResizedCrop.get_params on a numpy Generator: (top, left, h, w)."""
    area = height * width
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target = area * rng.uniform(*scale)
        aspect = math.exp(rng.uniform(*log_ratio))
        w = int(round(math.sqrt(target * aspect)))
        h = int(round(math.sqrt(target / aspect)))
        if 0 < w <= width and 0 < h <= height:
            return int(rng.integers(0, height - h + 1)), int(rng.integers(0, width - w + 1)), h, w
    in_ratio = width / height
    if in_ratio < min(ratio):
        w, h = width, int(round(width / min(ratio)))
    elif in_ratio > max(ratio):
        h, w = height, int(round(height * max(ratio)))
    else:
        w, h = width, height
    return (height - h) // 2, (width - w) // 2, h, w


class Member:
    """One embodiment of the mixture: its source, split episodes, stride and proprio."""

    def __init__(self, data_root, embodiment, emb_id, wm_fps, min_frames, split):
        if embodiment.proprio is None:
            raise ValueError(f"{embodiment.name}: a RoboJEPA clip needs a `proprio:` layout")
        self.embodiment = embodiment
        self.emb_id = emb_id
        self.source = LeRobotV3Source(Path(data_root).expanduser() / embodiment.source)
        self.fps = self.source.fps
        self.wm_fps = float(embodiment.wm_fps or wm_fps)
        if self.fps < self.wm_fps:
            raise ValueError(f"{embodiment.name}: {self.fps} fps cannot fill {self.wm_fps} Hz")
        self.stride = int(math.ceil(self.fps / self.wm_fps - 1e-9))
        pr = embodiment.proprio
        self.proprio = np.ascontiguousarray(self.source.column(pr.column)[:, list(pr.dims)])
        self.cameras = list(embodiment.cameras)
        self.exterior = [i for i, c in enumerate(self.cameras) if c.role != "wrist"] or \
            list(range(len(self.cameras)))

        manifest = load_manifest(data_root, embodiment, self.source)
        val = np.zeros(self.source.num_episodes, bool)
        val[manifest["val_episodes"]] = True
        side = np.ones_like(val) if split is None else (val if split == "val" else ~val)
        # five states give four frames, RoboJEPA's min_length_to_drop
        long_enough = self.source.lengths >= (min_frames + 1) * self.stride
        self.episodes = np.flatnonzero(side & long_enough)
        if not len(self.episodes):
            raise ValueError(f"{embodiment.name}: no {split or 'all'} episode is long enough")
        self.hours = manifest["hours"]

    def max_views(self):
        return len(self.cameras)

    def window(self, episode, start, frames):
        """Episode-relative frame indices, and the (n + 1, S) states behind them."""
        length = int(self.source.lengths[episode])
        s = self.stride
        frame_idx = np.arange(start, min(start + frames * s, length), s)
        state_idx = np.arange(start, min(start + (frames + 1) * s, length), s)
        row = int(self.source.first_row[episode])
        states = self.proprio[row + state_idx]
        n = min(len(frame_idx), len(state_idx) - 1)
        return frame_idx[:n], states[:n + 1]


class MixtureSource(BaseSource):
    provided_modalities = set(FIELDS)

    def __init__(self, cfg, split=None):
        self.cfg = cfg
        self.split = split
        self.frames = int(cfg.get("frames_per_clip", 16))
        self.pad = int(cfg.get("pad_size", 64))
        self.size = tuple(int(v) for v in cfg.get("image_size", (240, 320)))
        crop = cfg.get("random_resized_crop") or {}
        self.crop_scale = tuple(float(v) for v in crop.get("scale", (0.5, 1.0)))
        self.crop_ratio = tuple(float(v) for v in crop.get("ratio", (1.333, 1.777)))
        groups = {int(k): float(v) for k, v in dict(cfg.get("view_groups") or {1: 1.0}).items()}
        self.view_slots = max(groups)
        self.seed = int(cfg.get("seed", 0))
        self.train = split != "val"
        self._rows = self._field_rows(cfg)

        emb_dir = Path(cfg.embodiments_dir)
        if not emb_dir.is_absolute() and not emb_dir.exists():
            emb_dir = REPO_ROOT / emb_dir
        self.members = [
            Member(cfg.data_root, load_embodiment(emb_dir / f"{name}.yaml"), i,
                   float(cfg.get("fps", 5)), int(cfg.get("min_frames", 4)), split)
            for i, name in enumerate(cfg.members)
        ]
        overrides = dict(cfg.get("weights") or {})
        unknown = set(overrides) - set(cfg.members)
        if unknown:
            raise ValueError(f"weights for unknown members {sorted(unknown)}")
        temperature = float(cfg.get("temperature", 1.0))
        base = np.array([m.hours ** temperature * float(overrides.get(m.embodiment.name, 1.0))
                         for m in self.members])
        self.groups = []
        for views, prob in sorted(groups.items()):
            eligible = np.array([m.max_views() >= views for m in self.members])
            if not eligible.any():
                raise ValueError(f"no member has {views} cameras for view group {views}")
            w = np.where(eligible, base, 0.0)
            self.groups.append((views, prob, w / w.sum()))
        self.group_probs = np.array([g[1] for g in self.groups]) / sum(g[1] for g in self.groups)

        if not self.train:
            per = int(cfg.get("val_windows_per_member", 64))
            rng = np.random.default_rng([self.seed, 0xFACE])
            self.items = []
            for m in range(len(self.members)):
                for _ in range(per):
                    self.items.append((m, int(rng.integers(1 << 62))))
        self.epoch_size = int(cfg.get("epoch_size", 100_000))

    def _field_rows(self, cfg):
        """Rows per field, from the spec windows the dataset config interpolates; each
        field's union must be contiguous and start at the clip's first frame."""
        entries = {**dict(cfg.get("conditioning") or {}), **dict(cfg.get("predict") or {})}
        unions = {}
        for name, spec in entries.items():
            if "index" not in spec:
                continue
            rel, has_last = split_index(raw_index(spec))
            if has_last:
                raise ValueError(f"'{name}': the mixture backend has no `last` frame")
            for field in spec_fields(name, spec):
                unions.setdefault(field, set()).update(rel)
        if not unions:
            return {f: self.frames for f in ("observation.images", "action", "observation.state")}
        base = min(min(u) for u in unions.values())
        rows = {}
        for field, u in unions.items():
            u = sorted(u)
            if field not in FIELDS[:3]:
                raise ValueError(f"the mixture backend serves no field '{field}'")
            if u != list(range(base, base + len(u))):
                raise ValueError(
                    f"field '{field}' is read at {u}; mixture clips are one contiguous "
                    f"window from offset {base}, so every field's union must start there"
                )
            if len(u) > self.frames:
                raise ValueError(f"field '{field}' needs {len(u)} rows, clips have {self.frames}")
            rows[field] = len(u)
        return rows

    def __len__(self):
        return self.epoch_size if self.train else len(self.items)

    def _draw(self, rng, member=None):
        if member is None:
            g = int(rng.choice(len(self.groups), p=self.group_probs))
            views, _, weights = self.groups[g]
            member = self.members[int(rng.choice(len(self.members), p=weights))]
        else:
            fits = [v for v, _, _ in self.groups if v <= member.max_views()]
            views = fits[int(rng.integers(len(fits)))]
        episode = int(member.episodes[int(rng.integers(len(member.episodes)))])
        length = int(member.source.lengths[episode])
        s = member.stride
        start = int(rng.integers(0, max(length - (self.frames + 1) * s, s)))
        pool = member.exterior if views == 1 else list(range(len(member.cameras)))
        cams = [int(c) for c in rng.choice(pool, size=views, replace=False)]
        return member, episode, start, cams

    def __getitem__(self, idx):
        if self.train:
            rng = np.random.default_rng([torch.initial_seed() % (1 << 63), int(idx)])
            member, episode, start, cams = self._draw(rng)
        else:
            m, seed = self.items[idx]
            rng = np.random.default_rng([self.seed, seed])
            member, episode, start, cams = self._draw(rng, self.members[m])
        return self._clip(member, episode, start, cams, rng if self.train else None)

    def _clip(self, member, episode, start, cams, rng):
        T, V, P = self.frames, self.view_slots, self.pad
        frame_idx, states = member.window(episode, start, T)
        n = len(frame_idx)
        actions = state_delta(states, member.embodiment.proprio)
        S = states.shape[1]
        if S > P:
            raise ValueError(f"{member.embodiment.name}: {S}-dim proprio exceeds pad_size {P}")

        images = torch.zeros(T, V, 3, *self.size, dtype=torch.uint8)
        for slot, c in enumerate(cams):
            raw = member.source.frames(member.cameras[c].column, episode, frame_idx)
            images[:n, slot] = self._transform(raw, rng)
        action = np.zeros((T, P), np.float32)
        action[:n, :S] = actions[:n]
        state = np.zeros((T, P), np.float32)
        state[:n, :S] = states[:n]
        step_valid = np.arange(T) < n
        view_valid = np.arange(V) < len(cams)
        view_id = np.zeros(V, np.int64)
        view_id[:len(cams)] = [member.cameras[c].view_id for c in cams]

        rows = self._rows
        return {
            "observation.images": images[:rows.get("observation.images", T)],
            "action": torch.from_numpy(action[:rows.get("action", T)]),
            "observation.state": torch.from_numpy(state[:rows.get("observation.state", T)]),
            "step_valid": torch.from_numpy(step_valid),
            "view_valid": torch.from_numpy(view_valid),
            "view_id": torch.from_numpy(view_id),
            "embodiment_id": torch.tensor(member.emb_id, dtype=torch.long),
        }

    def _transform(self, frames, rng):
        """uint8 (n, 3, h, w) -> uint8 (n, 3, *image_size): one crop for the whole clip."""
        x = frames.float()
        if rng is not None:
            top, left, h, w = resized_crop_box(rng, x.shape[-2], x.shape[-1],
                                               self.crop_scale, self.crop_ratio)
            x = x[..., top:top + h, left:left + w]
        x = F.interpolate(x, size=self.size, mode="bilinear", antialias=True,
                          align_corners=False)
        return x.round().clamp(0, 255).to(torch.uint8)
