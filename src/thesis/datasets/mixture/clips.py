"""RoboJEPA clips over a mixture of LeRobot v3.0 embodiments (`backend: mixture`).

One sample is a fixed-length clip from one embodiment, sampled the way RoboJEPA's
RobotVideoDataset samples one (arXiv 2610.10515, App. B; facebookresearch/robo_jepa):

  * a view group first -- V cameras with probability `view_groups[V]` -- then a member by
    weight among the embodiments with at least V cameras in that group's pool, an episode
    uniformly, and a start frame uniformly in [0, max(L - (T + 1) * s, s)), where
    s = ceil(fps / wm_fps) is the embodiment's frame stride. RoboJEPA partitions its ranks
    into view groups; here a batch mixes them and the empty slot is masked (`view_valid`).
  * T frames at that stride, T + 1 proprioceptive states, and the T actions between them
    as STATE DELTAS: positions and other dims differenced, an euler rotation composed as
    R_{t+1} R_t^T (embodiment `proprio:` layout). Actions and states are zero-padded to
    `pad_size` dims; a clip that runs past its episode end is zero-padded in time and
    `step_valid` marks the real frames.
  * cameras drawn without replacement, in random order, from the group's pool:
    `camera_roles[V]` lists the roles a V-view clip may draw (`default` covers the
    unlisted V, absent = every camera). RoboJEPA's rule, the default, is
    `{1: [exterior]}`: a one-view clip from the exterior cameras, more from all of them.
    Each view gets its own Inception-style random resized crop (area `scale`, aspect
    `ratio`) resized to `image_size`; held-out clips are the full frame resized.

Options beyond RoboJEPA's recipe:

  normalize_per_embodiment  (default true) each embodiment's actions and states are mapped
        to [-1, 1] by its own 1st/99th percentiles, per dim, and clipped to +-5. Their
        units differ by orders of magnitude across Open-X (a robocook step moves ~1e-3,
        a jaco command is +-0.2, ucsd states reach 8), and every embodiment shares the
        trunk's input space. State percentiles are over every frame of the proprio dims,
        state-delta ones over every pair of frames one stride apart, commanded-action ones
        are the manifest's. A dim with no spread is only centred. The scales are
        `stats["mixture"]`, which the trainer stores as the model's `norm_stats`, so
        they are saved in every checkpoint; a resumed run, a held-out eval or a deployed
        policy passes them back (`build_dataset(norm_stats_override=...)`) and uses them
        as trained instead of recomputing them from the data on disk.
  clock: stride | time  (default stride) `time` shows the frame nearest each step's time,
        round(k * fps / wm_fps), so a 12 fps source runs at 5 Hz rather than the 4 Hz a
        ceil stride gives it. Both agree whenever fps is a multiple of wm_fps.
  actions: proprio_delta | commanded  (default proprio_delta) `commanded` serves the
        recorded command columns the embodiment file's `action:` groups name instead of
        state deltas: for step t, every native command between the frame shown at t and
        the one shown at t + 1, K = round(fps / wm_fps) of them (an uneven gap is
        resampled to K by each group's `kind`, clock.resample), normalized per dim, and
        flattened to K * D before the zero padding.
  state: proprio | recorded  (default proprio) `recorded` serves the embodiment file's
        `state:` groups, concatenated, normalized by the manifest's percentiles, instead of
        the `proprio:` layout; with `actions: commanded` the file then needs no `proprio:`.

Fields, each T rows on one time axis (the spec's windows are checked against it):

    observation.images  uint8 (T, V, 3, H, W)     V = largest view group; with a single
                                                  view slot the V axis is dropped
    action              float (T, pad_size)       action t moves state t to t + 1
    action_valid        bool  (pad_size,)         True on the dims the embodiment fills
    observation.state   float (T, pad_size)
    step_valid          bool  (T,)                False past the episode end
    view_valid          bool  (V,)                False for an empty camera slot
    view_id             long  (V,)                semantic camera id (embodiment.VIEW_IDS)
    camera_uid          long  (V,)                one id per (member, camera), from 0 in
                                                  member order; `num_cameras` of them
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
from .clock import resample
from .embodiment import load_embodiment
from .manifest import load_manifest
from .source import LeRobotV3Source

REPO_ROOT = Path(__file__).resolve().parents[4]
FIELDS = ("observation.images", "action", "observation.state", "step_valid", "view_valid",
          "view_id", "camera_uid", "embodiment_id", "action_valid")
CLIP = 5.0
STATS_PAIRS = 500_000


def pair_delta(before, after, proprio):
    """(n, S) states in `proprio` order before and after a step -> (n, S) RoboJEPA
    state-delta actions."""
    out = after - before
    if proprio.rotation:
        from scipy.spatial.transform import Rotation

        lo = len(proprio.position)
        r0 = Rotation.from_euler("xyz", before[:, lo:lo + 3].astype(np.float64))
        r1 = Rotation.from_euler("xyz", after[:, lo:lo + 3].astype(np.float64))
        out[:, lo:lo + 3] = (r1 * r0.inv()).as_euler("xyz")
    return out.astype(np.float32)


def state_delta(states, proprio):
    """(n, S) states in `proprio` order -> (n - 1, S) RoboJEPA state-delta actions."""
    return pair_delta(states[:-1], states[1:], proprio)


def quantile_scale(q01, q99):
    """Per-dim (mid, half) mapping [q01, q99] onto [-1, 1]; a dim without spread keeps
    unit scale and is only centred."""
    q01, q99 = np.asarray(q01, np.float32), np.asarray(q99, np.float32)
    half = (q99 - q01) / 2
    return (q99 + q01) / 2, np.where(half > 1e-6, half, 1.0).astype(np.float32)


def apply_scale(x, scale):
    mid, half = scale
    return np.clip((x - mid) / half, -CLIP, CLIP).astype(np.float32)


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
    """One embodiment of the mixture: its source, split episodes, clock, proprio, and the
    per-dim scales its actions and states are normalized with."""

    def __init__(self, data_root, embodiment, emb_id, wm_fps, min_frames, split,
                 clock="stride", actions="proprio_delta", normalize=True, scales=None,
                 state="proprio"):
        if state not in ("proprio", "recorded"):
            raise ValueError(f"unknown state '{state}'; use proprio or recorded")
        if embodiment.proprio is None and (state == "proprio" or actions == "proprio_delta"):
            raise ValueError(f"{embodiment.name}: a RoboJEPA clip needs a `proprio:` layout")
        if state == "recorded" and not embodiment.state:
            raise ValueError(f"{embodiment.name}: `state: recorded` needs `state:` groups")
        if clock not in ("stride", "time"):
            raise ValueError(f"unknown clock '{clock}'; use stride or time")
        if actions not in ("proprio_delta", "commanded"):
            raise ValueError(f"unknown actions '{actions}'; use proprio_delta or commanded")
        self.embodiment = embodiment
        self.emb_id = emb_id
        self.source = LeRobotV3Source(Path(data_root).expanduser() / embodiment.source)
        self.fps = self.source.fps
        self.wm_fps = float(embodiment.wm_fps or wm_fps)
        if self.fps < self.wm_fps:
            raise ValueError(f"{embodiment.name}: {self.fps} fps cannot fill {self.wm_fps} Hz")
        self.ratio = self.fps / self.wm_fps
        self.stride = int(math.ceil(self.ratio - 1e-9))
        self.clock = clock
        self.actions = actions
        self.substeps = max(1, int(round(self.ratio)))
        # `proprio` is the per-frame state a clip carries, whichever convention fills it
        if state == "recorded":
            self.proprio, _ = embodiment.assemble(self.source, embodiment.state)
        else:
            pr = embodiment.proprio
            self.proprio = np.ascontiguousarray(self.source.column(pr.column)[:, list(pr.dims)])
        self.cameras = list(embodiment.cameras)

        manifest = load_manifest(data_root, embodiment, self.source)
        val = np.zeros(self.source.num_episodes, bool)
        val[manifest["val_episodes"]] = True
        side = np.ones_like(val) if split is None else (val if split == "val" else ~val)
        # five states give four frames, RoboJEPA's min_length_to_drop
        long_enough = self.source.lengths >= self.reach(min_frames + 1)
        self.episodes = np.flatnonzero(side & long_enough)
        if not len(self.episodes):
            raise ValueError(f"{embodiment.name}: no {split or 'all'} episode is long enough")
        self.hours = manifest["hours"]

        if actions == "commanded":
            self.commanded, self.action_kinds = embodiment.assemble(self.source, embodiment.action)
            self.action_width = self.substeps * self.commanded.shape[1]
        else:
            self.action_width = self.proprio.shape[1]
        self.scales = None
        if normalize and scales is not None:
            self.scales = {k: (np.asarray(v["mid"], np.float32), np.asarray(v["half"], np.float32))
                           for k, v in scales.items()}
            widths = {"state": self.proprio.shape[1], "action": self.action_width // self.substeps
                      if actions == "commanded" else self.proprio.shape[1]}
            for k, (mid, _) in self.scales.items():
                if len(mid) != widths[k]:
                    raise ValueError(f"{embodiment.name}: stored {k} stats have {len(mid)} dims, "
                                     f"the data has {widths[k]}")
        elif normalize:
            if state == "recorded":
                st = manifest["stats"]["state"]
                state = quantile_scale(st["q01"], st["q99"])
            else:
                state = quantile_scale(*np.quantile(self.proprio, [0.01, 0.99], axis=0))
            if actions == "commanded":
                st = manifest["stats"]["action"]
                action = quantile_scale(st["q01"], st["q99"])
            else:
                action = quantile_scale(*np.quantile(self._delta_sample(), [0.01, 0.99], axis=0))
            self.scales = {"state": state, "action": action}

    def _delta_sample(self):
        """State deltas over pairs of frames one step apart within an episode, at most
        STATS_PAIRS of them (a fixed draw), for the action percentiles."""
        gap = self.stride if self.clock == "stride" else self.substeps
        ep = np.full(len(self.proprio), -1)
        for i, (first, length) in enumerate(zip(self.source.first_row, self.source.lengths)):
            ep[first:first + length] = i
        rows = (np.flatnonzero((ep[:-gap] == ep[gap:]) & (ep[:-gap] >= 0))
                if len(ep) > gap else np.zeros(0, np.int64))
        if not len(rows):
            raise ValueError(f"{self.embodiment.name}: no episode spans one step")
        if len(rows) > STATS_PAIRS:
            rows = np.sort(np.random.default_rng(0).choice(rows, STATS_PAIRS, replace=False))
        return pair_delta(self.proprio[rows], self.proprio[rows + gap], self.embodiment.proprio)

    def reach(self, n):
        """Native-frame offset of step n from a window's first frame."""
        if self.clock == "stride":
            return n * self.stride
        return np.round(np.asarray(n) * self.ratio).astype(np.int64)

    def pool(self, roles):
        """Camera indices whose role is in `roles` (None: every camera)."""
        return [i for i, c in enumerate(self.cameras) if roles is None or c.role in roles]

    def window(self, episode, start, frames):
        """Episode-relative indices of the n shown frames and of the n + 1 states behind
        them (state t + 1 sits at frame t + 1), and those (n + 1, S) states."""
        length = int(self.source.lengths[episode])
        state_idx = start + self.reach(np.arange(frames + 1))
        state_idx = state_idx[state_idx < length]
        n = min(frames, len(state_idx) - 1)
        row = int(self.source.first_row[episode])
        return state_idx[:n], state_idx[:n + 1], self.proprio[row + state_idx[:n + 1]]

    def action_rows(self, episode, state_idx, states):
        """(n, action_width) actions of a window, normalized when the member is."""
        if self.actions == "proprio_delta":
            actions = state_delta(states, self.embodiment.proprio)
        else:
            row = int(self.source.first_row[episode])
            K, kinds = self.substeps, self.action_kinds
            chunks = [self.commanded[row + a:row + b] for a, b in zip(state_idx[:-1], state_idx[1:])]
            actions = np.stack([c if len(c) == K else resample(c, kinds, K) for c in chunks])
        if self.scales is not None:
            actions = apply_scale(actions, self.scales["action"])
        return actions.reshape(len(actions), -1)


class MixtureSource(BaseSource):
    provided_modalities = set(FIELDS)

    def __init__(self, cfg, split=None, stats=None):
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
        roles = cfg.get("camera_roles")
        roles = {1: ["exterior"]} if roles is None else dict(roles)
        default_roles = roles.pop("default", None)
        roles = {int(k): v for k, v in roles.items()}
        self.camera_roles = {
            v: (None if r is None else [str(x) for x in r])
            for v, r in ((v, roles.get(v, default_roles)) for v in groups)}
        self.seed = int(cfg.get("seed", 0))
        self.train = split != "val"
        self._rows = self._field_rows(cfg)

        emb_dir = Path(cfg.embodiments_dir)
        if not emb_dir.is_absolute() and not emb_dir.exists():
            emb_dir = REPO_ROOT / emb_dir
        clock = str(cfg.get("clock", "stride"))
        actions = str(cfg.get("actions", "proprio_delta"))
        state = str(cfg.get("state", "proprio"))
        normalize = bool(cfg.get("normalize_per_embodiment", True))
        stored = self._stored_scales(stats, clock, actions, normalize, list(cfg.members))
        self.members = [
            Member(cfg.data_root, load_embodiment(emb_dir / f"{name}.yaml"), i,
                   float(cfg.get("fps", 5)), int(cfg.get("min_frames", 4)), split,
                   clock=clock, actions=actions, normalize=normalize,
                   scales=stored.get(name) if stored else None, state=state)
            for i, name in enumerate(cfg.members)
        ]
        for m in self.members:
            width = max(m.action_width, m.proprio.shape[1])
            if width > self.pad:
                raise ValueError(f"{m.embodiment.name}: {width}-dim actions or states exceed "
                                 f"pad_size {self.pad}")
        self.camera_offset = np.cumsum([0] + [len(m.cameras) for m in self.members])
        self.num_cameras = int(self.camera_offset[-1])
        if normalize:
            self.stats = {"mixture": {"clock": clock, "actions": actions, "embodiments": {
                m.embodiment.name: {k: {"mid": v[0].tolist(), "half": v[1].tolist()}
                                    for k, v in m.scales.items()}
                for m in self.members}}}
        overrides = dict(cfg.get("weights") or {})
        unknown = set(overrides) - set(cfg.members)
        if unknown:
            raise ValueError(f"weights for unknown members {sorted(unknown)}")
        temperature = float(cfg.get("temperature", 1.0))
        base = np.array([m.hours ** temperature * float(overrides.get(m.embodiment.name, 1.0))
                         for m in self.members])
        self.groups = []
        for views, prob in sorted(groups.items()):
            eligible = np.array([len(m.pool(self.camera_roles[views])) >= views
                                 for m in self.members])
            if not eligible.any():
                raise ValueError(f"no member has {views} cameras with roles "
                                 f"{self.camera_roles[views]} for view group {views}")
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

    @staticmethod
    def _stored_scales(stats, clock, actions, normalize, members):
        """Per-member scales from a checkpoint's `norm_stats`, or None to compute them.
        Stats made under another action convention or clock would silently mis-scale, so
        they are refused, as is a member the checkpoint never saw."""
        entry = (stats or {}).get("mixture")
        if not entry or not normalize:
            return None
        if (entry["clock"], entry["actions"]) != (clock, actions):
            raise ValueError(f"stored mixture stats are for clock={entry['clock']}, "
                             f"actions={entry['actions']}; this config has clock={clock}, "
                             f"actions={actions}")
        missing = [m for m in members if m not in entry["embodiments"]]
        if missing:
            raise ValueError(f"stored mixture stats have no entry for members {missing}")
        return entry["embodiments"]

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
            fits = [v for v, _, _ in self.groups
                    if len(member.pool(self.camera_roles[v])) >= v]
            views = fits[int(rng.integers(len(fits)))]
        episode = int(member.episodes[int(rng.integers(len(member.episodes)))])
        length = int(member.source.lengths[episode])
        start = int(rng.integers(0, max(length - int(member.reach(self.frames + 1)),
                                        member.stride)))
        pool = member.pool(self.camera_roles[views])
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
        frame_idx, state_idx, states = member.window(episode, start, T)
        n = len(frame_idx)
        actions = member.action_rows(episode, state_idx, states)
        if member.scales is not None:
            states = apply_scale(states, member.scales["state"])
        S, A = states.shape[1], actions.shape[1]

        rows = self._rows
        # only the frames some spec reads are decoded; the rest of the window is metadata
        shown = min(n, rows.get("observation.images", T))
        images = torch.zeros(T, V, 3, *self.size, dtype=torch.uint8)
        for slot, c in enumerate(cams):
            raw = member.source.frames(member.cameras[c].column, episode, frame_idx[:shown])
            images[:shown, slot] = self._transform(raw, rng)
        if V == 1:
            images = images[:, 0]
        action = np.zeros((T, P), np.float32)
        action[:n, :A] = actions[:n]
        state = np.zeros((T, P), np.float32)
        state[:n, :S] = states[:n]
        step_valid = np.arange(T) < n
        view_valid = np.arange(V) < len(cams)
        view_id = np.zeros(V, np.int64)
        view_id[:len(cams)] = [member.cameras[c].view_id for c in cams]
        camera_uid = np.zeros(V, np.int64)
        camera_uid[:len(cams)] = [self.camera_offset[member.emb_id] + c for c in cams]

        return {
            "observation.images": images[:rows.get("observation.images", T)],
            "action": torch.from_numpy(action[:rows.get("action", T)]),
            "observation.state": torch.from_numpy(state[:rows.get("observation.state", T)]),
            "step_valid": torch.from_numpy(step_valid),
            "view_valid": torch.from_numpy(view_valid),
            "view_id": torch.from_numpy(view_id),
            "camera_uid": torch.from_numpy(camera_uid),
            "embodiment_id": torch.tensor(member.emb_id, dtype=torch.long),
            "action_valid": torch.arange(P) < A,
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
