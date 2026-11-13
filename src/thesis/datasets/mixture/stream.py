"""A weighted infinite stream over (source, embodiment) members, and fixed validation
windows from held-out episodes.

Training is step-based: epochs mean nothing across a mixture. Each draw picks a member by
weight, then an anchor uniformly from that member's valid window starts.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .embodiment import load_embodiment
from .manifest import load_manifest
from .source import LeRobotV3Source


class Member:
    """One embodiment over one source, restricted to a split and a clock."""

    def __init__(self, data_root, embodiment, clock, split, emb_id):
        self.embodiment = embodiment
        self.emb_id = emb_id
        self.clock = clock
        self.source = LeRobotV3Source(Path(data_root).expanduser() / embodiment.source)
        self.manifest = load_manifest(data_root, embodiment, self.source)
        self.fps = self.source.fps
        self.action, self.kinds = embodiment.assemble(self.source, embodiment.action)
        self.state, _ = embodiment.assemble(self.source, embodiment.state)

        val = np.zeros(self.source.num_episodes, bool)
        val[self.manifest["val_episodes"]] = True
        span = clock.span(self.fps)
        fits = self.source.lengths > span
        self.episodes = np.flatnonzero((val if split == "val" else ~val) & fits)
        if len(self.episodes) == 0:
            raise ValueError(f"{embodiment.name}: no {split} episode is longer than {span} frames")
        self.dropped = int(((val if split == "val" else ~val) & ~fits).sum())
        self._cum = np.cumsum(self.source.lengths[self.episodes] - span)

    @property
    def num_anchors(self):
        return int(self._cum[-1])

    def anchor(self, r):
        """The r-th valid (episode, start frame), counting through the split in order."""
        e = int(np.searchsorted(self._cum, r, side="right"))
        return int(self.episodes[e]), int(r - (self._cum[e - 1] if e else 0))

    def window(self, episode, start, cams):
        row = int(self.source.first_row[episode])
        idx = start + self.clock.offsets(self.fps)
        actions = self.action[row:row + int(self.source.lengths[episode])]
        return {
            "emb": self.emb_id,
            "fps": self.fps,
            "cams": list(cams),
            "frames": torch.stack([
                self.source.frames(self.embodiment.cameras[c].column, episode, idx)
                for c in cams]),
            "action": np.ascontiguousarray(
                self.clock.chunks(actions, self.kinds, start, self.fps), np.float32),
            "state": self.state[row + idx],
        }


class CameraPolicy:
    """Which of an embodiment's cameras a sample keeps. `roles` filters by role tag,
    `max_views` caps the count, drawing which views survive at random per sample."""

    def __init__(self, roles=None, max_views=None):
        self.roles = set(roles) if roles else None
        self.max_views = max_views

    def candidates(self, embodiment):
        cams = [i for i, c in enumerate(embodiment.cameras)
                if self.roles is None or c.role in self.roles]
        if not cams:
            raise ValueError(f"{embodiment.name}: no camera has a role in {sorted(self.roles)}")
        return cams

    def select(self, embodiment, rng=None):
        cams = self.candidates(embodiment)
        if self.max_views and len(cams) > self.max_views:
            if rng is None:
                return cams[:self.max_views]
            return sorted(rng.choice(cams, self.max_views, replace=False).tolist())
        return cams


def mixture_weights(members, temperature=1.0, overrides=None):
    """p_i proportional to hours_i ** temperature, times any per-member override."""
    overrides = overrides or {}
    unknown = set(overrides) - {m.embodiment.name for m in members}
    if unknown:
        raise ValueError(f"weight overrides for unknown members {sorted(unknown)}")
    w = np.array([m.manifest["hours"] ** temperature * overrides.get(m.embodiment.name, 1.0)
                  for m in members])
    return w / w.sum()


class MixtureStream(IterableDataset):
    """Infinite samples. Every (seed, rank, worker) triple is its own random stream."""

    def __init__(self, members, weights, cameras, seed, rank=0):
        self.members = members
        self.weights = weights
        self.cameras = cameras
        self.seed = seed
        self.rank = rank

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng([self.seed, self.rank, info.id if info else 0])
        while True:
            member = self.members[rng.choice(len(self.members), p=self.weights)]
            episode, start = member.anchor(rng.integers(member.num_anchors))
            yield member.window(episode, start, self.cameras.select(member.embodiment, rng))


class FixedWindows(Dataset):
    """The same `per_member` windows from each member on every pass, grouped by member so
    validation metrics can be reported per embodiment."""

    def __init__(self, members, cameras, per_member, seed):
        rng = np.random.default_rng([seed, 0xFACE])
        self.members = members
        self.cameras = cameras
        self.items = [(m, *members[m].anchor(r))
                      for m in range(len(members))
                      for r in np.sort(rng.integers(members[m].num_anchors, size=per_member))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        m, episode, start = self.items[i]
        member = self.members[m]
        return member.window(episode, start, self.cameras.select(member.embodiment))


def build_members(data_root, embodiments_dir, names, clock, split):
    return [Member(data_root, load_embodiment(Path(embodiments_dir) / f"{name}.yaml"),
                   clock, split, emb_id)
            for emb_id, name in enumerate(names)]


def mixture_spec(members):
    """What the model needs to know about the mixture: per embodiment, its action chunk
    shape, cameras and normalization statistics. Plain data, saved with checkpoints."""
    return {"embodiments": [
        {"name": m.embodiment.name,
         "fps": m.fps,
         "substeps": m.clock.substeps(m.fps),
         "action_dim": m.manifest["action_dim"],
         "state_dim": m.manifest["state_dim"],
         "cameras": m.manifest["cameras"],
         "stats": m.manifest["stats"]}
        for m in members]}
