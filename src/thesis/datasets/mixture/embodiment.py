"""Embodiment files: what a source's columns mean.

    name: taco_play
    source: taco_play                  # dataset directory under the data root
    action:
      - {from: action, dims: [0, 1, 2, 3, 4, 5], kind: delta}
      - {from: action, dims: [6], kind: discrete}
    state:
      - {from: observation.state}
    cameras:
      - {name: static, from: observation.images.rgb_static, role: exterior, view: default}
      - {name: gripper, from: observation.images.rgb_gripper, role: wrist, view: wrist}
    proprio:                             # RoboJEPA's state-delta convention (clips.py)
      from: observation.state
      position: [0, 1, 2]                # end-effector xyz
      rotation: [3, 4, 5]                # euler xyz; deltas composed as rotations
      gripper: [6]
    wm_fps: 5                            # optional: this embodiment's world-model rate

`proprio` lays out the proprioceptive state a RoboJEPA clip carries, in the order
position, rotation, other, gripper; its per-step action is the state's finite difference,
with the rotation delta taken as euler(R_{t+1} R_t^T). `other` dims (joint angles, or
anything whose geometry is unknown) are plain differences. `view` is the camera's
semantic id in RoboJEPA's vocabulary (VIEW_IDS), shared across embodiments.

Action and state widths are whatever the groups add up to; nothing is padded to a shared
width. `kind` only says how a group behaves when it is resampled in time (clock.py):
`delta` increments are summed, `absolute` targets are interpolated, `discrete` channels
(grippers, mode switches) are held. `dims` defaults to the whole column. Several files may
point at one source, each becoming its own embodiment.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

KINDS = ("delta", "absolute", "discrete")

# RoboJEPA's semantic camera vocabulary (their Table 9); an id is what the model sees
VIEW_IDS = {"left": 0, "right": 1, "wrist": 2, "front": 3, "default": 4, "top": 5,
            "overhead": 6}


@dataclass(frozen=True)
class Group:
    column: str
    dims: tuple | None
    kind: str = "absolute"


@dataclass(frozen=True)
class Camera:
    name: str
    column: str
    role: str | None = None
    view: str = "default"

    @property
    def view_id(self):
        return VIEW_IDS[self.view]


@dataclass(frozen=True)
class Proprio:
    column: str
    position: tuple = ()
    rotation: tuple = ()
    other: tuple = ()
    gripper: tuple = ()

    @property
    def dims(self):
        return self.position + self.rotation + self.other + self.gripper


@dataclass(frozen=True)
class Embodiment:
    name: str
    source: str
    action: tuple
    state: tuple
    cameras: tuple
    text: str
    proprio: Proprio | None = None
    wm_fps: float | None = None

    def assemble(self, source, groups):
        """Concatenate groups into a (total_frames, width) array and a per-dim kind list."""
        parts, kinds = [], []
        for g in groups:
            column = source.column(g.column)
            part = column if g.dims is None else column[:, list(g.dims)]
            parts.append(part)
            kinds += [g.kind] * part.shape[1]
        if not parts:
            return np.zeros((source.total_frames, 0), np.float32), []
        return np.ascontiguousarray(np.concatenate(parts, axis=1)), kinds


def _groups(entries, default_kind):
    out = []
    for e in entries or []:
        kind = e.get("kind", default_kind)
        if kind not in KINDS:
            raise ValueError(f"unknown kind '{kind}', expected one of {KINDS}")
        dims = tuple(e["dims"]) if e.get("dims") is not None else None
        out.append(Group(e["from"], dims, kind))
    return tuple(out)


def load_embodiment(path):
    text = Path(path).read_text()
    spec = yaml.safe_load(text)
    cameras = tuple(Camera(c["name"], c["from"], c.get("role"), c.get("view", "default"))
                    for c in spec["cameras"])
    if len({c.name for c in cameras}) != len(cameras):
        raise ValueError(f"{path}: duplicate camera names")
    bad = [c.view for c in cameras if c.view not in VIEW_IDS]
    if bad:
        raise ValueError(f"{path}: unknown camera views {bad}; use one of {sorted(VIEW_IDS)}")
    if not spec.get("action"):
        raise ValueError(f"{path}: an embodiment needs at least one action group")
    proprio = None
    if spec.get("proprio"):
        pr = dict(spec["proprio"])
        proprio = Proprio(pr.pop("from"), *(tuple(pr.pop(k, ()) or ()) for k in
                                            ("position", "rotation", "other", "gripper")))
        if pr:
            raise ValueError(f"{path}: proprio has unknown keys {sorted(pr)}")
        if proprio.rotation and len(proprio.rotation) != 3:
            raise ValueError(f"{path}: proprio.rotation must name 3 euler dims")
        if len(set(proprio.dims)) != len(proprio.dims) or not proprio.dims:
            raise ValueError(f"{path}: proprio dims must be non-empty and distinct")
    return Embodiment(spec["name"], spec["source"], _groups(spec["action"], "absolute"),
                      _groups(spec.get("state"), "absolute"), cameras, text, proprio,
                      float(spec["wm_fps"]) if spec.get("wm_fps") else None)
