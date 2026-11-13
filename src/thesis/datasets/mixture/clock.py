"""The world-model clock: which native frames a window shows, and how the actions between
two shown frames become one fixed-shape chunk.

Time is defined once, in seconds. Step k of a window sits at `t0 + k / f_wm`, and shows the
nearest native frame. The action chunk for step k is every native action taken between the
frame shown at step k and the frame shown at step k + 1, so a chunk always describes the
transition the model actually sees, even when `fps / f_wm` is not an integer and the gaps
between shown frames are uneven.

Each source gets its own chunk length `K = round(fps / f_wm)`: the native number of actions
per step. Every embodiment owns its action projection, so nothing needs K to match across
sources, and a chunk is the measured actions unchanged whenever the ratio is an integer.
Only an uneven gap is resampled to K, by kind (`resample`).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Clock:
    f_wm: float
    steps: int

    def offsets(self, fps):
        """Native-frame offset of each step from the window's first frame, shape (steps,)."""
        if fps < self.f_wm:
            raise ValueError(f"a {fps} fps source cannot fill a {self.f_wm} Hz clock")
        return np.round(np.arange(self.steps) * fps / self.f_wm).astype(np.int64)

    def substeps(self, fps):
        return max(1, int(round(fps / self.f_wm)))

    def span(self, fps):
        """Native frames from a window's first shown frame to its last."""
        return int(self.offsets(fps)[-1])

    def chunks(self, actions, kinds, start, fps):
        """`actions` is one episode's (length, D) array. Returns (steps - 1, K, D)."""
        idx = start + self.offsets(fps)
        K = self.substeps(fps)
        gaps = np.diff(idx)
        if (gaps == K).all():
            return actions[idx[0]:idx[-1]].reshape(self.steps - 1, K, -1)
        return np.stack([resample(actions[a:b], kinds, K) for a, b in zip(idx[:-1], idx[1:])])


def resample(chunk, kinds, K):
    """Resample n native actions covering one step to K evenly spaced sub-steps.

    delta     the increments are integrated, the path is read at the K + 1 sub-step
              boundaries and differenced again, so total motion over the step is exact.
    absolute  linear interpolation between targets, each sitting at the start of its interval.
    discrete  each sub-step takes the value in force when it starts.
    """
    n = len(chunk)
    if n == K:
        return chunk
    out = np.empty((K, chunk.shape[1]), chunk.dtype)
    starts = np.arange(K) * n / K
    held = np.minimum(np.floor(starts + 1e-9).astype(np.int64), n - 1)
    edges = np.arange(K + 1) * n / K
    for d, kind in enumerate(kinds):
        if kind == "delta":
            path = np.concatenate([[0.0], np.cumsum(chunk[:, d], dtype=np.float64)])
            out[:, d] = np.diff(np.interp(edges, np.arange(n + 1), path))
        elif kind == "absolute":
            out[:, d] = np.interp(starts, np.arange(n), chunk[:, d])
        else:
            out[:, d] = chunk[held, d]
    return out
