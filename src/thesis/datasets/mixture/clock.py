"""Resampling a step's native actions to a fixed number of sub-steps.

A world-model step spans every native action between the frame shown at that step and the
one shown at the next (clips.py, `actions: commanded`). With K = round(fps / wm_fps)
sub-steps per step, a gap of exactly K native actions is kept as recorded; an uneven gap
(a non-integer rate ratio) is resampled to K by each dim's kind.
"""

import numpy as np


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
