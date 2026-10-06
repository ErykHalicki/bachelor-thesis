"""BatchDriver drives N envs exactly as N serial BaseDrivers would.

A vectorized RoboCasa eval exists to keep the GPU busy while MuJoCo works, which is only
worth anything if the actions it produces are the ones the serial rollout would have
produced -- otherwise the success rate stops being comparable to arms measured serially.
The fake predict path here encodes its inputs into its output, so any divergence in
history padding, replan timing, or the executed-action window shows up as a number.
"""

import numpy as np

from thesis.experiments.eval.chunking import BaseDriver, BatchDriver

COLUMNS = ["observation.state"]
OFFSETS = [-2, 0]
EXECUTE = 3
ACTION_OFFSETS = [-2, -1]
ACTION_DIM = 4


def _chunk_from(window, executed):
    seed = float(np.sum([frame["observation.state"] for frame in window]))
    past = float(np.sum([a for a in executed if a is not None])) if executed else 0.0
    return np.stack([np.full(ACTION_DIM, seed + past + i, dtype=np.float32)
                     for i in range(EXECUTE)])


class _FakeSerial(BaseDriver):
    def __init__(self):
        self._configure(COLUMNS, OFFSETS, EXECUTE, ACTION_OFFSETS)

    def predict_window(self, window, actions=()):
        return _chunk_from(window, actions)


class _FakeBatchable:
    """The attribute surface BatchDriver reads off a ChunkDriver."""

    columns, offsets, obs_len = COLUMNS, OFFSETS, 1 - OFFSETS[0]
    execute_len, action_offsets, action_len = EXECUTE, ACTION_OFFSETS, -ACTION_OFFSETS[0]

    def predict_windows(self, windows, actions_per_env=None):
        actions_per_env = list(actions_per_env or [()] * len(windows))
        return [_chunk_from(w, a) for w, a in zip(windows, actions_per_env, strict=True)]


def _streams(num_envs, ticks, seed=0):
    rng = np.random.default_rng(seed)
    return [[{"observation.state": rng.normal(size=3).astype(np.float32)}
             for _ in range(ticks)] for _ in range(num_envs)]


def test_every_env_matches_a_serial_driver():
    num_envs, ticks = 4, 11
    streams = _streams(num_envs, ticks)

    drivers = [_FakeSerial() for _ in range(num_envs)]
    serial = np.stack(
        [[d.step(f) for f in stream] for d, stream in zip(drivers, streams, strict=True)],
        axis=1,
    )
    batch = BatchDriver(_FakeBatchable(), num_envs)
    batched = np.stack(
        [batch.step([streams[e][t] for e in range(num_envs)]) for t in range(ticks)]
    )
    assert np.array_equal(serial, batched)


def test_a_replan_covers_every_env_in_one_call():
    batch = BatchDriver(_FakeBatchable(), 2)
    streams = _streams(2, 7)
    calls = []
    inner = batch.driver.predict_windows
    batch.driver.predict_windows = lambda w, a=None: (calls.append(len(w)), inner(w, a))[1]
    for tick in range(7):
        batch.step([streams[e][tick] for e in range(2)])
    assert calls == [2, 2, 2]


def test_a_serial_only_driver_is_refused():
    class Remote:
        columns, offsets, obs_len = COLUMNS, OFFSETS, 3
        execute_len, action_offsets, action_len = EXECUTE, ACTION_OFFSETS, 2

    try:
        BatchDriver(Remote(), 2)
    except TypeError as err:
        assert "predict_windows" in str(err)
    else:
        raise AssertionError("a driver without predict_windows must not be batched")
