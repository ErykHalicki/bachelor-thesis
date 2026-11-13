import numpy as np
import pytest

from thesis.datasets.mixture.clock import Clock, resample


def test_offsets_follow_seconds_not_frames():
    clock = Clock(f_wm=2.5, steps=5)
    assert clock.offsets(5).tolist() == [0, 2, 4, 6, 8]
    assert clock.offsets(30).tolist() == [0, 12, 24, 36, 48]
    assert clock.substeps(5) == 2 and clock.substeps(30) == 12


def test_uneven_ratio_snaps_to_nearest_frame():
    clock = Clock(f_wm=3.0, steps=4)
    assert clock.offsets(5).tolist() == [0, 2, 3, 5]
    assert clock.substeps(5) == 2


def test_source_slower_than_clock_is_rejected():
    with pytest.raises(ValueError):
        Clock(f_wm=3.0, steps=4).offsets(1)


def test_integer_ratio_chunks_are_the_native_actions():
    clock = Clock(f_wm=2.5, steps=4)
    actions = np.arange(40, dtype=np.float32).reshape(20, 2)
    chunks = clock.chunks(actions, ["delta", "absolute"], start=3, fps=10)
    assert chunks.shape == (3, 4, 2)
    np.testing.assert_array_equal(chunks.reshape(12, 2), actions[3:15])


def test_uneven_chunks_cover_the_transition_between_shown_frames():
    clock = Clock(f_wm=3.0, steps=4)
    actions = np.random.default_rng(0).normal(size=(12, 1)).astype(np.float32)
    chunks = clock.chunks(actions, ["delta"], start=1, fps=5)
    shown = 1 + clock.offsets(5)
    assert chunks.shape == (3, 2, 1)
    for k in range(3):
        np.testing.assert_allclose(chunks[k].sum(), actions[shown[k]:shown[k + 1]].sum(), rtol=1e-5)


@pytest.mark.parametrize("n,K", [(1, 4), (3, 2), (5, 3), (2, 7)])
def test_delta_resampling_preserves_total_motion(n, K):
    chunk = np.random.default_rng(n * 10 + K).normal(size=(n, 3))
    out = resample(chunk, ["delta"] * 3, K)
    assert out.shape == (K, 3)
    np.testing.assert_allclose(out.sum(0), chunk.sum(0), atol=1e-9)


def test_absolute_resampling_matches_source_at_native_times():
    chunk = np.random.default_rng(1).normal(size=(3, 2))
    out = resample(chunk, ["absolute"] * 2, 6)
    np.testing.assert_allclose(out[::2], chunk)
    np.testing.assert_allclose(out[1], (chunk[0] + chunk[1]) / 2)


def test_discrete_resampling_holds_values():
    chunk = np.array([[0.0], [1.0]])
    assert resample(chunk, ["discrete"], 4)[:, 0].tolist() == [0, 0, 1, 1]
    assert resample(chunk, ["discrete"], 1)[:, 0].tolist() == [0]
