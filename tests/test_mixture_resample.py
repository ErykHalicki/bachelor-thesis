import numpy as np
import pytest

from thesis.datasets.mixture.clock import resample


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
