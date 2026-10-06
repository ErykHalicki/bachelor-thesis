import pytest
import torch

scipy = pytest.importorskip("scipy")

from thesis.utils.smoothing import smooth_actions


def test_reduces_jitter():
    torch.manual_seed(0)
    t = torch.linspace(0, 3.14, 32)
    clean = torch.sin(t)[None, :, None].expand(1, 32, 2)
    noisy = clean + 0.3 * torch.randn(1, 32, 2)

    smoothed = smooth_actions(noisy)
    tv = lambda x: (x[:, 1:] - x[:, :-1]).abs().sum()
    assert tv(smoothed) < tv(noisy)


def test_short_chunk_window_clamped():
    actions = torch.randn(1, 4, 3)
    assert smooth_actions(actions).shape == (1, 4, 3)
