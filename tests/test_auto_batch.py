"""find_max_batch_size exercised without a GPU: the torch.cuda memory APIs are
monkeypatched so step_fn side effects drive the reported peak fraction."""

import pytest
import torch

from thesis.utils.auto_batch import find_max_batch_size


@pytest.fixture
def fake_cuda(monkeypatch):
    state = {"peak": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: state["peak"])
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda device=None: type("Props", (), {"total_memory": 100})(),
    )
    return state


def test_binary_search_stops_below_target_fraction(fake_cuda):
    def step(bs):
        fake_cuda["peak"] = bs * 10

    assert find_max_batch_size(step, target_fraction=0.85, log=lambda *_: None) == 8


def test_oom_bounds_the_search(fake_cuda):
    def step(bs):
        if bs > 13:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        fake_cuda["peak"] = bs

    assert find_max_batch_size(step, log=lambda *_: None) == 13


def test_oom_at_batch_one_is_fatal(fake_cuda):
    def step(bs):
        raise torch.cuda.OutOfMemoryError("fake OOM")

    with pytest.raises(RuntimeError, match="batch_size=1"):
        find_max_batch_size(step, log=lambda *_: None)


def test_regression_jump_beats_bisection_probe_count(fake_cuda):
    # 0.1% VRAM per sample -> optimum 849; doubling+bisection needs ~20 probes
    calls = []

    def step(bs):
        calls.append(bs)
        fake_cuda["peak"] = bs * 0.1

    got = find_max_batch_size(step, target_fraction=0.85, log=lambda *_: None)
    assert 0.83 <= got * 0.001 < 0.85
    assert len(calls) <= 14


def test_target_cap_short_circuits(fake_cuda):
    calls = []

    def step(bs):
        calls.append(bs)
        fake_cuda["peak"] = bs * 0.1

    got = find_max_batch_size(
        step, target_fraction=0.85, target_cap=64, log=lambda *_: None
    )
    assert got == 64
    assert max(calls) <= 64
