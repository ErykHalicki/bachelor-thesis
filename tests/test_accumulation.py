"""Micro-batch fitting: feed effective_batch with the fewest accumulation windows,
shrinking the micro-batch so the windows land on the target instead of overshooting."""

import math

from thesis.experiments.training.base import TrainingMixin

fit = TrainingMixin._fit_micro_batch


def test_shrinks_micro_batch_onto_the_target():
    assert fit(120, 256) == 86


def test_never_adds_a_window():
    for safe in range(1, 600, 7):
        for target in (1, 64, 256, 512):
            fitted = fit(safe, target)
            assert fitted <= safe
            assert math.ceil(target / fitted) == math.ceil(target / max(1, min(safe, target)))
            assert fitted * math.ceil(target / fitted) >= target


def test_large_safe_batch_caps_at_target():
    assert fit(539, 512) == 512
    assert fit(256, 256) == 256
