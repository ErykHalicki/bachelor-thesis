"""experiment.effective_batch (micro-batch accumulation) and experiment.gradient_checkpointing."""

import os

import torch
from omegaconf import OmegaConf

from thesis.algorithms import build_algorithm
from thesis.algorithms.dummy import DummyAlgorithm
from thesis.experiments.smoke import SmokeExperiment
from thesis.experiments.training.base import TrainingMixin

from test_generic_vit_predictor import lewm_cfg
from test_smoke import _cfg


class FakeAcc:
    def __init__(self, num_processes=1):
        self.num_processes = num_processes
        self.printed = []

    def print(self, *args):
        self.printed.append(" ".join(str(a) for a in args))


def accum(batch_size, effective_batch, num_processes=1):
    return TrainingMixin._accumulation_steps(FakeAcc(num_processes), batch_size, effective_batch)


def test_unset_effective_batch_steps_every_micro_batch():
    assert accum(64, None) == 1
    assert accum(64, 0) == 1


def test_steps_round_up_to_reach_the_target():
    assert accum(64, 256) == 4
    assert accum(48, 256) == 6
    assert accum(256, 256) == 1


def test_ranks_count_towards_the_effective_batch():
    assert accum(64, 256, num_processes=4) == 1
    assert accum(64, 256, num_processes=2) == 2


def test_micro_batch_larger_than_target_warns_and_does_not_accumulate():
    acc = FakeAcc()
    assert TrainingMixin._accumulation_steps(acc, 512, 256) == 1
    assert any("warning" in line for line in acc.printed)


def _run(tmp_path, monkeypatch, **experiment):
    """Train the smoke experiment, counting optimizer steps and micro-batches."""
    os.environ["WANDB_MODE"] = "disabled"
    cfg = _cfg()
    cfg.experiment.update(experiment)

    steps = {"opt": 0, "micro": 0}

    def counted(name, fn):
        def wrapper(self, *args, **kwargs):
            steps[name] += 1
            return fn(self, *args, **kwargs)
        return wrapper

    monkeypatch.setattr(torch.optim.AdamW, "step", counted("opt", torch.optim.AdamW.step))
    monkeypatch.setattr(DummyAlgorithm, "loss", counted("micro", DummyAlgorithm.loss))
    SmokeExperiment(cfg, tmp_path).exec_task("training")
    return steps


def test_max_steps_counts_optimizer_steps_not_micro_batches(tmp_path, monkeypatch):
    steps = _run(tmp_path, monkeypatch, batch_size=8, max_steps=6, effective_batch=32)
    assert steps["opt"] == 6
    assert steps["micro"] == 24


def test_no_accumulation_without_effective_batch(tmp_path, monkeypatch):
    steps = _run(tmp_path, monkeypatch, batch_size=8, max_steps=6)
    assert steps["opt"] == steps["micro"] == 6


def test_accumulated_gradient_matches_the_full_batch():
    """Four quarter-batches scaled by 1/accum sum to the whole batch's gradient."""
    torch.manual_seed(0)
    cfg = OmegaConf.create({"name": "dummy", "obs_dim": 8, "target_dim": 2, "hidden": 16,
                            "conditioning": ["observation"], "predict": ["target"]})
    algo = build_algorithm(cfg)
    batch = {"observation": torch.randn(32, 8), "target": torch.randn(32, 2)}

    algo.loss(batch)["loss"].backward()
    full = [p.grad.clone() for p in algo.parameters()]

    algo.zero_grad()
    for i in range(4):
        micro = {k: v[i * 8:(i + 1) * 8] for k, v in batch.items()}
        (algo.loss(micro)["loss"] / 4).backward()

    for expected, got in zip(full, (p.grad for p in algo.parameters())):
        assert torch.allclose(expected, got, atol=1e-6)


def test_gradient_checkpointing_reaches_trunk_and_encoders():
    algo = build_algorithm(lewm_cfg())
    algo.set_gradient_checkpointing(True)
    assert algo.predictor.gradient_checkpointing
    assert algo.encoders["vit"].gradient_checkpointing

    algo.set_gradient_checkpointing(False)
    assert not algo.predictor.gradient_checkpointing
    assert not algo.encoders["vit"].gradient_checkpointing


def test_checkpointed_forward_matches_the_plain_one():
    torch.manual_seed(0)
    algo = build_algorithm(lewm_cfg())
    batch = {
        "observation.images.pixels": (torch.rand(2, 4, 3, 96, 96) * 255).to(torch.uint8),
        "action": torch.randn(2, 3, 2),
    }
    algo.train()

    torch.manual_seed(1)
    plain = algo.loss(batch)["loss"]
    algo.set_gradient_checkpointing(True)
    torch.manual_seed(1)
    checkpointed = algo.loss(batch)["loss"]

    assert torch.allclose(plain, checkpointed, atol=1e-5)
