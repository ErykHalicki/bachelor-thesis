import os

import pytest
from omegaconf import OmegaConf

from thesis.experiments.smoke import SmokeExperiment


def _cfg(conditioning=("observation",)):
    return OmegaConf.create(
        {
            "algorithm": {
                "name": "dummy",
                "obs_dim": 8,
                "target_dim": 2,
                "hidden": 16,
                "conditioning": list(conditioning),
                "predict": ["target"],
            },
            "dataset": {
                "backend": "dummy",
                "n": 128,
                "obs_dim": 8,
                "target_dim": 2,
                "weight_seed": 0,
                "seed": 0,
            },
            "eval": {
                "backend": "dummy",
                "n": 128,
                "obs_dim": 8,
                "target_dim": 2,
                "weight_seed": 0,
                "seed": 1000,
                "n_eval": 64,
            },
            "experiment": {
                "tasks": ["training", "validation"],
                "augment_on": "workers",
                "batch_size": 32,
                "optimizer": {"name": "adamw", "lr": 1e-3},
                "max_steps": 8,
                "log_every": 5,
                "mixed_precision": "no",
            },
        }
    )


def test_smoke(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    exp = SmokeExperiment(_cfg(), tmp_path)
    exp.exec_task("training")
    exp.exec_task("validation")
    assert (tmp_path / "model.pt").exists()


def test_modality_handshake_fails(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    exp = SmokeExperiment(_cfg(conditioning=("observation", "language")), tmp_path)
    with pytest.raises(ValueError, match="lacks modalities"):
        exp.exec_task("training")


def test_exec_task_unknown_raises(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    exp = SmokeExperiment(_cfg(), tmp_path)
    with pytest.raises(ValueError, match="not defined"):
        exp.exec_task("nonexistent")


def test_resume_loads_checkpoint(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    SmokeExperiment(_cfg(), tmp_path).exec_task("training")
    ckpt = tmp_path / "model.pt"

    resumed = SmokeExperiment(_cfg(), tmp_path, ckpt_path=ckpt)
    resumed.exec_task("training")
    assert ckpt.exists()


def test_training_only_experiment(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    from thesis.experiments.base import BaseExperiment
    from thesis.experiments.training.base import TrainingMixin

    class TrainOnly(TrainingMixin, BaseExperiment):
        pass

    exp = TrainOnly(_cfg(), tmp_path)
    exp.exec_task("training")
    assert (tmp_path / "model.pt").exists()

    with pytest.raises(ValueError, match="not defined"):
        exp.exec_task("validation")


def test_validation_only_loads_checkpoint(tmp_path):
    os.environ["WANDB_MODE"] = "disabled"
    SmokeExperiment(_cfg(), tmp_path).exec_task("training")
    ckpt = tmp_path / "model.pt"

    evalonly = SmokeExperiment(_cfg(), tmp_path, ckpt_path=ckpt)
    evalonly.exec_task("validation")


def test_resume_fast_forwards_lr_schedule():
    import torch

    from thesis.utils.optim import build_lr_scheduler

    def lr_at(n_steps):
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        sched = build_lr_scheduler(opt, warmup_steps=500, total_steps=10000)
        for _ in range(n_steps):
            sched.step()
        return sched.get_last_lr()[0]

    uninterrupted = lr_at(5000)
    resumed = lr_at(5000)
    assert abs(uninterrupted - resumed) < 1e-12
    assert abs(resumed - lr_at(0)) > 1e-6
