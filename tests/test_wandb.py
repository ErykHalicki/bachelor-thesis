import pytest
from omegaconf import OmegaConf

from thesis.experiments.eval.base import EvalResult
from thesis.utils.wandb import (
    finish_wandb,
    init_wandb,
    log_eval,
    log_metrics,
    log_model_summary,
    wandb_enabled,
)


@pytest.fixture(autouse=True)
def _reset_wandb():
    # wandb keeps a process-global service that caches WANDB_DIR from the first init,
    # so a later test would write into an earlier test's torn-down tmp_path
    yield
    try:
        import wandb

        wandb.teardown()
    except Exception:
        pass


def test_disabled_fallback(monkeypatch, capsys):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert wandb_enabled() is False

    init_wandb(OmegaConf.create({"wandb": {"mode": "disabled"}}))
    log_metrics({"train/loss": 1.0})
    log_eval(EvalResult(metrics={"mse": 0.5}))
    finish_wandb()

    out = capsys.readouterr().out
    assert "train/loss" in out
    assert "mse" in out


def test_log_model_summary_disabled(monkeypatch, capsys):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    from omegaconf import OmegaConf

    from thesis.algorithms import build_algorithm

    model = build_algorithm(
        OmegaConf.create({"name": "dummy", "obs_dim": 8, "target_dim": 2, "hidden": 16})
    )
    log_model_summary(model)

    out = capsys.readouterr().out
    assert "params/total" in out
    assert "hidden_dim" in out


def _offline_env(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")


def _captured_init(monkeypatch):
    pytest.importorskip("wandb")
    import wandb

    monkeypatch.setenv("WANDB_MODE", "offline")
    captured = {}
    monkeypatch.setattr(wandb, "init", lambda **kw: captured.update(kw))
    return captured


def test_posthoc_eval_is_named_after_the_training_run(monkeypatch):
    captured = _captured_init(monkeypatch)
    init_wandb(OmegaConf.create({"wandb": {"project": "p"}, "name": None}),
               eval_source="htx3t7td", source_name="pnpt_policy_frozen")

    assert captured["name"] == "eval-pnpt_policy_frozen"
    assert captured["group"] == "htx3t7td"
    assert captured["id"] is None and captured["resume"] is None


def test_resumed_eval_reattaches_to_the_run_it_left(monkeypatch):
    captured = _captured_init(monkeypatch)
    init_wandb(OmegaConf.create({"wandb": {"project": "p"}, "name": None}),
               eval_source="htx3t7td", source_name="pnpt_policy_frozen",
               resume_id="9q9o9bc3")

    assert captured["id"] == "9q9o9bc3" and captured["resume"] == "must"
    # the run keeps the name it was opened with; passing one would rename it
    assert captured["name"] is None


def test_offline_logging(monkeypatch, tmp_path):
    pytest.importorskip("wandb")
    _offline_env(monkeypatch, tmp_path)
    assert wandb_enabled() is True

    cfg = OmegaConf.create({"experiment": {"project": "test"}})
    init_wandb(cfg)
    log_metrics({"train/loss": 1.0}, step=0)
    log_eval(EvalResult(metrics={"mse": 0.5}), step=1)
    finish_wandb()

    assert list(tmp_path.glob("wandb/offline-run-*")), "no offline wandb run was created"


def test_log_eval_with_video_and_episodes(monkeypatch, tmp_path):
    pytest.importorskip("wandb")
    pytest.importorskip("moviepy")
    import numpy as np

    _offline_env(monkeypatch, tmp_path)

    init_wandb(OmegaConf.create({"experiment": {"project": "test"}}))
    result = EvalResult(
        metrics={"success_rate": 0.5},
        videos={"rollout": np.random.randint(0, 255, (4, 3, 16, 16), dtype=np.uint8)},
        episodes=[{"episode": 0, "success": 1}, {"episode": 1, "success": 0}],
    )
    log_eval(result, step=0)
    finish_wandb()

    assert list(tmp_path.glob("wandb/offline-run-*")), "no offline wandb run was created"


def test_experiment_logs_to_offline_wandb(monkeypatch, tmp_path):
    pytest.importorskip("wandb")
    _offline_env(monkeypatch, tmp_path)

    from thesis.experiments.smoke import SmokeExperiment

    cfg = OmegaConf.create(
        {
            "algorithm": {
                "name": "dummy",
                "obs_dim": 8,
                "target_dim": 2,
                "hidden": 16,
                "conditioning": ["observation"],
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
            "experiment": {
                "augment_on": "workers",
                "tasks": ["training"],
                "batch_size": 32,
                "optimizer": {"name": "adamw", "lr": 1e-2},
                "max_steps": 12,
                "log_every": 5,
                "mixed_precision": "no",
            },
            "wandb": {"entity": None, "project": "test", "mode": "offline"},
        }
    )
    init_wandb(cfg)
    SmokeExperiment(cfg, tmp_path / "runs").exec_task("training")
    finish_wandb()

    assert list(tmp_path.glob("wandb/offline-run-*")), "experiment did not create a wandb run"
