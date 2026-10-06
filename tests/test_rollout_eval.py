"""Post-hoc eval pieces (main.py's load= + eval-only mode), exercised offline: the
validation task captures the EvalResult + checkpoint step for the write-back, and the
wandb write-back degrades to stdout when disabled. main.py's glue on top (config fetch,
run linking) is wandb-network code, exercised manually like ckpt_utils.
"""

import torch
from omegaconf import OmegaConf

from thesis.experiments.smoke import SmokeExperiment
from thesis.utils.checkpoint import save_checkpoint


def _cfg():
    dataset = {
        "backend": "dummy", "n": 64, "obs_dim": 8, "target_dim": 2,
        "weight_seed": 0, "seed": 0,
    }
    return OmegaConf.create({
        "algorithm": {
            "name": "dummy", "obs_dim": 8, "target_dim": 2, "hidden": 16,
            "conditioning": ["observation"], "predict": ["target"],
        },
        "dataset": dataset,
        "eval": {**dataset, "seed": 1, "n_eval": 32},
        "experiment": {"tasks": ["validation"]},
    })


def test_validation_captures_result_and_step(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    cfg = _cfg()

    from thesis.algorithms import build_algorithm
    algo = build_algorithm(cfg.algorithm)
    opt = torch.optim.SGD(algo.parameters(), lr=0.0)
    ckpt_path = save_checkpoint(algo, opt, tmp_path, step=7)

    exp = SmokeExperiment(cfg, tmp_path, ckpt_path=ckpt_path)
    exp.validation()

    assert "mse" in exp.last_eval_result.metrics
    assert exp.last_eval_step == 7


def test_resumed_eval_run_id_comes_from_the_rollout_dir(tmp_path):
    """main.py reopens the eval run an interrupted session left, so both sittings score one
    run. The rollout dir is named explicitly by eval.resume, never sniffed."""
    import json
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from main import _resumed_eval_run_id

    (tmp_path / "eval_session.json").write_text(
        json.dumps({"wandb_run_id": "9q9o9bc3", "episodes": [{"episode": 0}]})
    )
    cfg = OmegaConf.create({"eval": {"resume": str(tmp_path)}})
    assert _resumed_eval_run_id(cfg) == "9q9o9bc3"

    assert _resumed_eval_run_id(OmegaConf.create({"eval": {"resume": None}})) is None
