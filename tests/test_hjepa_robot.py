"""H-JEPA on the robot path: the same lerobot eval loop and inference server a flow policy
uses, driven through `PlanDriver` toward a goal image, locally and over the wire."""

import cv2
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_lerobot_eval import FakeRobot, _cfg, run_eval
from test_remote_inference import Server

from thesis.scripts import serve


def _model(levels=2):
    from thesis.algorithms.hjepa import HJEPAWorldModel

    fs = 2
    L1 = 4 if levels == 1 else 7
    lvl = {"lr": 1e-4, "wm": {"history_size": 3, "rollout_n": 1, "embed_dim": 16},
           "predictor": {"depth": 1, "heads": 2, "mlp_dim": 32, "dim_head": 8, "dropout": 0.0,
                         "emb_dropout": 0.0, "hidden_dim": 16, "projector": {"type": "mlp"}},
           "loss": {"embed": {"sigreg": {"enabled": True, "weight": 0.09}}}}
    cfg = {
        "name": "hjepa", "frameskip": fs, "img_size": 32, "patch_size": 8, "num_levels": levels,
        "fields": {"pixels": "observation.images.cam", "action": "action"},
        "level1": {**lvl, "wm": {**lvl["wm"], "action_dim": 2},
                   "encoder": {"type": "vit", "scale": "tiny", "num_hidden_layers": 1,
                               "projector": {"type": "mlp"}}},
        "conditioning": {"obs": {"from": "observation.images.cam", "index": "0"}},
        "predict": {"future": {"from": "observation.images.cam", "index": f"1..{L1 - 1}",
                               "raw_index": f"{fs}..{(L1 - 1) * fs}:{fs}"},
                    "action": {"from": "action", "index": f"0..{L1 * fs - 1}"}},
        "plan": {"receding_horizon": 2,
                 "levels": {"level1": {"horizon": 3, "cost_last_n": 2},
                            "level2": {"horizon": 2}},
                 "solvers": {"level1": {"name": "gd", "n_steps": 3, "num_samples": 4, "seed": 1},
                             "level2": {"name": "cem", "n_steps": 2, "num_samples": 8,
                                        "topk": 2, "seed": 1}}},
    }
    if levels > 1:
        cfg["level2"] = {**lvl, "stride": 2, "window_size": 1,
                         "encoder": {"type": "latent_mlp", "input_dim": 16, "hidden_dim": 16,
                                     "final_ln": False, "output_dim": 16,
                                     "projector": {"type": "identity"}},
                         "action_encoder": {"output_dim": 3, "input_dim": 16, "max_chunk": 3,
                                            "queue_size": 32},
                         "action_embed": {"patch_embed": False, "input_dim": 3}}
    torch.manual_seed(0)
    model = HJEPAWorldModel(OmegaConf.create({"algorithm": cfg}).algorithm)
    model.loss({"observation.images.cam": torch.randint(0, 256, (4, L1, 3, 32, 32), dtype=torch.uint8),
                "action": torch.randn(4, L1 * fs, 2)})
    model.norm_stats = {"action": {"mean": [0.5, -0.5], "std": [2.0, 2.0]}}
    model.norm_method = "mean_std"
    return model.eval()


@pytest.fixture
def goal_png(tmp_path):
    path = tmp_path / "goal.png"
    cv2.imwrite(str(path), np.full((48, 64, 3), 128, dtype=np.uint8))
    return str(path)


@pytest.mark.parametrize("levels", [1, 2])
def test_world_model_drives_the_robot_toward_a_goal(monkeypatch, goal_png, levels):
    model = _model(levels)
    robot = FakeRobot()
    result = run_eval(_cfg(goal_image=goal_png), robot, model, monkeypatch, ["", "y", "", ""])
    assert len(robot.sent) == 12
    for action in robot.sent:
        assert set(action) == {"j1.pos", "j2.pos"} or set(action) == {"j1", "j2"}
        assert all(np.isfinite(v) for v in action.values())
    assert "success_rate" in result.metrics


def test_a_world_model_without_a_goal_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="goal_image"):
        run_eval(_cfg(), FakeRobot(), _model(), monkeypatch, ["", "y", "", ""])


def test_remote_plan_matches_local(monkeypatch, goal_png):
    """Served from scripts/serve.py, the planner sends the robot exactly what it sends
    in-process: same goal, same solver seeds, same horizons."""
    local = FakeRobot()
    run_eval(_cfg(goal_image=goal_png), local, _model(), monkeypatch, ["", "y", "", ""])

    inst = Server(_model())
    monkeypatch.setattr(serve, "load_model", inst.load_model)
    try:
        remote = FakeRobot()
        run_eval(_cfg(goal_image=goal_png, server=inst.address, run="me/proj/hjepa"),
                 remote, None, monkeypatch, ["", "y", "", ""])
    finally:
        inst.close()
    assert len(remote.sent) == len(local.sent) == 12
    for a, b in zip(remote.sent, local.sent, strict=True):
        assert a.keys() == b.keys()
        np.testing.assert_allclose([a[k] for k in a], [b[k] for k in b], rtol=1e-5, atol=1e-5)


def test_plan_knobs_pick_the_solver_per_eval(monkeypatch, goal_png):
    """`eval.plan` swaps solvers without touching the model: here level 1 runs CEM."""
    model = _model()
    seen = []
    from thesis.planning import cem

    real = cem.CEMSolver.solve

    def spy(self, *a, **k):
        seen.append(self.level)
        return real(self, *a, **k)

    monkeypatch.setattr(cem.CEMSolver, "solve", spy)
    plan = {"solvers": {"level1": {"name": "cem", "n_steps": 2, "num_samples": 8, "topk": 2}}}
    run_eval(_cfg(goal_image=goal_png, plan=plan), FakeRobot(), model, monkeypatch,
             ["", "y", "", ""])
    assert 1 in seen and 2 in seen


def test_naming_another_solver_replaces_the_spec():
    """An eval naming a different solver gets that solver's knobs only, not the model's
    gradient-descent ones merged in; the same name merges."""
    model = _model()
    plan = model.plan_config({"solvers": {"level1": {"name": "cem", "topk": 3}}})
    assert plan["solvers"]["level1"] == {"name": "cem", "topk": 3}
    plan = model.plan_config({"solvers": {"level1": {"n_steps": 9}}})
    assert plan["solvers"]["level1"]["name"] == "gd" and plan["solvers"]["level1"]["n_steps"] == 9


def test_budgeted_horizons_decay_across_an_episode(monkeypatch, goal_png):
    """With `eval_budget` the driver's step count reaches the planner, so horizons shrink
    as the episode runs out, and reset with the next episode."""
    model = _model()
    horizons = []
    from thesis.planning import gd

    real = gd.GradientSolver.solve

    def spy(self, info, init_action=None, planning_horizon=None, **k):
        horizons.append(planning_horizon)
        return real(self, info, init_action, planning_horizon, **k)

    monkeypatch.setattr(gd.GradientSolver, "solve", spy)
    plan = {"eval_budget": 12, "levels": {"level2": {"horizon": 1}}}
    run_eval(_cfg(goal_image=goal_png, plan=plan), FakeRobot(), model, monkeypatch,
             ["", "y", "", ""])
    assert horizons == [3, 2, 3, 2]


def test_goal_distance_rolls_level1_over_primitive_actions():
    """At init AdaLN-zero makes the predictor ignore actions, so the modulation is perturbed
    here to give the actions something to change."""
    model = _model()
    torch.manual_seed(0)
    for name, p in model.hjepa.named_parameters():
        if "adaLN_modulation" in name:
            p.data.normal_(0, 0.05)
    px = torch.randint(0, 256, (3, 1, 3, 32, 32), dtype=torch.uint8)
    obs, goal = {model.field: px}, {model.field: px.flip(0)}
    d = model.goal_distance(obs, goal, torch.zeros(3, 6 * model.frameskip, 2))
    assert d.shape == (3,) and torch.isfinite(d).all()
    assert not torch.allclose(d, model.goal_distance(obs, goal, torch.ones(3, 12, 2)))
