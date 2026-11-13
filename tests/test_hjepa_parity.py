"""The H-JEPA port against the reference itself (external/h-jepa): the reference's own
`create_world_model`, `hjepa_forward`, `GradientSolver` and `HierarchicalSolver`, run side by
side with ours on shared weights and seeds. Losses, gradients, encodings and plans must match.

Also checks that a released checkpoint loads strictly (skipped when not downloaded).
"""

import importlib
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT / "external" / "h-jepa"
CKPT = pathlib.Path("/mnt/hdd/hjepa/ckpts/cube/cube_hjepa_l2/seed42/cube_hjepa_l2_object.ckpt")

pytestmark = pytest.mark.skipif(not (REF / "h_jepa" / "hjepa_utils.py").exists(),
                                reason="external/h-jepa submodule not checked out")

REF_TOP_LEVEL = ("loss", "models", "hjepa_utils", "hierarchical_solver", "stable_worldmodel",
                 "gymnasium")


@pytest.fixture(scope="module")
def ref():
    """The reference modules, imported from the submodule. Its package `__init__` pulls in
    simulators, so `stable_worldmodel` is registered as a bare namespace and gymnasium (a
    type-hint-only import of the solver) as a stub."""
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in REF_TOP_LEVEL}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(REF / "h_jepa"))
    swm = types.ModuleType("stable_worldmodel")
    swm.__path__ = [str(REF / "stable_worldmodel")]
    sys.modules["stable_worldmodel"] = swm
    if importlib.util.find_spec("gymnasium") is None:
        gym = types.ModuleType("gymnasium")
        gym.Space = object
        sys.modules["gymnasium"] = gym
    try:
        yield SimpleNamespace(
            utils=importlib.import_module("hjepa_utils"),
            hsolver=importlib.import_module("hierarchical_solver"),
            gd=importlib.import_module("stable_worldmodel.solver.gd"),
        )
    finally:
        sys.path.remove(str(REF / "h_jepa"))
        for k in [k for k in sys.modules if k.split(".")[0] in REF_TOP_LEVEL]:
            del sys.modules[k]
        sys.modules.update(saved)


def _cfg(levels=2, img=64, patch=8, frameskip=5, action_dim=5, proprio=False):
    """One config in the shared schema: the reference reads it directly, ours under
    `algorithm`. Small widths keep the CPU test fast; the structure is the paper's."""
    lv1 = {
        "lr": 5e-5,
        "wm": {"history_size": 3, "rollout_n": 1, "embed_dim": 32, "use_proprio": proprio,
               "action_dim": action_dim},
        "predictor": {"depth": 2, "heads": 2, "mlp_dim": 64, "dim_head": 16, "dropout": 0.0,
                      "emb_dropout": 0.0, "hidden_dim": 32, "projector": {"type": "mlp"}},
        "loss": {"embed": {"sigreg": {"enabled": True, "weight": 0.09}}},
    }
    vit = {"type": "vit", "scale": "tiny", "num_hidden_layers": 2}
    if proprio:
        lv1["wm"].update(proprio_dim=6, proprio_emb_dim=8)
        lv1["encoder"] = {
            "projector": {"type": "mlp"},
            "pixel_encoder": {"encoder": vit, "projector": {"type": "mlp", "output_dim": 24}},
            "proprio_encoder": {"encoder": {"type": "mlp", "hidden_dim": 16},
                                "projector": {"type": "identity"}},
        }
    else:
        lv1["encoder"] = {**vit, "projector": {"type": "mlp"}}
    cfg = {"num_levels": levels, "img_size": img, "patch_size": patch, "level1": lv1,
           "frameskip": frameskip,
           "data": {"dataset": {"level1": {"frameskip": frameskip}}}}
    for n in range(2, levels + 1):
        cfg[f"level{n}"] = {
            "lr": 1e-5, "stride": 2, "window_size": 1,
            "wm": {"history_size": 3, "rollout_n": 1, "embed_dim": 32, "use_proprio": False},
            "encoder": {"type": "latent_mlp", "input_dim": 32, "hidden_dim": 48,
                        "final_ln": False, "output_dim": 32, "projector": {"type": "identity"}},
            "action_encoder": {"output_dim": 4, "input_dim": 32, "max_chunk": 3,
                               "queue_size": 64},
            "action_embed": {"patch_embed": False, "input_dim": 4},
            "predictor": {"depth": 2, "heads": 2, "mlp_dim": 64, "dim_head": 16, "dropout": 0.0,
                          "emb_dropout": 0.0, "hidden_dim": 32, "projector": {"type": "mlp"}},
            "loss": {"embed": {"sigreg": {"enabled": True, "weight": 0.09}}},
        }
    return OmegaConf.create(cfg)


def _ours(cfg, l1_steps):
    from thesis.algorithms.hjepa import HJEPAWorldModel

    fs = int(cfg.frameskip)
    algo = OmegaConf.merge(cfg, {
        "name": "hjepa",
        "fields": {"pixels": "pixels", "action": "action",
                   "proprio": "proprio" if cfg.level1.wm.use_proprio else None},
        "conditioning": {"obs": {"from": "pixels", "index": "0"},
                         **({"prop": {"from": "proprio", "index": "0"}}
                            if cfg.level1.wm.use_proprio else {})},
        "predict": {"future": {"from": "pixels", "index": f"1..{l1_steps - 1}",
                               "raw_index": f"{fs}..{(l1_steps - 1) * fs}:{fs}"},
                    "action": {"from": "action", "index": f"0..{l1_steps * fs - 1}"},
                    **({"prop_future": {"from": "proprio", "index": f"1..{l1_steps - 1}",
                                        "raw_index": f"{fs}..{(l1_steps - 1) * fs}:{fs}"}}
                       if cfg.level1.wm.use_proprio else {})},
    })
    return HJEPAWorldModel(OmegaConf.create({"algorithm": algo}).algorithm)


def _pair(ref, cfg, l1_steps):
    torch.manual_seed(0)
    ref_model, ref_losses = ref.utils.create_world_model(cfg)
    ours = _ours(cfg, l1_steps)
    ours.hjepa.load_state_dict(ref_model.state_dict(), strict=True)
    return ref_model, ref_losses, ours


def _ref_forward(ref, ref_model, ref_losses, cfg, info):
    host = SimpleNamespace(model=ref_model, log_dict=lambda *a, **k: None, **ref_losses)
    return ref.utils.hjepa_forward(host, dict(info), "train", cfg, normalize_batch=lambda b: None)


@pytest.mark.parametrize("levels,proprio", [(1, False), (2, False), (3, False), (2, True)])
def test_training_forward_matches_reference(ref, levels, proprio):
    from thesis.algorithms.hjepa import required_level1_steps

    cfg = _cfg(levels=levels, proprio=proprio)
    L1 = required_level1_steps(cfg)
    ref_model, ref_losses, ours = _pair(ref, cfg, L1)
    ref_model.train()
    ours.train()

    g = torch.Generator().manual_seed(1)
    B = 6
    info = {"pixels_level1": torch.randn(B, L1, 3, 64, 64, generator=g),
            "action_level1": torch.randn(B, L1, 25, generator=g)}
    if proprio:
        info["proprio_level1"] = torch.randn(B, L1, 6, generator=g)

    torch.manual_seed(7)
    expected = _ref_forward(ref, ref_model, ref_losses, cfg, info)
    expected["loss"].backward()
    torch.manual_seed(7)
    got = ours.forward_levels(dict(info))
    got["loss"].backward()

    keys = [k for k in expected if "loss" in k and torch.is_tensor(expected[k])
            and expected[k].ndim == 0 and k in got]
    assert {"loss", "pred_loss_level1", "sigreg_loss_level1"} <= set(keys)
    for k in keys:
        torch.testing.assert_close(got[k], expected[k], rtol=1e-5, atol=1e-6, msg=k)
    for level in range(1, levels + 1):
        torch.testing.assert_close(got[f"embed_{level}"], expected[f"embed_{level}"])
    ref_grads = dict(ref_model.named_parameters())
    for name, p in ours.hjepa.named_parameters():
        if p.grad is None:
            assert ref_grads[name].grad is None, name
            continue
        torch.testing.assert_close(p.grad, ref_grads[name].grad, rtol=1e-4, atol=1e-6, msg=name)
    for name, buf in ref_model.named_buffers():
        torch.testing.assert_close(dict(ours.hjepa.named_buffers())[name], buf, msg=name)


def test_batch_adapter_builds_the_reference_stream():
    """Raw rows -> `*_level1`: frames already at the frameskip, actions stacked (L1, fs*A)
    in the reference's row-major order, pixels ImageNet-normalized."""
    cfg = _cfg(levels=2)
    ours = _ours(cfg, 7)
    pixels = torch.randint(0, 256, (2, 7, 3, 64, 64), dtype=torch.uint8)
    actions = torch.arange(2 * 35 * 5, dtype=torch.float32).reshape(2, 35, 5)
    info = ours.level1_stream({"pixels": pixels, "action": actions})
    torch.testing.assert_close(info["action_level1"][:, 1], actions[:, 5:10].reshape(2, 25))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    torch.testing.assert_close(info["pixels_level1"], (pixels.float() / 255 - mean) / std)


class _Cfg(SimpleNamespace):
    def __getattr__(self, name):
        return None


def _plan_cfgs(levels):
    lv = [dict(horizon=3, receding_horizon=2, action_block=5, cost_last_n=2,
               action_cost_weight=0.2, pre_decay_horizon=2, num_subgoals=1,
               intermediate_cost_weight=1.0, horizon_one_goal_cost_space="lower")]
    for _ in range(2, levels + 1):
        lv.append(dict(horizon=2, receding_horizon=2, action_block=1, action_dim=4,
                       cost_last_n=1, num_subgoals=1, intermediate_cost_weight=1.0,
                       action_cost_weight=0.0, pre_decay_horizon=None,
                       horizon_one_goal_cost_space="upper"))
    return lv


@pytest.mark.parametrize("levels,budget", [(1, 50), (2, 50), (2, None), (3, 60)])
def test_hierarchical_gd_plan_matches_reference(ref, levels, budget):
    from thesis.planning import HierarchicalPlanConfig, PlanConfig

    cfg = _cfg(levels=levels)
    L1 = 4 if levels == 1 else (7 if levels == 2 else 13)
    ref_model, ref_losses, ours = _pair(ref, cfg, L1)
    ref_model.train()
    torch.manual_seed(3)
    for _ in range(3):
        _ref_forward(ref, ref_model, ref_losses, cfg,
                     {"pixels_level1": torch.randn(8, L1, 3, 64, 64),
                      "action_level1": torch.randn(8, L1, 25)})
    ours.hjepa.load_state_dict(ref_model.state_dict(), strict=True)
    ref_model.eval()
    ours.eval()

    solver_kw = dict(n_steps=6, min_its=2, num_samples=5, var_scale=1.0, action_noise=0.01,
                     device="cpu", seed=11, optimizer_cls=torch.optim.AdamW,
                     optimizer_kwargs={"lr": 0.1})
    levels_cfg = _plan_cfgs(levels)
    ref_level_models = [ref_model.get_level(i) for i in range(1, levels + 1)]
    ref_solvers = {
        i: ref.gd.GradientSolver(model=ref.hsolver.LevelCostModel(m, ref_level_models[i:]),
                                 **solver_kw)
        for i, m in enumerate(ref_level_models, start=1)
    }
    ref_solver = ref.hsolver.HierarchicalSolver(ref_solvers, ref_level_models)
    ref_cfg = _Cfg(receding_horizon=2, **{f"level{i + 1}": _Cfg(**c) for i, c in enumerate(levels_cfg)})
    ref_solver.configure(action_space=SimpleNamespace(shape=(2, 5)), n_envs=2, config=ref_cfg)

    planner = ours.make_planner({
        "eval_budget": budget, "receding_horizon": 2,
        "levels": {f"level{i + 1}": {k: v for k, v in c.items() if k not in ("receding_horizon",)}
                   for i, c in enumerate(levels_cfg)},
        "solvers": {f"level{i}": {"name": "gd", **{k: v for k, v in solver_kw.items()
                                                    if k != "optimizer_cls"},
                                  "optimizer": "AdamW"} for i in range(1, levels + 1)},
    }, n_envs=2)
    assert isinstance(planner.config, HierarchicalPlanConfig)
    assert isinstance(planner.config.level1, PlanConfig)

    g = torch.Generator().manual_seed(5)
    pixels = torch.randint(0, 256, (2, 1, 3, 64, 64), dtype=torch.uint8, generator=g)
    goal_px = torch.randint(0, 256, (2, 1, 3, 64, 64), dtype=torch.uint8, generator=g)
    info = ours.plan_inputs({"pixels": pixels}, {"pixels": goal_px})

    for steps in (0, 10, 30):
        ref_budget = budget if budget is not None else 10 ** 9
        expected = ref_solver.solve(dict(info), steps_taken=steps, eval_budget=ref_budget)
        got = ours.plan(planner, {"pixels": pixels}, {"pixels": goal_px}, steps_taken=steps)
        full = planner.last["actions"]
        torch.testing.assert_close(full, expected["actions"], rtol=1e-5, atol=1e-6)
        keep = min(2, expected["actions"].shape[1])
        torch.testing.assert_close(got, expected["actions"][:, :keep].reshape(2, keep * 5, 5))


def test_cem_and_custom_solvers_plug_in():
    """Any registered solver runs a level: CEM out of the box, and a user class through
    `register_solver` with nothing else changed."""
    from thesis.planning import register_solver
    from thesis.planning.gd import PriorSolverMixin

    cfg = _cfg(levels=2)
    ours = _ours(cfg, 7)
    ours.train()
    ours.loss({"pixels": torch.randint(0, 256, (4, 7, 3, 64, 64), dtype=torch.uint8),
               "action": torch.randn(4, 35, 5)})
    ours.eval()

    @register_solver("first_sample_test")
    class FirstSample(PriorSolverMixin):
        def __init__(self, model, device="cpu"):
            self.model, self.device, self.action_clip_sigma = model, device, None

        @torch.no_grad()
        def solve(self, info, init_action=None, planning_horizon=None, steps_taken=None,
                  eval_budget=None):
            h = self._resolve_horizon(planning_horizon, steps_taken, eval_budget)
            a = torch.zeros(self.n_envs, h, self.action_dim)
            return {"actions": a, "predictions": self._rollout_final_actions(info, a)}

    for solvers in ({"level1": {"name": "cem", "n_steps": 3, "num_samples": 16, "topk": 4},
                     "level2": {"name": "cem", "n_steps": 3, "num_samples": 16, "topk": 4}},
                    {"level1": {"name": "gd", "n_steps": 2, "num_samples": 4},
                     "level2": {"name": "first_sample_test"}}):
        planner = ours.make_planner({"solvers": solvers})
        px = torch.randint(0, 256, (3, 1, 3, 64, 64), dtype=torch.uint8)
        chunk = ours.plan(planner, {"pixels": px}, {"pixels": px[:1]})
        assert chunk.shape == (3, ours.execute_len, 5)
        assert torch.isfinite(chunk).all()


@pytest.mark.skipif(not CKPT.exists(), reason="released cube_hjepa_l2 checkpoint not downloaded")
def test_released_checkpoint_loads_into_our_model(ref):
    """The paper's own weights load strictly into the port, and the port encodes and plans
    with them exactly as the reference model built from the same tensors does."""
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    from thesis.algorithms.hjepa import HJEPAWorldModel

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        full = compose(config_name="base", return_hydra_config=True,
                       overrides=["run=hjepa_b601_pnpt"])
    HydraConfig.instance().set_config(full)
    for key, value in {"frameskip": 5, "level1.wm.action_dim": 5,
                       "predict.scene_future.raw_index": "5..30:5",
                       "predict.action.index": "0..34"}.items():
        OmegaConf.update(full, f"algorithm.{key}", value)
    algo = full.algorithm
    ours = HJEPAWorldModel(algo)
    ours.load_reference_checkpoint(CKPT)
    ref_cfg = OmegaConf.merge(OmegaConf.to_container(algo, resolve=True),
                              {"data": {"dataset": {"level1": {"frameskip": 5}}}})
    torch.manual_seed(0)
    ref_model, _ = ref.utils.create_world_model(ref_cfg)
    ref_model.load_state_dict(ours.hjepa.state_dict(), strict=True)
    assert ours.hjepa.get_level(2).get_latent_action_queue().shape == (4000, 4)
    ref_model.eval()
    ours.eval()

    g = torch.Generator().manual_seed(2)
    px = torch.randint(0, 256, (2, 1, 3, 224, 224), dtype=torch.uint8, generator=g)
    goal = torch.randint(0, 256, (2, 1, 3, 224, 224), dtype=torch.uint8, generator=g)
    info = ours.plan_inputs({ours.field: px}, {ours.field: goal})
    level_models = [ref_model.get_level(i) for i in (1, 2)]
    ref_solver = ref.hsolver.HierarchicalSolver({}, level_models)
    with torch.no_grad():
        expected = ref_solver._encode_all_levels(dict(info), "pixels")
    got = ours.encode_levels(dict(info), "pixels")
    for k in expected:
        torch.testing.assert_close(got[k], expected[k], rtol=1e-5, atol=1e-5)
