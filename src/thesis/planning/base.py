"""The planning contract shared by every world model and every solver.

A world model plans by exposing one `CostModel` per level, and a solver searches action
space against it. Neither knows how the other works, so a new search (CEM, gradient
descent, MCTS, ...) is a class registered here, and a new world model is a `CostModel`:

  CostModel
    level                       1 for primitive actions; >1 for latent macro-actions
    rollout(info, actions)      actions (B, S, T, A) -> info["predicted_embed_0"] (B, S, T+1, D)
    get_cost(info, actions)     -> (B, S) cost, differentiable in `actions` for gradient solvers
    get_latent_action_queue()   (N, A) latent actions seen in training, or None; a solver's
                                prior over macro-actions

  Solver
    configure(action_dim=, n_envs=, config=, level=)   once per episode setup
    solve(info, init_action=None, planning_horizon=None, steps_taken=None, eval_budget=None)
        -> {"actions": (n_envs, H, action_dim * action_block),
            "predictions": {"predicted_embed_0": (n_envs, H + 1, D), ...}}

`info` carries the encoded observation as `embed_0` (B, 1, D) and the goal as
`goal_embed_0`; solvers treat everything else in it as opaque and pass it through.
Solvers are built by name from a plain dict: `build_solver({"name": "cem", ...}, model)`.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class CostModel(Protocol):
    level: int

    def rollout(self, info_dict: dict, action_candidates: torch.Tensor) -> dict: ...

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor: ...

    def get_latent_action_queue(self) -> torch.Tensor | None: ...


@runtime_checkable
class Solver(Protocol):
    def configure(self, *, action_dim: int, n_envs: int, config: Any, level: int = 1) -> None: ...

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None,
              planning_horizon: int | None = None, steps_taken: int | None = None,
              eval_budget: int | None = None) -> dict: ...


@dataclass(frozen=True)
class PlanConfig:
    """One level's MPC knobs (stable_worldmodel.policy.PlanConfig).

    horizon          planning horizon in this level's steps; decays over an eval budget
    receding_horizon steps executed before replanning (level 1 only)
    action_block     primitive actions per level-1 step (the dataset frameskip)
    action_dim       latent action width for levels > 1
    pre_decay_horizon  lower-level horizon before its decay segment starts
    num_subgoals     upper predictions handed down as goals
    cost_last_n      predictions scored against the goal, counted from the end
    intermediate_cost_weight  weight of all but the last scored prediction
    action_cost_weight  pull toward the upper level's macro-action (hierarchical)
    horizon_one_goal_cost_space  'upper' scores through a skipped horizon-1 upper level
    """

    horizon: int
    receding_horizon: int = 1
    action_block: int = 1
    action_dim: int | None = None
    pre_decay_horizon: int | None = None
    num_subgoals: int = 1
    cost_last_n: int = 1
    intermediate_cost_weight: float = 1.0
    action_cost_weight: float = 0.0
    horizon_one_goal_cost_space: str = "lower"

    @classmethod
    def from_dict(cls, cfg, **defaults):
        names = {f.name for f in fields(cls)}
        merged = {**defaults, **{k: v for k, v in dict(cfg).items() if k in names}}
        unknown = set(dict(cfg)) - names
        if unknown:
            raise ValueError(f"unknown plan config keys {sorted(unknown)}; known: {sorted(names)}")
        return cls(**merged)


@dataclass(frozen=True)
class HierarchicalPlanConfig:
    receding_horizon: int
    levels: tuple = field(default_factory=tuple)

    def __getattr__(self, name):
        index = name.removeprefix("level")
        if name.startswith("level") and index.isdigit() and 1 <= int(index) <= len(self.levels):
            return self.levels[int(index) - 1]
        raise AttributeError(name)


SOLVERS = {}


def register_solver(name):
    def wrap(cls):
        SOLVERS[name] = cls
        return cls
    return wrap


def build_solver(spec, model):
    """`{"name": <registered solver>, **kwargs}` -> a solver bound to `model`."""
    spec = dict(spec)
    name = spec.pop("name", "gd")
    if name not in SOLVERS:
        from . import cem, demo_shooting, gd  # noqa: F401  registers the built-in solvers
    if name not in SOLVERS:
        raise ValueError(f"unknown solver '{name}'; registered: {sorted(SOLVERS)}")
    if "optimizer" in spec:
        spec["optimizer_cls"] = getattr(torch.optim, spec.pop("optimizer"))
    return SOLVERS[name](model=model, **spec)
