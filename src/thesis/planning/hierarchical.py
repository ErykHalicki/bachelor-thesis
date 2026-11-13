"""Top-down planning over a level stack (external/h-jepa h_jepa/hierarchical_solver.py).

Each level gets its own solver of any registered kind, bound to a `LevelCostModel` that rolls
out in that level's latent space and scores against goals given in the level above. The top
level plans to the real goal; each lower level plans to the first prediction(s) of the plan
above it, measured after encoding its own predictions up a level.

Levels are duck-typed JEPA modules: `level`, `temporal_stride`, `temporal_window_size`,
`encode(info, key, temporal_stride, chunk_temporal_inputs)`, `rollout`, `criterion`,
`action_embed`, `action_encoder`, `_chunk_temporal_info`, `get_latent_action_queue`.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .base import build_solver
from .schedule import decreasing_horizon_schedule, num_planning_calls, planning_call_index


def build_hierarchical_solver(level_models, solver_specs, encode_levels):
    """`solver_specs`: {"level1": {"name": "gd", ...}, "level2": ...}. `encode_levels(info,
    key)` -> {"embed_<level>": (B, 1, D)} encodes an observation (`key="pixels"`) or the goal
    (`key="goal"`) through every level."""
    solvers = {}
    for level, model in enumerate(level_models, start=1):
        spec = solver_specs[f"level{level}"]
        solvers[level] = build_solver(spec, LevelCostModel(model, level_models[level:]))
    return HierarchicalSolver(solvers, level_models, encode_levels)


class HierarchicalSolver:
    def __init__(self, level_solvers, level_models, encode_levels):
        self.level_solvers = dict(level_solvers)
        self.level_models = list(level_models)
        self.encode_levels = encode_levels
        self._config = None

    def configure(self, *, action_dim, n_envs, config, level=1):
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(action_dim)
        for lv, solver in sorted(self.level_solvers.items()):
            solver.configure(action_dim=action_dim, n_envs=n_envs,
                             config=self._level_cfg(lv), level=lv)

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self._action_dim * self._level_cfg(1).action_block

    @property
    def horizon(self):
        return self._level_cfg(1).horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def _level_cfg(self, level):
        return getattr(self._config, f"level{level}")

    def _replanning_interval(self):
        return int(self._config.receding_horizon) * int(self._level_cfg(1).action_block)

    def _pre_decay_horizon(self, level, fallback):
        configured = getattr(self._level_cfg(level), "pre_decay_horizon", None)
        if configured is None:
            configured = getattr(self.level_models[level], "temporal_stride", fallback)
        if int(configured) <= 0:
            raise ValueError(f"pre_decay_horizon must be positive for level {level}")
        return int(configured)

    def horizon_schedules(self, eval_budget):
        """{level: [horizon per planning call]}; None budget -> one call at full horizons."""
        top = len(self.level_models)
        if eval_budget is None:
            out = {top: [int(self._level_cfg(top).horizon)]}
            for level in range(top - 1, 0, -1):
                h = int(self._level_cfg(level).horizon)
                out[level] = [h if out[level + 1][0] == 1 else self._pre_decay_horizon(level, h)]
            return out
        calls = num_planning_calls(eval_budget=int(eval_budget),
                                   replanning_interval=self._replanning_interval())
        out = {}
        for level in range(top, 0, -1):
            h = self._level_cfg(level).horizon
            if level == top:
                out[level] = decreasing_horizon_schedule(num_calls=calls, configured_horizon=h)
                continue
            pre = self._pre_decay_horizon(level, h)
            upper = out[level + 1]
            if 1 not in upper:
                out[level] = [pre] * calls
                continue
            tail_start = upper.index(1)
            out[level] = [pre] * tail_start + decreasing_horizon_schedule(
                num_calls=calls - tail_start, configured_horizon=h)
        return out

    def solve(self, info_dict, steps_taken=None, eval_budget=None, init_action=None,
              planning_horizon=None):
        subgoal_latents = subgoal_actions = skipped_goal_level = None
        upper_anchored = True
        schedules = self.horizon_schedules(eval_budget)
        call = planning_call_index(replanning_interval=self._replanning_interval(),
                                   steps_taken=steps_taken)
        obs_emb = self.encode_levels(info_dict, "pixels")
        goal_emb = self.encode_levels(info_dict, "goal")
        plans = {}

        for level in range(len(self.level_models), 0, -1):
            cfg = self._level_cfg(level)
            sched = schedules[level]
            horizon = sched[min(call, len(sched) - 1)]
            if level > 1 and horizon == 1:
                skipped_goal_level = ((skipped_goal_level or level)
                                      if cfg.horizon_one_goal_cost_space == "upper" else None)
                subgoal_latents = subgoal_actions = None
                continue

            level_info = dict(info_dict)
            level_info["embed_0"] = obs_emb[f"embed_{level}"]
            level_info["action_cost_weight"] = cfg.action_cost_weight
            if subgoal_latents is None:
                goal_level = skipped_goal_level or level
                level_info["goal_embed_0"] = goal_emb[f"embed_{goal_level}"]
                level_info["goal_embed_0_level"] = goal_level
                level_info["goal_embed_0_in_upper_space"] = goal_level > level
                level_info["dense_upper_goal_projection"] = goal_level > level
                anchored = True
            else:
                n = self._level_cfg(level + 1).num_subgoals
                first = 1 if subgoal_latents.shape[1] > 1 else 0
                first_act = max(0, first - 1)
                includes_last = first + n >= subgoal_latents.shape[1]
                next_goal = subgoal_latents[:, first:first + n].clone()
                next_action = subgoal_actions[:, first_act:first_act + n].clone()
                if next_action.shape[1] != next_goal.shape[1]:
                    raise ValueError(f"level {level}: {next_action.shape[1]} action subgoals "
                                     f"for {next_goal.shape[1]} state subgoals")
                anchored = False
                if includes_last and upper_anchored:
                    next_goal[:, -1] = goal_emb[f"embed_{level + 1}"].squeeze(1).to(next_goal.device)
                    anchored = True
                level_info["goal_embed_0"] = next_goal
                level_info["goal_action"] = next_action
                level_info["goal_embed_0_in_upper_space"] = True

            goal = level_info.get("goal_embed_0")
            if level_info.get("goal_embed_0_in_upper_space") and torch.is_tensor(goal) and goal.shape[-2] > 1:
                stride = self.level_solvers[level].model.upper_models[0].temporal_stride
                expected = (horizon + stride) // stride - 1
                assert goal.shape[-2] == expected, (
                    f"level {level}: {goal.shape[-2]} subgoals, but stride {stride} at horizon "
                    f"{horizon} leaves {expected}"
                )

            result = self.level_solvers[level].solve(
                level_info, planning_horizon=horizon, steps_taken=steps_taken,
                eval_budget=eval_budget)
            plans[level] = result
            upper_anchored = anchored
            skipped_goal_level = None
            subgoal_latents = result["predictions"]["predicted_embed_0"]
            subgoal_actions = result["actions"]

        return {"actions": plans[1]["actions"], "levels": plans}


class LevelCostModel(nn.Module):
    """Rolls out one level and scores in the goal's space: this level's, or the level above
    after encoding the predicted stream with the upper encoders."""

    def __init__(self, model, upper_models):
        super().__init__()
        self.model = model
        self.upper_models = list(upper_models)
        self.level = model.level

    def rollout(self, info_dict, action_candidates):
        return self.model.rollout(info_dict, action_candidates)

    def get_latent_action_queue(self):
        return self.model.get_latent_action_queue()

    def get_cost(self, info_dict, action_candidates):
        info_dict = self.rollout(info_dict, action_candidates)
        weight = float(info_dict.get("action_cost_weight", 0.0))
        pred, act = self._project_to_goal_space(
            info_dict["predicted_embed_0"],
            action_candidates if weight != 0.0 else None,
            use_upper_space=bool(info_dict.get("goal_embed_0_in_upper_space", True)),
            dense=bool(info_dict.get("dense_upper_goal_projection", False)),
            goal_level=int(info_dict.get("goal_embed_0_level", self.level + 1)),
        )
        cost_info = dict(info_dict)
        cost_info["predicted_embed_0"] = pred
        cost = self.model.criterion(cost_info)
        if weight != 0.0:
            extra = self._action_cost(act, cost_info)
            if extra is not None:
                cost = cost + weight * extra
        return cost

    def _project_to_goal_space(self, pred, actions, *, use_upper_space, dense, goal_level):
        if not self.upper_models or not use_upper_space:
            return pred, None
        window = self.upper_models[0].temporal_window_size
        pad = max(0, window - pred.shape[-2]) if dense else window - 1
        states = self._prefix_repeat_first(pred, pad)
        up = self._encode_states_up(states, dense=dense, goal_level=goal_level)
        act = None
        if actions is not None:
            act = self._encode_actions_up(states, actions, pad=pad, dense=dense)
        return up, act

    def _encode_states_up(self, states, *, dense, goal_level):
        stride = 1 if dense else None
        b, s = states.shape[:2]
        key = f"embed_{self.level}"
        out = self.upper_models[0].encode({key: states.flatten(0, 1)}, key=key,
                                          chunk_temporal_inputs=True, temporal_stride=stride)
        for upper in range(self.level + 2, goal_level + 1):
            key = f"embed_{upper - 1}"
            out = self.upper_models[upper - self.level - 1].encode(
                {key: out["embed_0"]}, key=key, chunk_temporal_inputs=True,
                temporal_stride=stride)
        return out["embed_0"].unflatten(0, (b, s))

    def _encode_actions_up(self, states, actions, *, pad, dense):
        upper = self.upper_models[0]
        b, s = states.shape[:2]
        emb = self._prefix_zeros(self.model.action_embed(actions), pad)
        emb = self._pad_zeros_to_length(emb, states.shape[-2])
        key = f"embed_{self.level}"
        chunked = upper._chunk_temporal_info(
            {key: states.flatten(0, 1), "action": emb.flatten(0, 1)}, key=key,
            stride=1 if dense else upper.temporal_stride, window_size=upper.temporal_window_size)
        pooled = upper.action_encoder(chunked["action"], chunked["action_mask"])
        return pooled.unflatten(0, (b, s))[..., :-1, :]

    @staticmethod
    def _action_cost(pred, info):
        target = info.get("goal_action")
        if pred is None or target is None:
            return None
        n = min(max(1, int(info.get("cost_last_n", 1))), pred.shape[-2], target.shape[-2])
        pred = pred[..., -n:, :]
        target = target[..., -n:, :].to(pred.device).expand_as(pred)
        cost = F.mse_loss(pred, target.detach(), reduction="none")
        return cost.sum(dim=tuple(range(2, cost.ndim)))

    @staticmethod
    def _prefix_repeat_first(x, count):
        if count <= 0:
            return x
        return torch.cat([x[..., :1, :].expand(*x.shape[:-2], count, x.shape[-1]), x], dim=-2)

    @staticmethod
    def _prefix_zeros(x, count):
        if count <= 0:
            return x
        return torch.cat([x.new_zeros(*x.shape[:-2], count, x.shape[-1]), x], dim=-2)

    @staticmethod
    def _pad_zeros_to_length(x, target_len):
        pad = target_len - x.shape[-2]
        if pad == 0:
            return x
        return torch.cat([x, x.new_zeros(*x.shape[:-2], pad, x.shape[-1])], dim=-2)
