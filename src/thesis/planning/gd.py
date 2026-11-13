"""Gradient descent through the world model (stable_worldmodel.solver.gd.GradientSolver).

`num_samples` action sequences are optimized in parallel: sample 0 starts at the prior mean,
the rest are perturbed by the prior std, and the cheapest after optimization is returned.
The random draws use one seeded generator in the same order as the reference, so equal
seeds give equal plans.
"""

import time

import numpy as np
import torch

from .base import register_solver
from .schedule import get_planning_horizon


def expand_for_samples(info_dict, start, end, num_samples, device):
    """Slice the env batch [start:end] and add a sample axis after it."""
    bs = end - start
    out = {}
    for k, v in info_dict.items():
        if torch.is_tensor(v):
            v = v[start:end].to(device)
            out[k] = v.unsqueeze(1).expand(bs, num_samples, *v.shape[1:])
        elif isinstance(v, np.ndarray):
            out[k] = np.repeat(v[start:end][:, None, ...], num_samples, axis=1)
        else:
            out[k] = v
    return out


class PriorSolverMixin:
    """Prior and clip bounds over a level's actions, from the model's latent action queue
    when it has one (macro-actions), else a standard normal (normalized primitive actions)."""

    def _prior_stats(self):
        queue = getattr(self.model, "get_latent_action_queue", lambda: None)()
        if not (torch.is_tensor(queue) and queue.numel() > 0):
            return None
        queue = queue.detach().to(self.device)
        return queue.mean(dim=0), queue.std(dim=0, unbiased=False).clamp_min(1e-4)

    def _clip_bounds(self):
        if self.action_clip_sigma is not None:
            stats = self._prior_stats()
            if stats is None:
                mean = torch.zeros(self.action_dim, device=self.device)
                std = torch.ones(self.action_dim, device=self.device)
            else:
                mean, std = stats
            return mean - self.action_clip_sigma * std, mean + self.action_clip_sigma * std
        queue = getattr(self.model, "get_latent_action_queue", lambda: None)()
        if not (torch.is_tensor(queue) and queue.numel() > 0):
            return None
        queue = queue.detach().to(self.device)
        return torch.quantile(queue, 0.02, dim=0), torch.quantile(queue, 0.98, dim=0)

    def configure(self, *, action_dim, n_envs, config, level=1):
        self._n_envs = int(n_envs)
        self._config = config
        self.level = int(level)
        self._action_dim = int(action_dim) if level == 1 else int(config.action_dim)

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self._action_dim * self._config.action_block

    @property
    def horizon(self):
        return self._config.horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def _resolve_horizon(self, planning_horizon, steps_taken, eval_budget):
        if planning_horizon is not None:
            return int(planning_horizon)
        return get_planning_horizon(
            configured_horizon=self.horizon,
            replanning_interval=int(self._config.receding_horizon) * int(self._config.action_block),
            steps_taken=steps_taken, eval_budget=eval_budget,
        )

    @torch.inference_mode()
    def _rollout_final_actions(self, info_dict, actions):
        """One rollout of the chosen plan: the subgoals a hierarchical planner hands down."""
        info = expand_for_samples(info_dict, 0, self.n_envs, 1, self.device)
        info = self.model.rollout(info, actions.to(self.device).unsqueeze(1))
        out = {}
        for k, v in info.items():
            if k.startswith("predicted") and torch.is_tensor(v):
                v = v.detach().cpu()
                out[k] = v[:, 0] if v.ndim > 1 and v.shape[1] == 1 else v
        return out


@register_solver("gd")
class GradientSolver(PriorSolverMixin, torch.nn.Module):
    def __init__(self, model, n_steps, min_its=30, batch_size=None, var_scale=1.0,
                 num_samples=1, action_noise=0.0, early_stop_patience=5,
                 early_stop_rel_delta=1e-2, action_clip_sigma=None, device="cpu", seed=1234,
                 optimizer_cls=torch.optim.SGD, optimizer_kwargs=None):
        super().__init__()
        self.model = model
        self.n_steps = int(n_steps)
        self.min_its = max(0, int(min_its))
        self.batch_size = batch_size
        self.num_samples = int(num_samples)
        self.var_scale = float(var_scale)
        self.action_noise = float(action_noise)
        self.early_stop_patience = max(0, int(early_stop_patience))
        self.early_stop_rel_delta = max(0.0, float(early_stop_rel_delta))
        self.action_clip_sigma = action_clip_sigma
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(int(seed))
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(optimizer_kwargs) if optimizer_kwargs is not None else {"lr": 1.0}
        self.solve_records = []

    @staticmethod
    def _significant(score, best, rel_delta):
        if not np.isfinite(best):
            return True
        return best - score > rel_delta * max(abs(best), 1e-12)

    def init_action(self, actions=None, planning_horizon=None):
        horizon = self.horizon if planning_horizon is None else planning_horizon
        stats = self._prior_stats()
        if self.model.level > 1:
            assert stats is not None, "planning a level > 1 needs the model's latent action queue"
        if stats is None:
            mean = torch.zeros(self.action_dim, device=self.device)
            std = torch.ones(self.action_dim, device=self.device)
        else:
            mean, std = stats
        std = std * self.var_scale
        if actions is None:
            actions = mean.view(1, 1, -1).expand(self.n_envs, horizon, -1).clone()
        else:
            actions = actions.to(self.device)
            if actions.shape[1] < horizon:
                fill = mean.view(1, 1, -1).expand(self.n_envs, horizon - actions.shape[1], -1)
                actions = torch.cat([actions, fill.clone()], dim=1)
            else:
                actions = actions[:, :horizon]
        actions = actions.unsqueeze(1).repeat_interleave(self.num_samples, dim=1)
        actions[:, 1:] += torch.randn(actions[:, 1:].shape, generator=self.torch_gen,
                                      device=self.device) * std.view(1, 1, 1, -1)
        if hasattr(self, "init") and self.init.shape == actions.shape:
            self.init.copy_(actions)
        else:
            self.init = torch.nn.Parameter(actions)

    def solve(self, info_dict, init_action=None, planning_horizon=None, steps_taken=None,
              eval_budget=None):
        start_time = time.time()
        cost_last_n = max(1, int(self._config.cost_last_n))
        intermediate = float(self._config.intermediate_cost_weight)
        horizon = self._resolve_horizon(planning_horizon, steps_taken, eval_budget)
        with torch.no_grad():
            self.init_action(init_action, planning_horizon=horizon)
        clip = self._clip_bounds()

        batch_size = self.batch_size if self.batch_size is not None else self.n_envs
        top, iters, costs_out = [], [], []
        for s in range(0, self.n_envs, batch_size):
            e = min(s + batch_size, self.n_envs)
            bs = e - s
            batch_init = self.init[s:e].clone().detach()
            batch_init.requires_grad = True
            optim = self.optimizer_cls([batch_init], **self.optimizer_kwargs)
            infos = expand_for_samples(info_dict, s, e, self.num_samples, self.device)
            best, stalled = float("inf"), 0
            step = -1
            for step in range(self.n_steps):
                info = infos.copy()
                info["cost_last_n"] = cost_last_n
                info["intermediate_cost_weight"] = intermediate
                costs = self.model.get_cost(info, batch_init)
                for k in ("embed_0", "goal_embed_0"):
                    if k in info:
                        infos.setdefault(k, info[k])
                assert costs.shape == (bs, self.num_samples) and costs.requires_grad, (
                    f"cost must be a differentiable ({bs}, {self.num_samples}) tensor, "
                    f"got {tuple(costs.shape)}"
                )
                costs.sum().backward()
                optim.step()
                optim.zero_grad(set_to_none=True)
                if self.action_noise > 0:
                    batch_init.data += torch.randn(batch_init.shape, generator=self.torch_gen,
                                                   device=batch_init.device) * self.action_noise
                if clip is not None:
                    batch_init.data.clamp_(min=clip[0].view(1, 1, 1, -1),
                                           max=clip[1].view(1, 1, 1, -1))
                if self.early_stop_patience > 0:
                    score = float(np.mean(costs.min(dim=1).values.detach().cpu().tolist()))
                    if self._significant(score, best, self.early_stop_rel_delta):
                        best, stalled = score, 0
                    else:
                        stalled += 1
                        if step + 1 >= self.min_its and stalled >= self.early_stop_patience:
                            break
            with torch.no_grad():
                self.init[s:e] = batch_init
                info = infos.copy()
                info["cost_last_n"] = cost_last_n
                info["intermediate_cost_weight"] = intermediate
                final = self.model.get_cost(info, batch_init)
            idx = torch.argsort(final, dim=1)[:, 0]
            top.append(batch_init[torch.arange(bs, device=batch_init.device), idx].detach().cpu())
            costs_out.append(final.gather(1, idx[:, None])[:, 0].detach().cpu())
            iters.extend([step + 1] * bs)

        actions = torch.cat(top, dim=0)
        out = {"actions": actions, "costs": torch.cat(costs_out),
               "predictions": self._rollout_final_actions(info_dict, actions)}
        self.solve_records.append({"level": self.level, "horizon": horizon,
                                   "num_samples": self.num_samples, "n_iters": iters,
                                   "wall_clock": time.time() - start_time})
        return out
