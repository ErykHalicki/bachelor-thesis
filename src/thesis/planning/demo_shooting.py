"""Shooting over recorded action sequences: score `num_samples` segments drawn from a library
of demonstrated actions with the model's cost, keep the cheapest, optionally refine it with a
few narrow CEM iterations around it.

Every candidate is behaviour the world model was trained on, so the search cannot walk into
actions where the model is wrong in its own favour, the failure mode of unconstrained
gradient descent or CEM on a learned model.

`library` is a .npy of (N, T, A) action segments in the model's normalized action units,
with T at least horizon x action_block (see scripts/build_action_library.py).
"""

import time

import numpy as np
import torch

from .base import register_solver
from .gd import PriorSolverMixin, expand_for_samples


@register_solver("demo_shooting")
class DemoShootingSolver(PriorSolverMixin):
    def __init__(self, model, library, num_samples=512, refine_steps=0, refine_samples=128,
                 refine_std=0.1, topk=16, batch_size=None, device="cpu", seed=1234):
        self.model = model
        self.library = torch.as_tensor(np.load(library), dtype=torch.float32)
        self.num_samples = int(num_samples)
        self.refine_steps = int(refine_steps)
        self.refine_samples = int(refine_samples)
        self.refine_std = float(refine_std)
        self.topk = int(topk)
        self.batch_size = batch_size
        self.device = device
        self.action_clip_sigma = None
        self.torch_gen = torch.Generator().manual_seed(int(seed))
        self.solve_records = []

    def _cost(self, infos, cand):
        info = infos.copy()
        info["cost_last_n"] = max(1, int(self._config.cost_last_n))
        info["intermediate_cost_weight"] = float(self._config.intermediate_cost_weight)
        costs = self.model.get_cost(info, cand)
        for k in ("embed_0", "goal_embed_0"):
            if k in info:
                infos.setdefault(k, info[k])
        return costs

    @torch.no_grad()
    def solve(self, info_dict, init_action=None, planning_horizon=None, steps_taken=None,
              eval_budget=None):
        started = time.time()
        horizon = self._resolve_horizon(planning_horizon, steps_taken, eval_budget)
        steps = horizon * int(self._config.action_block)
        if self.library.shape[1] < steps:
            raise ValueError(f"library segments are {self.library.shape[1]} steps, the plan needs {steps}")
        lib = self.library[:, :steps].reshape(len(self.library), horizon, -1)
        assert lib.shape[-1] == self.action_dim, (
            f"library rows are {lib.shape[-1]}-d per level step, the level needs {self.action_dim}"
        )
        batch_size = self.batch_size if self.batch_size is not None else self.n_envs
        chosen, costs_out = [], []
        for s in range(0, self.n_envs, batch_size):
            e = min(s + batch_size, self.n_envs)
            bs = e - s
            pick = torch.randint(0, len(lib), (bs, self.num_samples), generator=self.torch_gen)
            cand = lib[pick].to(self.device)
            infos = expand_for_samples(info_dict, s, e, self.num_samples, self.device)
            costs = self._cost(infos, cand)
            best = costs.argmin(dim=1)
            ar = torch.arange(bs, device=self.device)
            mu, best_c = cand[ar, best], costs[ar, best]
            if self.refine_steps:
                rinfos = expand_for_samples(info_dict, s, e, self.refine_samples, self.device)
                elite = cand.gather(1, costs.argsort(1)[:, : self.topk, None, None]
                                    .expand(-1, -1, *cand.shape[2:]))
                sd = torch.full_like(mu, self.refine_std)
                for _ in range(self.refine_steps):
                    noise = torch.randn((bs, self.refine_samples, *mu.shape[1:]),
                                        generator=self.torch_gen).to(self.device)
                    r = mu[:, None] + sd[:, None] * noise
                    r[:, 0] = mu
                    rc = self._cost(rinfos, r)
                    order = rc.argsort(1)
                    mu = r.gather(1, order[:, : self.topk, None, None].expand(-1, -1, *r.shape[2:])).mean(1)
                    low = rc.gather(1, order[:, :1])[:, 0]
                    better = low < best_c
                    best_c = torch.where(better, low, best_c)
            chosen.append(mu.cpu())
            costs_out.append(best_c.cpu())
        actions = torch.cat(chosen)
        out = {"actions": actions, "costs": torch.cat(costs_out),
               "predictions": self._rollout_final_actions(info_dict, actions)}
        self.solve_records.append({"level": self.level, "horizon": horizon,
                                   "wall_clock": time.time() - started})
        return out
