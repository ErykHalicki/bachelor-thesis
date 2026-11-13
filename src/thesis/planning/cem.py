"""Cross-entropy method: sample around a Gaussian, refit it to the `topk` cheapest, repeat.

Needs no gradients, so any `CostModel` works, including ones whose rollout is not
differentiable. The prior comes from the same place as the gradient solver's.
"""

import time

import torch

from .base import register_solver
from .gd import PriorSolverMixin, expand_for_samples


@register_solver("cem")
class CEMSolver(PriorSolverMixin):
    def __init__(self, model, n_steps=30, num_samples=300, topk=30, var_scale=1.0,
                 min_std=0.0, momentum=0.0, batch_size=None, action_clip_sigma=None,
                 device="cpu", seed=1234, return_best=False):
        self.model = model
        self.n_steps = int(n_steps)
        self.num_samples = int(num_samples)
        self.topk = int(topk)
        assert 1 <= self.topk <= self.num_samples, "need 1 <= topk <= num_samples"
        self.var_scale = float(var_scale)
        self.min_std = float(min_std)
        self.momentum = float(momentum)
        self.batch_size = batch_size
        self.action_clip_sigma = action_clip_sigma
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(int(seed))
        self.return_best = bool(return_best)
        self.solve_records = []

    def _initial(self, init_action, horizon):
        stats = self._prior_stats()
        if stats is None:
            mean = torch.zeros(self.action_dim, device=self.device)
            std = torch.ones(self.action_dim, device=self.device)
        else:
            mean, std = stats
        mu = mean.view(1, 1, -1).expand(self.n_envs, horizon, -1).clone()
        if init_action is not None:
            n = min(horizon, init_action.shape[1])
            mu[:, :n] = init_action[:, :n].to(self.device)
        sigma = (std * self.var_scale).view(1, 1, -1).expand_as(mu).clone()
        return mu, sigma

    @torch.no_grad()
    def solve(self, info_dict, init_action=None, planning_horizon=None, steps_taken=None,
              eval_budget=None):
        start_time = time.time()
        horizon = self._resolve_horizon(planning_horizon, steps_taken, eval_budget)
        mu, sigma = self._initial(init_action, horizon)
        clip = self._clip_bounds()
        batch_size = self.batch_size if self.batch_size is not None else self.n_envs
        chosen, costs_out = [], []
        for s in range(0, self.n_envs, batch_size):
            e = min(s + batch_size, self.n_envs)
            bs = e - s
            infos = expand_for_samples(info_dict, s, e, self.num_samples, self.device)
            m, sd = mu[s:e], sigma[s:e]
            best_a = m.clone()
            best_c = torch.full((bs,), float("inf"), device=self.device)
            for _ in range(self.n_steps):
                noise = torch.randn((bs, self.num_samples, *m.shape[1:]),
                                    generator=self.torch_gen, device=self.device)
                cand = m.unsqueeze(1) + sd.unsqueeze(1) * noise
                cand[:, 0] = m
                if clip is not None:
                    cand = cand.clamp(min=clip[0].view(1, 1, 1, -1), max=clip[1].view(1, 1, 1, -1))
                info = infos.copy()
                info["cost_last_n"] = max(1, int(self._config.cost_last_n))
                info["intermediate_cost_weight"] = float(self._config.intermediate_cost_weight)
                costs = self.model.get_cost(info, cand)
                for k in ("embed_0", "goal_embed_0"):
                    if k in info:
                        infos.setdefault(k, info[k])
                order = costs.argsort(dim=1)
                elite = cand.gather(1, order[:, : self.topk, None, None].expand(-1, -1, *cand.shape[2:]))
                lowest = costs.gather(1, order[:, :1])[:, 0]
                better = lowest < best_c
                best_c = torch.where(better, lowest, best_c)
                best_a[better] = cand[torch.arange(bs, device=self.device), order[:, 0]][better]
                new_m, new_sd = elite.mean(dim=1), elite.std(dim=1, unbiased=False).clamp_min(self.min_std)
                m = self.momentum * m + (1 - self.momentum) * new_m
                sd = self.momentum * sd + (1 - self.momentum) * new_sd
            chosen.append((best_a if self.return_best else m).cpu())
            costs_out.append(best_c.cpu())
        actions = torch.cat(chosen, dim=0)
        out = {"actions": actions, "costs": torch.cat(costs_out),
               "predictions": self._rollout_final_actions(info_dict, actions)}
        self.solve_records.append({"level": self.level, "horizon": horizon,
                                   "num_samples": self.num_samples,
                                   "wall_clock": time.time() - start_time})
        return out
