"""Held-out losses plus open-loop planning on held-out windows, for world models.

Each planned window starts at its first frame and aims at its last. The planner (the
model's own `plan:` merged with this eval's) solves once from the start frame, and the
executed chunk is compared with what the demonstrator did over the same rows:

    plan/action_mse        executed chunk vs recorded actions, normalized units
    plan/action_mse_zero   the same against an all-zero chunk (the normalized mean), the
                           score of a planner that learned nothing
    plan/cost              the planner's cost for its own plan at level 1
    plan/goal_dist_demo    latent distance to the goal after rolling out the recorded
                           actions through the model; `_zero` the same for all-zero actions.
                           demo well below zero means the model's predictions depend on the
                           actions, which planning needs before any solver can do well
    plan/solve_s           wall-clock per solve

Same open-loop protocol as the reference's DROID eval: no environment, so it measures
whether planning on this data recovers the demonstrated behavior, not task success.
"""

import time

from .offline import OfflineEval


class OfflinePlanEval(OfflineEval):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.plan_samples = int(cfg.get("plan_samples", 16))
        self.plan_overrides = cfg.get("plan") or {}

    def run(self, model):
        import torch
        from omegaconf import OmegaConf

        result = super().run(model)
        if self.plan_samples <= 0:
            return result
        if not hasattr(model, "make_planner"):
            raise ValueError(f"{type(model).__name__} cannot plan: it has no make_planner()")

        dataset = self._build(model)
        overrides = (OmegaConf.to_container(self.plan_overrides, resolve=True)
                     if not isinstance(self.plan_overrides, dict) else self.plan_overrides)
        device = next(model.parameters()).device
        gen = torch.Generator().manual_seed(self.seed)
        picks = torch.randperm(len(dataset), generator=gen)[: self.plan_samples].tolist()
        was_training = model.training
        model.eval()
        planner = model.make_planner(overrides)
        sums = {"plan/action_mse": 0.0, "plan/action_mse_zero": 0.0, "plan/cost": 0.0,
                "plan/solve_s": 0.0, "plan/goal_dist_demo": 0.0, "plan/goal_dist_zero": 0.0}
        try:
            for i in picks:
                sample = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
                          for k, v in dataset[i].items()}
                frames = sample[model.field]
                obs = {model.field: frames[:, :1]}
                goal = {model.field: frames[:, -1:]}
                if model.proprio_field:
                    obs[model.proprio_field] = sample[model.proprio_field][:, :1]
                    goal[model.proprio_field] = sample[model.proprio_field][:, -1:]
                planner.reset()
                started = time.perf_counter()
                chunk = model.plan(planner, obs, goal).to(device)
                sums["plan/solve_s"] += time.perf_counter() - started
                truth = sample[model.action_field][:, : chunk.shape[1]].float()
                sums["plan/action_mse"] += torch.mean((chunk - truth) ** 2).item()
                sums["plan/action_mse_zero"] += torch.mean(truth ** 2).item()
                if hasattr(model, "goal_distance"):
                    steps = (frames.shape[1] - 1) * model.frameskip
                    demo = sample[model.action_field][:, :steps].float()
                    sums["plan/goal_dist_demo"] += model.goal_distance(obs, goal, demo).mean().item()
                    sums["plan/goal_dist_zero"] += model.goal_distance(
                        obs, goal, torch.zeros_like(demo)).mean().item()
                cost = planner.last["levels"][1].get("costs")
                if cost is not None:
                    sums["plan/cost"] += float(cost.mean())
        finally:
            model.train(was_training)
        n = len(picks)
        result.metrics.update({k: v / n for k, v in sums.items()})
        result.metrics["plan_samples"] = n
        return result
