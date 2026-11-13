"""Receding-horizon control around any solver (stable_worldmodel.policy.WorldModelPolicy
without the environment): plan, keep the first `receding_horizon` level-1 steps, expand
each into its `action_block` primitive actions, and count the steps taken so horizons decay
over the budget."""

import torch


class MPCPlanner:
    def __init__(self, solver, config, *, action_dim, n_envs=1, eval_budget=None):
        self.solver = solver
        self.config = config
        self.action_dim = int(action_dim)
        self.eval_budget = None if eval_budget is None else int(eval_budget)
        self.n_envs = int(n_envs)
        solver.configure(action_dim=action_dim, n_envs=n_envs, config=config)
        self.reset()

    @property
    def level1(self):
        return getattr(self.config, "level1", self.config)

    @property
    def execute_len(self):
        """Primitive actions executed per replan."""
        return int(self.config.receding_horizon) * int(self.level1.action_block)

    def reset(self):
        self.steps_taken = 0
        self.last = None

    def resize(self, n_envs):
        if int(n_envs) != self.n_envs:
            self.n_envs = int(n_envs)
            self.solver.configure(action_dim=self.action_dim, n_envs=self.n_envs, config=self.config)

    def plan(self, info_dict, steps_taken=None):
        """-> (n_envs, execute_len, action_dim) primitive actions, in the model's normalized
        action units. Advances the step counter by what it returns unless `steps_taken`
        is given."""
        if steps_taken is not None:
            self.steps_taken = int(steps_taken)
        out = self.solver(info_dict, steps_taken=self.steps_taken, eval_budget=self.eval_budget)
        self.last = out
        actions = out["actions"]
        keep = min(int(self.config.receding_horizon), actions.shape[1])
        chunk = actions[:, :keep].reshape(actions.shape[0], keep * int(self.level1.action_block), -1)
        self.steps_taken += chunk.shape[1]
        return chunk

    def full_plan(self):
        """The last solve's whole level-1 plan as primitive actions (n_envs, H*block, A)."""
        a = self.last["actions"]
        return a.reshape(a.shape[0], -1, self.action_dim)


def to_device(info, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in info.items()}
