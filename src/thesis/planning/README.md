# planning

Model-predictive planning for world models, independent of any one model or search.
Imported by `algorithms/` (a world model builds its planner here) and never imports from
`datasets/`, `experiments/` or `eval/`.

Three layers, each swappable on its own:

| layer | what it is | ships |
| --- | --- | --- |
| `CostModel` | one level of a world model: `rollout`, differentiable `get_cost`, a latent-action prior | `hierarchical.LevelCostModel` (H-JEPA levels) |
| `Solver` | searches one level's action space against a cost model | `gd` (gradient descent, the H-JEPA paper's), `cem` |
| composition + MPC | how levels feed each other, how often to replan | `HierarchicalSolver`, `MPCPlanner` |

The contracts are spelled out in `base.py`. A new search is a class with `configure` and
`solve`, registered by name:

```python
from thesis.planning import register_solver
from thesis.planning.gd import PriorSolverMixin

@register_solver("mcts")
class MCTSSolver(PriorSolverMixin):
    def __init__(self, model, device="cpu", **knobs): ...
    def solve(self, info, init_action=None, planning_horizon=None, steps_taken=None,
              eval_budget=None):
        h = self._resolve_horizon(planning_horizon, steps_taken, eval_budget)
        actions = ...                                   # (n_envs, h, self.action_dim)
        return {"actions": actions, "predictions": self._rollout_final_actions(info, actions)}
```

and selected per level from config, with nothing else changing:

```yaml
plan:
  solvers:
    level1: {name: cem, n_steps: 30, num_samples: 300, topk: 30}
    level2: {name: mcts, ...}
```

`predictions` must hold `predicted_embed_0`: a hierarchical planner hands the first
prediction(s) of a level's plan down as the next level's goal.

`gd` and `HierarchicalSolver` are numerically identical to the reference
(external/h-jepa) under the same seeds: `tests/test_hjepa_parity.py`.

## Horizons and budgets

`eval_budget` (environment steps) makes horizons decay to 1 as an episode runs out, the
paper's schedule. `null` keeps them fixed, which is what an open-ended robot episode wants.
The driver passes the steps taken in the episode, including over the inference server.
