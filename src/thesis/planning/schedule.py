"""Planning-horizon schedules over an evaluation budget (stable_worldmodel.solver.solver).

With a budget the horizon decays monotonically to 1 as the episode runs out; with none
(`eval_budget=None`, the open-ended real-robot case) it stays at its configured value.
"""

import math


def num_planning_calls(*, eval_budget, replanning_interval):
    if eval_budget <= 0 or replanning_interval <= 0:
        raise ValueError(f"budget {eval_budget} and interval {replanning_interval} must be positive")
    return math.ceil(int(eval_budget) / int(replanning_interval))


def planning_call_index(*, replanning_interval, steps_taken=None):
    if replanning_interval <= 0:
        raise ValueError(f"replanning interval must be positive, got {replanning_interval}")
    return 0 if steps_taken is None else int(steps_taken) // int(replanning_interval)


def decreasing_horizon_schedule(*, num_calls, configured_horizon):
    num_calls, h = int(num_calls), int(configured_horizon)
    if num_calls <= 0 or h <= 0:
        raise ValueError(f"num_calls {num_calls} and horizon {h} must be positive")
    if h == 1:
        return [1] * num_calls
    if num_calls == 1:
        return [h]
    if num_calls < h:
        span, denom = h - 1, num_calls - 1
        return [max(1, math.ceil(1 + span * (num_calls - 1 - i) / denom)) for i in range(num_calls)]
    values = list(range(h, 0, -1))
    counts = [num_calls // len(values)] * len(values)
    for i in range(len(values) - num_calls % len(values), len(values)):
        counts[i] += 1
    return [v for v, c in zip(values, counts, strict=True) for _ in range(c)]


def get_planning_horizon(*, configured_horizon, replanning_interval, steps_taken=None,
                         eval_budget=None):
    if eval_budget is None:
        return int(configured_horizon)
    schedule = decreasing_horizon_schedule(
        num_calls=num_planning_calls(eval_budget=eval_budget,
                                     replanning_interval=replanning_interval),
        configured_horizon=configured_horizon,
    )
    call = planning_call_index(replanning_interval=replanning_interval, steps_taken=steps_taken)
    return schedule[min(call, len(schedule) - 1)]
