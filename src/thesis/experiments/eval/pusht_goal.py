"""Goal-reaching on Push-T from dataset states: LeWM's Push-T protocol (le-wm eval.py,
config/eval/pusht.yaml), on the stable-worldmodel env vendored in external/h-jepa.

Each task is a dataset row: the env is set to the row's state, the policy is given the
dataset frame `goal_offset` steps later as its goal image, and it succeeds once the env's
own check (`eval_state`: agent and block within 20 px, block angle within pi/9 of the goal
state) passes at any step of the `budget_steps` budget. Rows are drawn exactly as LeWM draws
them: every row with more than `goal_offset` steps left in its episode is a candidate, and
`np.random.default_rng(seed).choice` picks `episodes` of them. LeWM draws from its training
file, the dataset converted by scripts/convert_pusht_to_lerobot.py, in the same row order,
so `split: all` and seed 42 reproduce its 50 tasks; `split: val` draws from the held-out
episodes instead.

Knobs:
  goal_offset    K, env steps between start and goal (LeWM: 25)
  budget_steps   steps a task may take (LeWM: 50)
  episodes       tasks to score (LeWM: 50)
  seed           the task draw (LeWM: 42)
  split          'all' (LeWM's) or 'val' (the dataset's held-out episodes)
  num_envs       worlds planned in one batched model call
  image_column   the dataset column goal frames are read from
  state_column   the dataset column holding the env's 7-d state
  max_videos     tasks recorded as videos
  plan           merged over the model's own `plan:` section
"""

import sys
import time
from pathlib import Path

import numpy as np

from .base import EvalResult

ENV_ID = "swm/PushT-v1"
REFERENCE = Path(__file__).resolve().parents[4] / "external" / "h-jepa"


def make_pusht(resolution):
    import gymnasium as gym

    if str(REFERENCE) not in sys.path:
        sys.path.insert(0, str(REFERENCE))
    import stable_worldmodel.envs  # noqa: F401  (registers swm/PushT-v1)

    return gym.make(ENV_ID, resolution=int(resolution)).unwrapped


def to_hwc_uint8(frame):
    """A LeRobot (C, H, W) frame -> (H, W, 3) uint8. Inline parquet image columns come back
    as [0, 1] floats whatever `return_uint8` says (it applies to decoded video only)."""
    frame = frame.permute(1, 2, 0)
    if frame.dtype.is_floating_point:
        frame = (frame * 255).round().clamp(0, 255)
    return frame.numpy().astype(np.uint8)


_TASK_CACHE = {}


class PushTGoalEval:
    def __init__(self, cfg):
        self.cfg = cfg

    def _cached_tasks(self):
        cfg = self.cfg
        key = (str(cfg.dataset.repo_id), str(cfg.dataset.get("root")), int(cfg.goal_offset),
               int(cfg.episodes), int(cfg.get("seed", 42)), str(cfg.get("split", "all")))
        if key not in _TASK_CACHE:
            _TASK_CACHE[key] = self._tasks()
        return _TASK_CACHE[key]

    def _tasks(self):
        from lerobot.datasets import LeRobotDataset

        from ...datasets.lerobot import _arrow_column, split_episodes

        cfg, dcfg = self.cfg, self.cfg.dataset
        K = int(cfg.goal_offset)
        image_col = str(cfg.get("image_column", "observation.image"))
        state_col = str(cfg.get("state_column", "state"))
        ds = LeRobotDataset(dcfg.repo_id, root=dcfg.get("root"), return_uint8=True)
        hf = ds.reader.hf_dataset if hasattr(ds, "reader") else ds.hf_dataset
        lengths = np.asarray(ds.meta.episodes["length"])
        episode = _arrow_column(hf, "episode_index")[:, 0]
        step = _arrow_column(hf, "frame_index")[:, 0]
        valid = step <= lengths[episode] - K - 1
        split = str(cfg.get("split", "all"))
        if split == "val":
            held = split_episodes(range(len(lengths)), dcfg.get("validation_split", 0.0), "val",
                                  seed=int(dcfg.get("validation_split_seed", 0)))
            valid &= np.isin(episode, np.asarray(list(held)))
        elif split != "all":
            raise ValueError(f"unknown pusht eval split '{split}'; known: all, val")
        candidates = np.flatnonzero(valid)
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        picks = rng.choice(len(candidates) - 1, size=int(cfg.episodes), replace=False)
        rows = np.sort(candidates[picks])
        state = _arrow_column(hf, state_col)
        tasks = []
        for row in rows:
            row = int(row)
            tasks.append({
                "row": row, "episode": int(episode[row]), "t": int(step[row]),
                "state": state[row].astype(np.float64),
                "goal_state": state[row + K].astype(np.float64),
                "start_image": to_hwc_uint8(ds[row][image_col]),
                "goal": to_hwc_uint8(ds[row + K][image_col]),
            })
        print(f"pusht goal eval: {len(tasks)} tasks from {len(candidates)} rows with "
              f">{K} steps left ({split})", flush=True)
        return tasks

    def _make_driver(self, model, num_envs, size):
        from .chunking import BatchDriver, PlanDriver

        if self.cfg.get("server"):
            raise ValueError("the pusht goal eval runs in-process: unset eval.server")
        blank = np.zeros((size, size, 3), dtype=np.uint8)
        features = {model.field: {"dtype": "image", "shape": (size, size, 3), "names": None}}
        plan = dict(self.cfg.get("plan") or {})
        plan.setdefault("eval_budget", int(self.cfg.budget_steps))
        driver = PlanDriver(model, self.cfg, features, blank, plan=plan)
        return driver, BatchDriver(driver, num_envs)

    def run(self, model):
        cfg = self.cfg
        K, budget = int(cfg.goal_offset), int(cfg.budget_steps)
        tasks = self._cached_tasks()
        size = int(tasks[0]["goal"].shape[0])
        num_envs = max(1, min(int(cfg.get("num_envs") or len(tasks)), len(tasks)))
        model.eval()
        envs = [make_pusht(size) for _ in range(num_envs)]
        driver, batch = self._make_driver(model, num_envs, size)
        rows, videos = [], {}
        try:
            for start in range(0, len(tasks), num_envs):
                wave = tasks[start: start + num_envs]
                r, v = self._wave(envs, driver, batch, wave, start, budget,
                                  max_videos=int(cfg.get("max_videos", 2)) - len(videos))
                rows += r
                videos.update(v)
        finally:
            driver.close()
            for env in envs:
                env.close()
        reached = [r["success"] for r in rows]
        metrics = {
            "success_rate": 100.0 * float(np.mean(reached)),
            "episodes": len(rows),
            "goal_offset": K,
            "budget_steps": budget,
            "mean_success_steps": float(np.mean([r["steps"] for r in rows if r["success"]]))
            if any(reached) else 0.0,
            "mean_start_dist": float(np.mean([r["start_dist"] for r in rows])),
            "mean_final_dist": float(np.mean([r["final_dist"] for r in rows])),
            "mean_min_dist": float(np.mean([r["min_dist"] for r in rows])),
        }
        if cfg.get("holdout", True):
            from .offline import OfflineEval

            metrics.update(OfflineEval(cfg).run(model).metrics)
        return EvalResult(metrics=metrics, videos=videos, episodes=rows)

    def _wave(self, envs, driver, batch, wave, start, budget, max_videos):
        num_envs, live = len(envs), len(wave)
        pad = [wave[min(w, live - 1)] for w in range(num_envs)]
        for w, (env, task) in enumerate(zip(envs, pad, strict=True)):
            env.reset(seed=int(self.cfg.get("seed", 42)) + start + w)
            env._set_state(task["state"])
            env._set_goal_state(task["goal_state"])
        driver.set_goals(np.stack([t["goal"] for t in pad]))
        batch.reset(np.zeros(2, dtype=np.float32))

        def dist(w):
            return envs[w].eval_state(pad[w]["goal_state"], envs[w]._get_obs())

        start_dist = np.array([dist(w)[1] for w in range(num_envs)])
        success = np.zeros(num_envs, dtype=bool)
        steps = np.zeros(num_envs, dtype=int)
        final, closest = start_dist.copy(), start_dist.copy()
        recording = list(range(min(live, max(0, max_videos))))
        frames = [t["start_image"] for t in pad]
        clips = {w: [frames[w]] for w in recording}
        field = driver.model.field
        opened = time.perf_counter()
        for tick in range(1, budget + 1):
            actions = np.asarray(batch.step([{field: f} for f in frames]), dtype=np.float32)
            frames = []
            for w, env in enumerate(envs):
                env.step(actions[w])
                ok, d = dist(w)
                if not success[w]:
                    steps[w] = tick
                success[w] |= bool(ok)
                final[w], closest[w] = d, min(closest[w], d)
                frames.append(env.render())
            for w in recording:
                clips[w].append(frames[w])
        rows = [{"episode": start + w, "demo_episode": wave[w]["episode"], "demo_t": wave[w]["t"],
                 "success": bool(success[w]), "steps": int(steps[w]),
                 "start_dist": float(start_dist[w]), "final_dist": float(final[w]),
                 "min_dist": float(closest[w])}
                for w in range(live)]
        videos = {f"goal{start + w}": np.stack(clips[w]).transpose(0, 3, 1, 2) for w in clips}
        print(f"pusht goal eval tasks {start}-{start + live - 1}: {success[:live].mean():.0%} "
              f"reached, state dist {start_dist[:live].mean():.1f} -> {final[:live].mean():.1f}, "
              f"{time.perf_counter() - opened:.0f}s", flush=True)
        return rows, videos
