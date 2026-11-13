"""Goal-reaching on OCBench from demonstration states: the H-JEPA paper's Cube protocol.

Each task is a held-out demonstration window: the simulator is set to the demo's state at
step t, the policy is given the demo's camera frame at t + K as its goal image, and it
succeeds when the cube ends within `tolerance` of where the demo had put it at t + K. The
budget is `budget_steps` (default 2K). Unlike the full task, the goal is always K steps of
demonstrated behaviour away, so this measures whether planning reaches goals the model can
see from where it starts, as the paper's 20-step Cube tasks do.

Knobs (beside the ocbench eval's):
  goal_offset    K, in control steps; a multiple of the dataset's frame interval
  budget_steps   steps a task may take (default 2K)
  tolerance      cube distance counted as reached, metres (default 0.04, the env's own)
  min_cube_move  windows whose demo moves the cube less than this are skipped (default
                 2 x tolerance): they would count as reached before the first action
  goal_camera    the dataset column the goal frame is read from
  episodes       tasks to score; drawn from the dataset's held-out split
  sampling       'random' (any window), or 'pickup': one window per episode centred on
                 `pickup_anchor`, the reference's cube_pickup_centered tasks
  pickup_anchor  'grasp' (the first step of the gripper-close run before the lift) or
                 'lift' (the first step the cube is `pickup_height_delta` above its start)
  gripper_dim    action column commanding the gripper, closed at >= 0.5
  stop_at_goal   end a task once its cube is within tolerance (default true); false runs the
                 full budget, so the final cube and arm errors measure where planning settles
  zero_actions   send zero actions instead of planning: the do-nothing baseline

Besides success it reports the cube's distance (m) and the arm's joint-space distance (rad,
the six arm joints) to the demo's state at the goal: at the start, at the end, and the
closest either came during the task.
"""

import time

import numpy as np

from .ocbench import OCBenchEval, episode_seed, observation_to_frames

CUBE_XYZ = slice(14, 17)
ARM_JOINTS = slice(0, 6)


def set_world_states(env, qpos, qvel):
    """Overwrite every world's state and recompute its kinematics, then return the fresh
    (observation, info) a reset would have."""
    raw = env.unwrapped
    wp = raw._wp
    raw._wp.copy(raw._data.qpos, wp.array(np.ascontiguousarray(qpos, dtype=np.float32), dtype=wp.float32))
    raw._wp.copy(raw._data.qvel, wp.array(np.ascontiguousarray(qvel, dtype=np.float32), dtype=wp.float32))
    raw._mjw.forward(raw._model, raw._data)
    return raw._finish_reset()


_TASK_CACHE = {}


class OCBenchGoalEval(OCBenchEval):
    def _cached_tasks(self, model):
        """The same seed draws the same tasks every eval of a run; decode their goal frames
        once instead of stalling the GPU on it at every eval."""
        from omegaconf import OmegaConf

        cfg = self.cfg
        key = (str(cfg.dataset.repo_id), str(cfg.dataset.get("root")), int(cfg.goal_offset),
               int(cfg.episodes), int(cfg.get("seed", 0)), float(cfg.get("tolerance", 0.04)),
               str(cfg.get("min_cube_move")), str(cfg.get("sampling")), str(cfg.get("pickup_anchor")),
               str(OmegaConf.to_container(cfg.dataset.get("slices") or {})),
               None if model is None else (model.field, *getattr(model, "view_fields", []),
                                           getattr(model, "proprio_field", None)))
        if key not in _TASK_CACHE:
            _TASK_CACHE[key] = self._tasks(model)
        return _TASK_CACHE[key]

    def _make_driver(self, model, obs_features, num_envs):
        """A planning driver with one goal per world, set at every wave."""
        from .chunking import BatchDriver, PlanDriver

        if self.cfg.get("server"):
            raise ValueError("the goal-offset eval runs in-process: unset eval.server")
        size = self.cfg.get("image_size") or (224, 224)
        blank = np.zeros((int(size[0]), int(size[1]), 3), dtype=np.uint8)
        fields = [model.field, *getattr(model, "view_fields", [])]
        state = None
        if getattr(model, "proprio_field", None):
            state = np.zeros(int(model.cfg.level1.encoder.get("proprio_dim")
                                 or model.cfg.level1.wm.proprio_dim), dtype=np.float32)
        driver = PlanDriver(model, self.cfg, obs_features, {f: blank for f in fields},
                            plan=self.cfg.get("plan"), goal_state=state)
        self.sampling = driver.sampling
        return driver, BatchDriver(driver, num_envs)

    def _tasks(self, model):
        """(qpos, qvel, goal frame, goal cube xyz) per task, from held-out episodes."""
        from ...datasets.lerobot import split_episodes
        from lerobot.datasets import LeRobotDataset

        cfg = self.cfg
        K = int(cfg.goal_offset)
        dcfg = cfg.dataset
        ds = LeRobotDataset(dcfg.repo_id, root=dcfg.get("root"))
        meta = ds.meta
        fps = meta.fps
        lengths = np.asarray(meta.episodes["length"])
        held = split_episodes(range(len(lengths)), dcfg.get("validation_split", 0.0), "val",
                              seed=int(dcfg.get("validation_split_seed", 0)))
        stride = int(cfg.get("frame_interval", 25))
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        hf = ds.reader.hf_dataset if hasattr(ds, "reader") else ds.hf_dataset
        starts = np.asarray(meta.episodes["dataset_from_index"])
        from ...datasets.lerobot import _arrow_column

        cube = _arrow_column(hf, "qpos")[:, CUBE_XYZ]
        # a window whose demo barely moves the cube is solved before the first action
        min_move = float(cfg.get("min_cube_move", 2 * float(cfg.get("tolerance", 0.04))))
        sampling = str(cfg.get("sampling", "random"))
        if sampling == "pickup":
            windows = self._pickup_windows(cfg, hf, held, starts, lengths, cube, K, stride)
        elif sampling == "random":
            windows = [(e, t) for e in held for t in range(0, int(lengths[e]) - K, stride)]
        else:
            raise ValueError(f"unknown goal eval sampling '{sampling}'; known: random, pickup")
        candidates = [
            (e, t) for e, t in windows
            if np.linalg.norm(cube[int(starts[e]) + t + K] - cube[int(starts[e]) + t]) >= min_move
        ]
        if not candidates:
            raise ValueError(f"no held-out window of {K} steps moves the cube {min_move} m")
        picks = rng.choice(len(candidates), size=min(int(cfg.episodes), len(candidates)), replace=False)
        from ...datasets.lerobot import resolve_slice

        cams = ([str(cfg.get("goal_camera", "observation.images.front"))] if model is None
                else [model.field, *getattr(model, "view_fields", [])])
        state_field = getattr(model, "proprio_field", None) if model is not None else None
        state_dims = None
        if state_field:
            patterns = dict(dcfg.get("slices") or {}).get(state_field)
            names = meta.features[state_field]["names"]
            state_dims = (resolve_slice(patterns, names, state_field) if patterns
                          else list(range(len(names))))
        tasks = []
        for i in picks:
            e, t = candidates[i]
            row = int(starts[e]) + t
            now, later = hf[row], hf[row + K]
            item = ds[row + K]
            goals = {c: (item[c].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
                     for c in cams}
            task = {"episode": int(e), "t": int(t),
                    "qpos": np.asarray(now["qpos"], dtype=np.float32),
                    "qvel": np.asarray(now["qvel"], dtype=np.float32),
                    "goal": goals,
                    "goal_cube": np.asarray(later["qpos"], dtype=np.float32)[CUBE_XYZ],
                    "goal_arm": np.asarray(later["qpos"], dtype=np.float32)[ARM_JOINTS]}
            if state_field:
                task["goal_state"] = np.asarray(item[state_field], dtype=np.float32)[state_dims]
            tasks.append(task)
        del fps
        print(f"goal eval: {len(tasks)} tasks from {len(candidates)} held-out windows whose "
              f"demo moves the cube >= {min_move} m in {K} steps ({cfg.get('sampling', 'random')} sampling)", flush=True)
        return tasks

    @staticmethod
    def _pickup_windows(cfg, hf, held, starts, lengths, cube, K, stride):
        """One window per held-out episode, centred on its grasp or lift and snapped down to
        the stored-frame grid, so the goal frame is a real frame."""
        from ...datasets.lerobot import _arrow_column

        anchor = str(cfg.get("pickup_anchor", "grasp"))
        delta = float(cfg.get("pickup_height_delta", 0.02))
        gripper = _arrow_column(hf, "action")[:, int(cfg.get("gripper_dim", 6))] >= 0.5
        windows = []
        for e in held:
            s, n = int(starts[e]), int(lengths[e])
            z = cube[s: s + n, 2]
            lifted = np.flatnonzero(z >= z[0] + delta)
            if not lifted.size:
                continue
            centre = int(lifted[0])
            if anchor == "grasp":
                closed = gripper[s: s + centre + 1]
                if not closed.any():
                    continue
                centre = int(np.flatnonzero(closed)[-1])
                while centre > 0 and closed[centre - 1]:
                    centre -= 1
            elif anchor != "lift":
                raise ValueError(f"unknown pickup_anchor '{anchor}'; known: grasp, lift")
            t = (centre - K // 2) // stride * stride
            if t >= 0 and t + K < n:
                windows.append((e, t))
        return windows

    def run(self, model):
        from .base import EvalResult

        cfg = self.cfg
        K = int(cfg.goal_offset)
        budget = int(cfg.get("budget_steps") or 2 * K)
        tol = float(cfg.get("tolerance", 0.04))
        tasks = self._cached_tasks(model)
        num_envs = max(1, min(int(cfg.get("num_envs") or len(tasks)), len(tasks)))
        if model is not None:
            model.eval()
        env = self._make_env(str(cfg.env_name), num_envs, **self._render_kwargs())
        rows, videos, driver = [], {}, None
        try:
            _, info = env.reset(seeds=np.zeros(num_envs, dtype=np.uint32))
            obs_features = self._obs_features(env, info)
            cameras = [c.removeprefix("observation.images.") for c in obs_features
                       if c.startswith("observation.images.")]
            driver, batch = self._make_driver(model, obs_features, num_envs)
            action_dim = int(env.single_action_space.shape[-1])
            for start in range(0, len(tasks), num_envs):
                wave = tasks[start: start + num_envs]
                r, v = self._goal_wave(env, driver, batch, cameras, wave, start, num_envs,
                                       budget, tol, action_dim,
                                       max_videos=int(cfg.get("max_videos", 2)) - len(videos))
                rows += r
                videos.update(v)
        finally:
            if driver is not None:
                driver.close()
            env.close()
        reached = [r["success"] for r in rows]
        metrics = {
            "success_rate": 100.0 * float(np.mean(reached)) if rows else 0.0,
            "episodes": len(rows),
            "goal_offset": K,
            "mean_final_cube_error": float(np.mean([r["final_error"] for r in rows])),
            "mean_start_cube_error": float(np.mean([r["start_error"] for r in rows])),
            "start_within_tolerance": 100.0 * float(np.mean([r["start_error"] <= tol for r in rows])),
            "mean_success_steps": float(np.mean([r["steps"] for r in rows if r["success"]]))
            if any(reached) else 0.0,
        }
        for k in ("min_error", "start_arm_error", "final_arm_error", "min_arm_error"):
            metrics[f"mean_{k}"] = float(np.mean([r[k] for r in rows]))
        metrics["cube_closer"] = 100.0 * float(np.mean([r["final_error"] < r["start_error"] for r in rows]))
        metrics["arm_closer"] = 100.0 * float(np.mean([r["final_arm_error"] < r["start_arm_error"] for r in rows]))
        metrics.update(self._holdout_loss(model))
        return EvalResult(metrics=metrics, videos=videos, episodes=rows)

    def _goal_wave(self, env, driver, batch, cameras, wave, start, num_envs, budget, tol,
                   action_dim, max_videos):
        live = len(wave)
        pad = [wave[min(w, live - 1)] for w in range(num_envs)]
        seeds = np.array([episode_seed(int(self.cfg.get("seed", 0)), start + w) for w in range(num_envs)],
                         dtype=np.uint32)
        env.reset(seeds=seeds)
        observation, info = set_world_states(env, np.stack([t["qpos"] for t in pad]),
                                             np.stack([t["qvel"] for t in pad]))
        driver.set_goals({c: np.stack([t["goal"][c] for t in pad]) for c in pad[0]["goal"]},
                         np.stack([t["goal_state"] for t in pad]) if "goal_state" in pad[0] else None)
        batch.reset(np.zeros(action_dim, dtype=np.float32))
        goal_cube = np.stack([t["goal_cube"] for t in pad])
        goal_arm = np.stack([t["goal_arm"] for t in pad])
        start_error = np.linalg.norm(np.asarray(info["qpos"])[:, CUBE_XYZ] - goal_cube, axis=1)
        start_arm = np.linalg.norm(np.asarray(info["qpos"])[:, ARM_JOINTS] - goal_arm, axis=1)
        stop_at_goal = bool(self.cfg.get("stop_at_goal", True))
        zero = bool(self.cfg.get("zero_actions", False))

        render_every = max(1, int(self.cfg.get("pixel_interval", 1)))
        done = np.arange(num_envs) >= live
        success = np.zeros(num_envs, dtype=bool)
        steps = np.zeros(num_envs, dtype=int)
        error, min_error = start_error.copy(), start_error.copy()
        arm, min_arm = start_arm.copy(), start_arm.copy()
        recording = list(range(min(live, max(0, max_videos))))
        clips = {w: [env.render_world(w).copy()] for w in recording}
        stride = max(1, int(self.cfg.get("video_stride", 4)))
        pixels, opened = None, time.perf_counter()
        for tick in range(1, budget + 1):
            if (tick - 1) % render_every == 0:
                fresh = env.get_pixel_observation(np.arange(num_envs))
                pixels = np.array(fresh, dtype=np.uint8)
            if zero:
                actions = np.zeros((num_envs, action_dim), dtype=np.float32)
            else:
                actions = batch.step(observation_to_frames(observation, cameras, info, pixels))
            actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
            actions[done] = 0.0
            observation, _, _, _, info = env.step(actions)
            qpos = np.asarray(info["qpos"])
            active = ~done
            error = np.where(active, np.linalg.norm(qpos[:, CUBE_XYZ] - goal_cube, axis=1), error)
            arm = np.where(active, np.linalg.norm(qpos[:, ARM_JOINTS] - goal_arm, axis=1), arm)
            min_error, min_arm = np.minimum(min_error, error), np.minimum(min_arm, arm)
            steps[active & ~success] = tick
            reached = active & (error <= tol) & ~success
            success |= reached
            for w in recording:
                if active[w] and (tick % stride == 0 or reached[w]):
                    clips[w].append(env.render_world(w).copy())
            if stop_at_goal:
                done |= reached
            if done.all():
                break
        rows = [{"task": str(self.cfg.env_name), "episode": start + w, "demo_episode": wave[w]["episode"],
                 "demo_t": wave[w]["t"], "success": bool(success[w]), "steps": int(steps[w]),
                 "start_error": float(start_error[w]), "final_error": float(error[w]),
                 "min_error": float(min_error[w]), "start_arm_error": float(start_arm[w]),
                 "final_arm_error": float(arm[w]), "min_arm_error": float(min_arm[w])}
                for w in range(live)]
        videos = {f"goal{start + w}": np.stack(clips[w]).transpose(0, 3, 1, 2) for w in clips}
        print(f"goal eval tasks {start}-{start + live - 1}: {success[:live].mean():.0%} reached, "
              f"cube error {start_error[:live].mean():.3f} -> {error[:live].mean():.3f} m, "
              f"arm error {start_arm[:live].mean():.3f} -> {arm[:live].mean():.3f} rad, "
              f"{time.perf_counter() - opened:.0f}s", flush=True)
        return rows, videos
