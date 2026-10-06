"""RoboCasa rollout eval: simulated rollouts on the RoboCasa365 kitchen benchmark.

Success is the simulator's own check; the rollout protocol is the one every backend runs,
a chunk of `execute_len` actions per replan driven by the ChunkDriver.

Every `reset(seed)` draws a kitchen — layout, style, objects — from the task's own
distribution, so each episode opens on a scene no demonstration showed. The
per-(task, episode) seed recipe keeps those draws reproducible across evals.

Tasks are named by RoboCasa task name (`OpenDrawer`, `PrepareCoffee`, ...), the same
strings the dataset's `tasks:` list carries, and the eval config interpolates that list
so the evaluated task cannot drift from the trained one.
"""

import numpy as np

from .base import EvalResult
from .chunking import BatchDriver, ChunkDriver, RemoteDriver

DEFAULT_CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
IMAGE_PREFIX = "observation.images"
STATE_COLUMN = "observation.state"
STATE_DIM = 16   # base_pos(3) + base_quat(4) + ee_pos_rel(3) + ee_quat_rel(4) + gripper(2)

# action is base_motion(4) + control_mode(1) + ee_pos(3) + ee_rot(3) + gripper(1). Every
# actuated dimension is relative and rests at zero; these two carry a latched mode, which
# the dataset records at -1 (arm mode, gripper open) at an episode start, never at zero.
LATCHED_ACTION_DIMS = {4: -1.0, 11: -1.0}

# must exceed any episode count, so no two (task, episode) pairs share a seed
TASK_SEED_STRIDE = 10_000


def image_column(name):
    return f"{IMAGE_PREFIX}.{name}"


def observation_to_frame(observation):
    """A RoboCasaEnv observation -> the dataset-keyed frame the driver consumes.

    Nothing needs fixing up: the dataset was recorded through this same wrapper, so the
    images arrive in the recorded orientation and `agent_pos` IS `observation.state`.
    """
    frame = {
        image_column(name): np.ascontiguousarray(np.asarray(image))
        for name, image in observation["pixels"].items()
    }
    frame[STATE_COLUMN] = np.asarray(observation["agent_pos"], dtype=np.float32)
    return frame


def observation_to_frames(observation, num_envs):
    """The same, for a vector env: leading-axis-stacked observation -> one frame per env."""
    return [
        {
            **{image_column(name): np.ascontiguousarray(np.asarray(images[env]))
               for name, images in observation["pixels"].items()},
            STATE_COLUMN: np.asarray(observation["agent_pos"][env], dtype=np.float32),
        }
        for env in range(num_envs)
    ]


def _build_env(kwargs):
    """Construct one RoboCasaEnv. Module-level and taking a plain dict so
    `partial(_build_env, kwargs)` pickles into an AsyncVectorEnv forkserver worker.
    """
    from lerobot.envs.robocasa import RoboCasaEnv

    return RoboCasaEnv(**kwargs)


def _success_flags(info, num_envs):
    """Per-env `is_success` out of a vector env's aggregated info dict.

    Must raise rather than default to False: a missing or mis-shaped key would silently
    score every episode a failure, which reads exactly like a model that never succeeds.
    """
    flags = info.get("is_success")
    if flags is None:
        raise ValueError(
            f"vector env info has no 'is_success' (keys: {sorted(info)}). RoboCasaEnv.step "
            f"sets it on every step, so this means the wrapper stack changed shape."
        )
    flags = np.asarray(flags).reshape(-1)
    if flags.size != num_envs:
        raise ValueError(
            f"'is_success' has {flags.size} entries for {num_envs} envs; cannot tell which "
            f"episode succeeded."
        )
    return flags.astype(bool)


class RoboCasaEval:
    """Runs a policy over RoboCasa tasks and reports per-task and overall success."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.checkpoint_step = None
        self.sampling = {}

    def _scene_seed(self, task_ordinal, episode):
        base = int(self.cfg.get("scene_seed", 0))
        return base + TASK_SEED_STRIDE * int(task_ordinal) + int(episode)

    def _obs_features(self):
        cameras = self.cfg.get("cameras") or [image_column(c) for c in DEFAULT_CAMERAS]
        size = int(self.cfg.get("render_size", 256))
        features = {
            column: {"dtype": "video", "shape": (size, size, 3), "names": None}
            for column in cameras
        }
        features[STATE_COLUMN] = {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": [f"state.{i}" for i in range(STATE_DIM)],
        }
        return features

    @staticmethod
    def _hold_action(driver):
        """The raw action the model should believe preceded an episode's first step.

        This never reaches the environment -- it only warms the executed-action history the
        policy conditions on -- so it must match what the DATASET holds at an episode start,
        not what the controller treats as a no-op. Those differ: the controller thresholds
        the latched dimensions and reads zero and -1 alike, but a zero there is a value the
        model has never seen.
        """
        dim = getattr(driver, "action_context_dim", 0)
        if not dim:
            return None
        hold = np.zeros(dim, dtype=np.float32)
        for index, value in LATCHED_ACTION_DIMS.items():
            if index < dim:
                hold[index] = value
        return hold

    def _make_driver(self, model):
        obs_features = self._obs_features()
        if not self.cfg.get("server"):
            driver = ChunkDriver(model, self.cfg, obs_features)
            self.sampling = driver.sampling
            return driver
        driver = RemoteDriver(self.cfg, obs_features, self.cfg.get("run"))
        self.checkpoint_step = driver.checkpoint_step
        self.sampling = driver.sampling
        return driver

    def _env_kwargs(self, task):
        cfg = self.cfg
        cameras_raw = cfg.get("cameras_raw") or DEFAULT_CAMERAS
        size = int(cfg.get("render_size", 256))
        split = cfg.get("split")
        length = cfg.get("max_episode_steps")
        return {
            "task": str(task),
            "camera_name": ",".join(cameras_raw),
            "obs_type": "pixels_agent_pos",
            "observation_height": size,
            "observation_width": size,
            "split": None if split is None else str(split),
            "episode_length": None if length is None else int(length),
            # RoboCasaEnv adds `episode_index` to the seed it is reset with; pinning it to 0
            # in EVERY worker is what makes the vectorized path draw the serial path's
            # kitchens from its explicit per-episode seeds
            "episode_index": 0,
        }

    def _make_env(self, task):
        return _build_env(self._env_kwargs(task))

    def run(self, model):
        cfg = self.cfg
        tasks = [str(t) for t in (cfg.get("tasks") or [])]
        if not tasks:
            raise ValueError(
                "eval.tasks is empty. Name the RoboCasa task(s) to evaluate -- normally "
                "by interpolating the dataset's own list, so the evaluated tasks cannot "
                "drift from the trained ones."
            )

        if model is not None:
            model.eval()
        driver = self._make_driver(model)
        episodes = int(cfg.get("episodes", 20))
        max_videos = int(cfg.get("max_videos", 2))

        # parallel envs do not change which episodes are scored
        num_envs = max(1, int(cfg.get("num_envs", 1) or 1))

        rows, videos = [], {}
        try:
            for ordinal, task in enumerate(tasks):
                if num_envs > 1:
                    task_rows, task_videos = self._rollout_task_vector(
                        driver, task, ordinal, episodes, max_videos, num_envs
                    )
                else:
                    env = self._make_env(task)
                    try:
                        task_rows, task_videos = self._rollout_task(
                            driver, env, task, ordinal, episodes, max_videos
                        )
                    finally:
                        env.close()
                rows.extend(task_rows)
                videos.update(task_videos)
        finally:
            driver.close()

        metrics = self._metrics(rows)
        metrics.update(self._holdout_loss(model))
        return EvalResult(metrics=metrics, videos=videos, episodes=rows)

    def _holdout_loss(self, model):
        """Held-out loss alongside the success rate, under the usual `loss` names."""
        if model is None or not self.cfg.get("holdout", True):
            return {}
        from .offline import OfflineEval

        return OfflineEval(self.cfg).run(model).metrics

    def _rollout_task(self, driver, env, task, ordinal, episodes, max_videos):
        cfg = self.cfg
        max_steps = env._max_episode_steps
        stride = max(1, int(cfg.get("video_stride", 4)))
        rows, videos = [], {}

        import time
        for episode in range(episodes):
            # RoboCasaEnv.step() resets itself when an episode terminates; this seeded reset
            # must follow it regardless, or that self-reset picks the kitchen
            opened = time.perf_counter()
            observation, _ = env.reset(seed=self._scene_seed(ordinal, episode))
            driver.reset(self._hold_action(driver))
            record = len(videos) < max_videos
            frames = []
            success, steps = False, 0
            for steps in range(1, max_steps + 1):
                frame = observation_to_frame(observation)
                if record and (steps - 1) % stride == 0:
                    frames.append(frame[image_column(next(iter(observation["pixels"])))])
                action = np.asarray(driver.step(frame), dtype=np.float32)
                observation, _, terminated, _, info = env.step(action)
                if terminated:
                    success = bool(info.get("is_success", False))
                    break

            rows.append({
                "task": task,
                "task_id": ordinal,
                "episode": episode,
                "success": success,
                "steps": steps,
            })
            print(f"eval {task} ep {episode}: success={success} steps={steps} "
                  f"({time.perf_counter() - opened:.0f}s)", flush=True)
            if record and frames:
                videos[f"{ordinal}_ep{episode}"] = np.stack(frames).transpose(0, 3, 1, 2)
        return rows, videos

    def _rollout_task_vector(self, driver, task, ordinal, episodes, max_videos, num_envs):
        """`_rollout_task`'s protocol, `num_envs` kitchens at a time.

        Episodes run in waves of `num_envs`, each wave reset with an explicit list of the
        SAME per-(task, episode) seeds the serial path draws, so both paths score the
        identical set of kitchens.
        """
        import time
        from functools import partial

        from lerobot.envs.utils import _LazyAsyncVectorEnv

        cfg = self.cfg
        kwargs = self._env_kwargs(task)
        # RoboCasaEnv builds its spaces in __init__ but the kitchen only on first reset,
        # so this reads the horizon without paying for a scene compile
        probe = _build_env(kwargs)
        max_steps = probe._max_episode_steps
        probe.close()
        stride = max(1, int(cfg.get("video_stride", 4)))

        vec = _LazyAsyncVectorEnv([partial(_build_env, kwargs) for _ in range(num_envs)])
        batch = BatchDriver(driver, num_envs)
        rows, videos = [], {}
        try:
            for start in range(0, episodes, num_envs):
                live = min(num_envs, episodes - start)
                opened = time.perf_counter()
                observation, _ = vec.reset(
                    seed=[self._scene_seed(ordinal, start + env) for env in range(num_envs)]
                )
                batch.reset(self._hold_action(driver))
                success = np.zeros(num_envs, dtype=bool)
                steps = np.zeros(num_envs, dtype=int)
                done = np.zeros(num_envs, dtype=bool)
                recording = [env for env in range(live) if len(videos) + env < max_videos]
                clips = {env: [] for env in recording}
                camera = image_column(next(iter(observation["pixels"])))

                for tick in range(1, max_steps + 1):
                    frames = observation_to_frames(observation, num_envs)
                    for env in recording:
                        if not done[env] and (tick - 1) % stride == 0:
                            clips[env].append(frames[env][camera])
                    actions = batch.step(frames)
                    observation, _, terminated, truncated, info = vec.step(actions)
                    steps[~done] = tick
                    ended = (np.asarray(terminated) | np.asarray(truncated)) & ~done
                    if ended.any():
                        flags = _success_flags(info, num_envs)
                        success[ended] = flags[ended]
                        done |= ended
                    if done[:live].all():
                        break

                for env in range(live):
                    episode = start + env
                    rows.append({
                        "task": task,
                        "task_id": ordinal,
                        "episode": episode,
                        "success": bool(success[env]),
                        "steps": int(steps[env]),
                    })
                    print(f"eval {task} ep {episode}: success={bool(success[env])} "
                          f"steps={int(steps[env])}", flush=True)
                    if env in clips and clips[env]:
                        videos[f"{ordinal}_ep{episode}"] = (
                            np.stack(clips[env]).transpose(0, 3, 1, 2)
                        )
                print(f"eval {task} wave {start}-{start + live - 1} of {episodes}: "
                      f"{time.perf_counter() - opened:.0f}s on {num_envs} envs", flush=True)
        finally:
            vec.close()
        return rows, videos

    def _metrics(self, rows):
        if not rows:
            return {"success_rate": 0.0, "episodes": 0}
        metrics = {
            "success_rate": 100.0 * np.mean([r["success"] for r in rows]),
            "episodes": len(rows),
            "mean_episode_steps": float(np.mean([r["steps"] for r in rows])),
        }
        for task_id in sorted({r["task_id"] for r in rows}):
            picked = [r["success"] for r in rows if r["task_id"] == task_id]
            metrics[f"success_rate/task_{task_id}"] = 100.0 * float(np.mean(picked))
        return metrics
