"""OCBench rollout eval: batched MJWarp rollouts of a policy on an OCBench task.

`num_envs` worlds simulate in one GPU batch and one batched model call replans every world
whose chunk ran out (BatchDriver), so a few hundred episodes cost about as much wall-clock
as one. Success is the env's own check, and an episode ends when it succeeds, when the
simulation goes unhealthy, or at `max_episode_steps` -- the env never ends an episode on
time by itself, so the cap lives here, defaulting to the env's registered limit.

Episodes are seeded per episode index, not per world, so a run scores the same initial
states whatever `num_envs` is.

A policy trained on a sparse visual dataset reads camera frames that the dataset stored
once every K steps. The eval matches that by rolling out a STATE env and rendering its
cameras only every `pixel_interval` steps (`pixel_cameras` / `pixel_size` set the render),
holding each frame in between -- with `execute_len` = K the policy replans exactly on the
rendered steps, so it sees frames spaced as in training. Rendering every step would cost
a CPU render per world per step.
"""

import time

import numpy as np

from .base import EvalResult
from .chunking import BatchDriver, RemoteDriver, make_driver

STATE_COLUMN = "observation.state"
IMAGE_PREFIX = "observation.images"
# simulator state columns, read off the env's info dict like the collector records them
INFO_COLUMNS = ("qpos", "qvel")
DEFAULT_CAMERAS = ("front", "side", "wrist")


def episode_seed(seed, episode):
    """The uint32 reset seed of one episode, independent of which wave or world runs it."""
    return np.random.SeedSequence([int(seed), int(episode)]).generate_state(1, dtype=np.uint32)[0]


def observation_to_frames(observation, cameras, info=None, pixels=None):
    """A batched env observation -> one dataset-keyed frame per world.

    A state env yields `(N, obs_dim)` floats; a visual env `(N, cameras, H, W, 3)` uint8,
    each camera served as its own column the way the dataset backend serves them. `info`
    adds the `qpos`/`qvel` columns, and `pixels` -- frames rendered from a state env --
    the camera columns.
    """
    observation = np.asarray(observation)
    if observation.dtype != np.uint8:
        frames = [{STATE_COLUMN: row.astype(np.float32)} for row in observation]
    else:
        if observation.ndim == 4:
            observation = observation[:, None]
        frames = [
            {f"{IMAGE_PREFIX}.{name}": np.ascontiguousarray(world[i])
             for i, name in enumerate(cameras)}
            for world in observation
        ]
    for col in INFO_COLUMNS:
        if info is not None and col in info:
            for frame, row in zip(frames, np.asarray(info[col])):
                frame[col] = row.astype(np.float32)
    if pixels is not None:
        for frame, world in zip(frames, pixels):
            frame.update({f"{IMAGE_PREFIX}.{name}": world[i] for i, name in enumerate(cameras)})
    return frames


def _make_env(env_name, num_envs, **kwargs):
    import ocbench

    return ocbench.make(env_name, nworld=num_envs, **kwargs)


class _SerialBatch:
    """BatchDriver's interface over one non-batching driver (the remote one), for a
    single-world rollout."""

    def __init__(self, driver):
        self.driver = driver

    def reset(self, hold_action=None):
        self.driver.reset(hold_action)

    def step(self, frames):
        return np.asarray([self.driver.step(frames[0])], dtype=np.float32)


class OCBenchEval:
    """Rolls a policy out on one OCBench env and reports its success rate.

    Config knobs:
      env_name           the OCBench env, normally interpolated from the dataset's.
      episodes           episodes to score.
      num_envs           worlds simulated per batch; episodes run in waves of this size.
      max_episode_steps  per-episode cap; null uses the env's registered limit.
      seed               the base of every episode's reset seed.
      cameras            camera names for a visual env's camera axis, in env order.
      max_videos         episodes recorded as videos; video_stride keeps every Nth step.
      pixel_cameras      env camera names to render from a state env (e.g. front,
                         ur5e/wrist), served under `cameras` names; null renders nothing.
      pixel_size         their square render size.
      pixel_interval     render every this many steps, holding frames in between.
      video_cameras      the policy cameras (`cameras` names) also recorded as eval videos,
                         beside the third-person one, at `video_stride`; null records every
                         camera the policy reads. Needs `pixel_cameras`.
      holdout            also report the held-out loss (needs dataset.validation_split).
    plus the ChunkDriver knobs every rollout backend takes (execute_len, num_flow_steps,
    cfg_scale, image_size, columns, server, ...).
    """

    def __init__(self, cfg, make_env=_make_env):
        self.cfg = cfg
        self.checkpoint_step = None
        self.sampling = {}
        self._make_env = make_env

    def _env_spec(self):
        import ocbench

        _, backend, _, max_steps = ocbench.parse_env_spec(str(self.cfg.env_name))
        if backend != "mjwarp":
            raise ValueError(
                f"eval env '{self.cfg.env_name}' is a CPU env; the ocbench eval rolls out "
                f"in MJWarp, so name it without `-cpu-`"
            )
        return max_steps

    def _render_kwargs(self):
        cams = self.cfg.get("pixel_cameras")
        if not cams:
            return {}
        size = int(self.cfg.get("pixel_size", 128))
        # a state env draws its info overlays (the goal-target marker) into renders by
        # default; OCBench's visual envs, and so every collected frame, turn them off
        return {"width": size, "height": size, "pixel_cameras": tuple(str(c) for c in cams),
                "visualize_info": bool(self.cfg.get("visualize_info", False))}

    def _obs_features(self, env, info):
        space = env.single_observation_space
        features = {}
        if space.dtype != np.uint8:
            dim = int(space.shape[-1])
            features[STATE_COLUMN] = {"dtype": "float32", "shape": (dim,),
                                      "names": [f"state.{i}" for i in range(dim)]}
            n_cams = len(self.cfg.get("pixel_cameras") or ())
            size = int(self.cfg.get("pixel_size", 128))
            image_shape = (size, size, 3)
        else:
            shape = tuple(space.shape)
            if len(shape) == 3:
                shape = (1, *shape)
            n_cams, image_shape = shape[0], shape[1:]
        for col in INFO_COLUMNS:
            if col in info:
                dim = int(np.asarray(info[col]).shape[-1])
                features[col] = {"dtype": "float32", "shape": (dim,),
                                 "names": [f"{col}.{i}" for i in range(dim)]}
        for name in self._cameras(n_cams) if n_cams else ():
            features[f"{IMAGE_PREFIX}.{name}"] = {"dtype": "video", "shape": image_shape,
                                                  "names": None}
        return features

    def _cameras(self, count):
        names = [str(c) for c in (self.cfg.get("cameras") or DEFAULT_CAMERAS)]
        if len(names) < count:
            raise ValueError(f"the env renders {count} cameras but `cameras:` names {names}")
        return names[:count]

    def _make_driver(self, model, obs_features, num_envs):
        if not self.cfg.get("server"):
            driver = make_driver(model, self.cfg, obs_features)
            self.sampling = driver.sampling
            return driver, BatchDriver(driver, num_envs)
        if num_envs != 1:
            raise ValueError(
                "remote inference is one blocking connection, so it rolls out one world at "
                "a time: set eval.num_envs=1"
            )
        driver = RemoteDriver(self.cfg, obs_features, self.cfg.get("run"))
        self.checkpoint_step = driver.checkpoint_step
        self.sampling = driver.sampling
        return driver, _SerialBatch(driver)

    def run(self, model):
        cfg = self.cfg
        episodes = int(cfg.get("episodes", 100))
        num_envs = max(1, min(int(cfg.get("num_envs") or episodes), episodes))
        max_steps = int(cfg.get("max_episode_steps") or self._env_spec())

        if model is not None:
            model.eval()
        env = self._make_env(str(cfg.env_name), num_envs, **self._render_kwargs())
        rows, videos = [], {}
        driver = None
        try:
            _, info = env.reset(seeds=np.zeros(num_envs, dtype=np.uint32))
            obs_features = self._obs_features(env, info)
            cameras = [c.removeprefix(f"{IMAGE_PREFIX}.") for c in obs_features
                       if c.startswith(IMAGE_PREFIX)]
            driver, batch = self._make_driver(model, obs_features, num_envs)
            action_dim = int(env.single_action_space.shape[-1])
            for start in range(0, episodes, num_envs):
                wave_rows, wave_videos = self._rollout_wave(
                    env, batch, cameras, start, min(num_envs, episodes - start), num_envs,
                    max_steps, action_dim, max_videos=int(cfg.get("max_videos", 2)) - len(videos),
                )
                rows.extend(wave_rows)
                videos.update(wave_videos)
        finally:
            if driver is not None:
                driver.close()
            env.close()

        metrics = self._metrics(rows)
        metrics.update(self._holdout_loss(model))
        return EvalResult(metrics=metrics, videos=videos, episodes=rows)

    def _rollout_wave(self, env, batch, cameras, start, live, num_envs, max_steps,
                      action_dim, max_videos):
        """Episodes `start .. start+live-1` on the first `live` worlds; any spare worlds of
        a short last wave are parked as done from the first step and never scored."""
        cfg = self.cfg
        seed = int(cfg.get("seed", 0))
        stride = max(1, int(cfg.get("video_stride", 4)))
        opened = time.perf_counter()
        seeds = np.array(
            [episode_seed(seed, start + min(w, live - 1)) for w in range(num_envs)], dtype=np.uint32
        )
        observation, info = env.reset(seeds=seeds)
        render = bool(self.cfg.get("pixel_cameras"))
        render_every = max(1, int(cfg.get("pixel_interval", 1)))
        pixels = None
        # OCBench actions are joint deltas, so zero holds the arm where it stands
        batch.reset(np.zeros(action_dim, dtype=np.float32))

        done = np.arange(num_envs) >= live
        success = np.zeros(num_envs, dtype=bool)
        healthy = np.ones(num_envs, dtype=bool)
        steps = np.zeros(num_envs, dtype=int)
        recording = list(range(min(live, max(0, max_videos))))
        clips = {w: [env.render_world(w).copy()] for w in recording}
        # the policy's own view, rendered on the video stride rather than taken from the
        # frames the policy reads, which only change every `pixel_interval` steps
        video_cams = self._video_cameras(cameras) if render else []
        cam_clips = {w: {c: [] for _, c in video_cams} for w in recording}

        def record_cameras(worlds):
            if video_cams and worlds:
                views = env.get_pixel_observation(worlds)
                for w, view in zip(worlds, views):
                    for i, name in video_cams:
                        cam_clips[w][name].append(np.asarray(view[i]).copy())

        record_cameras(recording)

        for tick in range(1, max_steps + 1):
            if render and (tick - 1) % render_every == 0:
                # finished worlds are parked; re-rendering them would only cost time
                live_worlds = np.flatnonzero(~done)
                if len(live_worlds):
                    fresh = env.get_pixel_observation(live_worlds)
                    # a new buffer per render, never an update in place: the driver's
                    # history holds views of the old one, which must keep the old frames
                    pixels = (np.zeros((num_envs, *fresh.shape[1:]), dtype=np.uint8)
                              if pixels is None else pixels.copy())
                    pixels[live_worlds] = fresh
            actions = batch.step(observation_to_frames(observation, cameras, info, pixels))
            actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
            actions[done] = 0.0
            observation, _, terminated, _, info = env.step(actions)
            active = ~done
            steps[active] = tick
            ended = active & (np.asarray(terminated, dtype=bool) | (tick >= max_steps))
            success[ended] = np.asarray(info["success"], dtype=bool)[ended]
            healthy[ended] = np.asarray(info["healthy"], dtype=bool)[ended]
            due = [w for w in recording if active[w] and (tick % stride == 0 or ended[w])]
            for w in due:
                clips[w].append(env.render_world(w).copy())
            record_cameras(due)
            done |= ended
            if done.all():
                break

        rows, videos = [], {}
        for w in range(live):
            episode = start + w
            rows.append({
                "task": str(cfg.env_name),
                "episode": episode,
                "success": bool(success[w]),
                "healthy": bool(healthy[w]),
                "steps": int(steps[w]),
            })
            if w in clips:
                videos[f"ep{episode}"] = np.stack(clips[w]).transpose(0, 3, 1, 2)
            for name, frames in cam_clips.get(w, {}).items():
                videos[f"ep{episode}_{name}"] = np.stack(frames).transpose(0, 3, 1, 2)
        print(f"eval {cfg.env_name} episodes {start}-{start + live - 1}: "
              f"{success[:live].mean():.0%} success, {time.perf_counter() - opened:.0f}s "
              f"on {num_envs} worlds", flush=True)
        return rows, videos

    def _video_cameras(self, cameras):
        """`video_cameras` -> [(camera axis, name)], null meaning every policy camera."""
        wanted = self.cfg.get("video_cameras")
        names = list(cameras) if wanted is None else [str(c) for c in wanted]
        unknown = [n for n in names if n not in cameras]
        if unknown:
            raise ValueError(f"video_cameras {unknown} are not policy cameras ({list(cameras)})")
        return [(cameras.index(n), n) for n in names]

    def _holdout_loss(self, model):
        """Held-out loss alongside the success rate, under the usual `loss` names."""
        if model is None or not self.cfg.get("holdout", True):
            return {}
        from .offline import OfflineEval

        return OfflineEval(self.cfg).run(model).metrics

    @staticmethod
    def _metrics(rows):
        if not rows:
            return {"success_rate": 0.0, "episodes": 0}
        succeeded = [r["steps"] for r in rows if r["success"]]
        return {
            "success_rate": 100.0 * float(np.mean([r["success"] for r in rows])),
            "episodes": len(rows),
            "mean_episode_steps": float(np.mean([r["steps"] for r in rows])),
            "mean_success_steps": float(np.mean(succeeded)) if succeeded else 0.0,
            "unhealthy_episodes": int(sum(not r["healthy"] for r in rows)),
        }
