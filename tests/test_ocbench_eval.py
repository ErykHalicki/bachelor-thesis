from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from thesis.experiments.eval.ocbench import OCBenchEval, episode_seed, observation_to_frames

MAX_STEPS = 6


def _solves(seed):
    """The fake task: an even seed succeeds on step 3, an odd one never does."""
    return int(seed) % 2 == 0


class FakeEnv:
    """A batched OCBench-shaped env. Each world's observation is its reset seed, and it
    records every action it is stepped with."""

    single_observation_space = SimpleNamespace(dtype=np.dtype(np.float64), shape=(3,))
    single_action_space = SimpleNamespace(shape=(7,))

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.actions = []
        self.renders = []
        self.closed = False

    def reset(self, seeds):
        self.seeds = np.asarray(seeds)
        self.t = 0
        info = {"qpos": np.zeros((len(self.seeds), 4)), "qvel": np.zeros((len(self.seeds), 2))}
        return np.repeat(self.seeds[:, None].astype(np.float64), 3, axis=1), info

    def step(self, actions):
        self.actions.append(np.array(actions))
        self.t += 1
        success = np.array([_solves(s) and self.t >= 3 for s in self.seeds])
        obs = np.repeat(self.seeds[:, None].astype(np.float64), 3, axis=1)
        info = {"success": success, "healthy": np.ones(self.num_envs, dtype=bool)}
        return obs, np.zeros(self.num_envs), success, np.zeros(self.num_envs, bool), info

    def get_pixel_observation(self, worlds):
        self.renders.append((self.t, list(worlds)))
        return np.full((len(worlds), 2, 4, 5, 3), self.t, dtype=np.uint8)

    def render_world(self, world):
        return np.zeros((4, 5, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


class StubBatch:
    """A policy that always commands 5.0 on every joint."""

    def __init__(self):
        self.resets = 0
        self.frames = []

    def reset(self, hold_action=None):
        self.resets += 1
        assert hold_action is not None and not hold_action.any()

    def step(self, frames):
        assert all(frame["observation.state"].dtype == np.float32 for frame in frames)
        self.frames.append(frames)
        return np.full((len(frames), 7), 5.0, dtype=np.float32)


def _run(episodes=5, num_envs=2, **overrides):
    cfg = OmegaConf.create({
        "env_name": "block-single-task1-v0",
        "episodes": episodes,
        "num_envs": num_envs,
        "max_episode_steps": MAX_STEPS,
        "holdout": False,
        "max_videos": 1,
        "video_stride": 2,
        **overrides,
    })
    envs = []

    def make_env(name, n, **kwargs):
        envs.append(FakeEnv(n))
        envs[-1].kwargs = kwargs
        return envs[-1]

    backend = OCBenchEval(cfg, make_env=make_env)
    batch = StubBatch()
    backend._make_driver = lambda model, features, n: (SimpleNamespace(close=lambda: None), batch)
    return backend.run(None), envs[0], batch


def test_each_episode_is_scored_from_its_own_seed():
    result, _, _ = _run()
    expected = [_solves(episode_seed(0, e)) for e in range(5)]
    assert [r["success"] for r in result.episodes] == expected
    assert [r["episode"] for r in result.episodes] == list(range(5))
    assert result.metrics["success_rate"] == 100.0 * np.mean(expected)
    for row in result.episodes:
        assert row["steps"] == (3 if row["success"] else MAX_STEPS)


def test_scores_do_not_depend_on_the_batch_size():
    serial, _, _ = _run(num_envs=1)
    batched, _, _ = _run(num_envs=5)
    assert serial.episodes == batched.episodes


def test_actions_are_clipped_and_finished_worlds_held():
    _, env, batch = _run(episodes=2, num_envs=2)
    assert batch.resets == 1
    first = env.actions[0]
    assert np.all(first == 1.0)
    # past step 3 a world that succeeded is parked with zero actions
    solved = [_solves(s) for s in env.seeds]
    for actions in env.actions[3:]:
        for world, done in enumerate(solved):
            assert np.all(actions[world] == 0.0) if done else np.all(actions[world] == 1.0)


def test_spare_worlds_of_a_short_wave_are_not_scored():
    result, env, _ = _run(episodes=3, num_envs=2)
    assert len(result.episodes) == 3
    assert result.metrics["episodes"] == 3
    assert env.closed


def test_videos_are_frames_first_channels_second():
    result, _, _ = _run()
    assert list(result.videos) == ["ep0"]
    video = result.videos["ep0"]
    assert video.shape[1:] == (3, 4, 5)


def test_visual_observations_split_into_camera_columns():
    pixels = np.zeros((2, 3, 4, 5, 3), dtype=np.uint8)
    pixels[:, 1] = 7
    frames = observation_to_frames(pixels, ["front", "side", "wrist"])
    assert sorted(frames[0]) == [
        "observation.images.front", "observation.images.side", "observation.images.wrist",
    ]
    assert frames[1]["observation.images.side"].shape == (4, 5, 3)
    assert np.all(frames[1]["observation.images.side"] == 7)


def test_cameras_render_every_interval_and_hold_in_between():
    result, env, batch = _run(
        episodes=4, num_envs=4, pixel_cameras=["front", "ur5e/wrist"], pixel_size=4,
        pixel_interval=2, cameras=["front", "wrist"],
    )
    assert env.kwargs == {"width": 4, "height": 4, "pixel_cameras": ("front", "ur5e/wrist"),
                          "visualize_info": False}
    # rendered before steps 1, 3 and 5 (env time 0, 2, 4) -- the last only for worlds
    # still running, and solved worlds finish on step 3
    solved = [_solves(s) for s in env.seeds]
    running = [w for w, s in enumerate(solved) if not s]
    assert env.renders[:2] == [(0, [0, 1, 2, 3]), (2, [0, 1, 2, 3])]
    assert env.renders[2:3] == ([(4, running)] if running else [])
    held = [frames[0]["observation.images.wrist"][0, 0, 0] for frames in batch.frames[:4]]
    assert held == [0, 0, 2, 2]
    first = batch.frames[0][0]
    assert first["qpos"].shape == (4,) and first["qvel"].dtype == np.float32
