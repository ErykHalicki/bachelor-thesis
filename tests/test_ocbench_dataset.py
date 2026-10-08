import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from thesis.datasets import build_dataset

# (length, success) per episode of the fixture file
EPISODES = [(5, True), (4, False), (6, True)]
OBS_DIM = 3
ACTION_DIM = 2


def _write(path, episodes, visual=False):
    """An OCBench-format .npz: one action row per step, one observation row per step plus
    each episode's final one. Step t of episode e carries action[0] = 100*e + t and state
    observation[0] = 100*e + t (the final observation is 100*e + L), so a window can be
    read back off its values. A visual file stores two cameras whose pixels are that same
    number, plus 50 on the second camera.
    """
    actions, observations, rewards, terminals, masks = [], [], [], [], []
    for e, (length, success) in enumerate(episodes):
        steps = 100 * e + np.arange(length + 1, dtype=np.float32)
        actions.append(np.repeat(steps[:length, None], ACTION_DIM, axis=1))
        if visual:
            frames = np.zeros((length + 1, 2, 4, 6, 3), dtype=np.uint8)
            frames[:, 0] = (steps % 50)[:, None, None, None]
            frames[:, 1] = (steps % 50 + 50)[:, None, None, None]
            observations.append(frames)
        else:
            observations.append(np.repeat(steps[:, None], OBS_DIM, axis=1).astype(np.float64))
        reward = np.zeros(length, dtype=np.float32)
        terminal = np.zeros(length, dtype=bool)
        terminal[-1] = True
        mask = np.ones(length, dtype=np.float32)
        if success:
            reward[-1] = 1.0
            mask[-1] = 0.0
        rewards.append(reward)
        terminals.append(terminal)
        masks.append(mask)
    np.savez(
        path,
        observations=np.concatenate(observations),
        actions=np.concatenate(actions),
        rewards=np.concatenate(rewards),
        terminals=np.concatenate(terminals),
        masks=np.concatenate(masks),
        observation_interval=np.asarray(1, dtype=np.int32),
    )


@pytest.fixture
def data_dir(tmp_path):
    _write(tmp_path / "shard-0.npz", EPISODES)
    # OCBench's own validation file beside a shard must not be read as training data
    _write(tmp_path / "shard-0-val.npz", [(3, True)])
    return tmp_path


def _cfg(path, **overrides):
    cfg = {
        "backend": "ocbench",
        "env_name": "block-single-task1-v0",
        "dataset_path": str(path),
        "conditioning": {"observation.state": {"index": "-1..0"}},
        "predict": {"action": {"index": "0..1"}},
    }
    return OmegaConf.create({**cfg, **overrides})


def test_success_only_drops_failed_episodes(data_dir):
    dataset = build_dataset(_cfg(data_dir))
    assert dataset.episodes.tolist() == [0, 2]
    # drop_boundary: an action window reaching one step ahead loses each episode's last step
    assert len(dataset) == (5 - 1) + (6 - 1)


def test_success_only_off_keeps_every_episode(data_dir):
    dataset = build_dataset(_cfg(data_dir, success_only=False))
    assert dataset.episodes.tolist() == [0, 1, 2]


def test_windows_read_the_right_rows(data_dir):
    dataset = build_dataset(_cfg(data_dir))
    batch = dataset[2]
    assert batch["observation.state"].dtype == torch.float32
    assert batch["observation.state"][:, 0].tolist() == [1.0, 2.0]
    assert batch["action"][:, 0].tolist() == [2.0, 3.0]
    # the first step of the second kept episode (episode 2) clamps its history onto step 0
    second = dataset[4]
    assert second["observation.state"][:, 0].tolist() == [200.0, 200.0]
    assert second["action"][:, 0].tolist() == [200.0, 201.0]


def test_last_is_the_final_observation_and_the_last_action(data_dir):
    dataset = build_dataset(_cfg(
        data_dir,
        conditioning={"observation.state": {"index": "0, last"}},
        predict={"action": {"index": "0, last"}},
    ))
    batch = dataset[0]
    assert batch["observation.state"][:, 0].tolist() == [0.0, 5.0]
    assert batch["action"][:, 0].tolist() == [0.0, 4.0]


def test_observation_windows_may_reach_the_final_observation(data_dir):
    dataset = build_dataset(_cfg(
        data_dir,
        conditioning={"observation.state": {"index": "0..1"}},
        predict={"action": {"index": "0"}},
    ))
    last_step = dataset[4]
    assert last_step["observation.state"][:, 0].tolist() == [4.0, 5.0]


def test_episode_split_is_disjoint_and_seeded(data_dir):
    _write(data_dir / "shard-0.npz", [(4, True)] * 10)
    train = build_dataset(_cfg(data_dir, validation_split=0.2), split="train")
    val = build_dataset(_cfg(data_dir, validation_split=0.2), split="val")
    assert len(val.episodes) == 2
    assert set(train.episodes) | set(val.episodes) == set(range(10))
    assert not set(train.episodes) & set(val.episodes)


def test_normalization_stats_are_per_step(data_dir):
    dataset = build_dataset(_cfg(data_dir, normalize={"keys": ["action", "observation.state"]}))
    raw = build_dataset(_cfg(data_dir))
    actions = raw.stats_columns(["action"])["action"]
    states = raw.stats_columns(["observation.state"])["observation.state"]
    # one row per action step of the kept episodes; the final observations are not counted
    assert len(actions) == len(states) == 5 + 6
    assert np.allclose(dataset.stats["action"]["mean"], actions.mean(axis=0))


def test_visual_cameras_are_separate_columns(tmp_path):
    _write(tmp_path / "visual.npz", EPISODES, visual=True)
    dataset = build_dataset(_cfg(
        tmp_path,
        cameras=["front", "wrist"],
        image_size=[2, 3],
        conditioning={
            "observation.images.front": {"index": "-1..0"},
            "observation.images.wrist": {"index": "0"},
        },
    ))
    batch = dataset[2]
    assert batch["observation.images.front"].shape == (2, 3, 2, 3)
    assert batch["observation.images.front"].dtype == torch.uint8
    assert batch["observation.images.front"][:, 0, 0, 0].tolist() == [1, 2]
    assert batch["observation.images.wrist"][:, 0, 0, 0].tolist() == [52]
    assert dataset.stats_columns(["observation.images.front"]) is None


def test_slices_keep_named_dimensions_of_a_step_column(tmp_path):
    _write_sparse(tmp_path / "shard-000.npz", [(12, True)], interval=5)
    dataset = build_dataset(_sparse_cfg(
        tmp_path, slices={"qpos": "qpos.0, qpos.[2-3]"},
        normalize={"keys": ["qpos"], "method": "mean_std"},
    ))
    assert dataset[1]["qpos"].shape == (2, 3)
    assert len(dataset.stats["qpos"]["mean"]) == 3


@pytest.mark.parametrize(
    ("slices", "message"),
    [
        ({"qpos": "qpos.9"}, "matches no dimension"),
        ({"qvel": "qvel.0"}, "no spec entry reads"),
        ({"observation.images.front": "x"}, "no per-dimension names"),
    ],
)
def test_bad_slices_fail_at_build_time(tmp_path, slices, message):
    _write_sparse(tmp_path / "shard-000.npz", [(12, True)], interval=5)
    with pytest.raises(ValueError, match=message):
        build_dataset(_sparse_cfg(tmp_path, slices=slices))


def test_a_field_no_column_serves_fails_at_build_time(data_dir):
    with pytest.raises(ValueError, match="none of the requested modalities"):
        build_dataset(_cfg(
            data_dir,
            conditioning={"observation.images.front": {"index": "0"}},
            predict={"reward": {"index": "0"}},
        ))


def test_sparse_datasets_are_refused(tmp_path):
    _write(tmp_path / "shard.npz", EPISODES)
    data = dict(np.load(tmp_path / "shard.npz"))
    data["observation_interval"] = np.asarray(5, dtype=np.int32)
    np.savez(tmp_path / "shard.npz", **data)
    with pytest.raises(ValueError, match="sparse state observations"):
        build_dataset(_cfg(tmp_path))


def _write_sparse(path, episodes, interval):
    """The layout scripts/collect_ocbench_visual.py writes: dense per-step columns in the
    .npz, and frames every `interval` steps (plus each episode's final one) in a memory-
    mapped `-pixels.npy`. Step t of episode e has qpos[0] = 100*e + t, and the frame of
    step s has every pixel 10*e + s // interval on camera 0, +100 on camera 1 (the final
    frame uses 10*e + 9).
    """
    actions, qpos, rewards, terminals, frames = [], [], [], [], []
    for e, (length, success) in enumerate(episodes):
        steps = 100 * e + np.arange(length, dtype=np.float32)
        actions.append(np.repeat(steps[:, None], ACTION_DIM, axis=1))
        qpos.append(np.repeat(steps[:, None], 4, axis=1))
        reward = np.zeros(length, dtype=np.float32)
        reward[-1] = float(success)
        terminal = np.zeros(length, dtype=bool)
        terminal[-1] = True
        rewards.append(reward)
        terminals.append(terminal)
        values = [10 * e + s // interval for s in range(0, length, interval)] + [10 * e + 9]
        for v in values:
            frame = np.zeros((2, 4, 6, 3), dtype=np.uint8)
            frame[0], frame[1] = v, v + 100
            frames.append(frame)
    np.savez(
        path,
        actions=np.concatenate(actions),
        qpos=np.concatenate(qpos),
        rewards=np.concatenate(rewards),
        terminals=np.concatenate(terminals),
        masks=np.ones(sum(length for length, _ in episodes), dtype=np.float32),
        observation_interval=np.asarray(interval, dtype=np.int32),
    )
    np.save(str(path)[: -len(".npz")] + "-pixels.npy", np.stack(frames))


def _sparse_cfg(path, **overrides):
    base = {
        "cameras": ["front", "wrist"],
        "conditioning": {
            "qpos": {"index": "-1..0"},
            "observation.images.front": {"index": "-5, 0"},
        },
        "predict": {
            "action": {"index": "0..4"},
            "observation.images.wrist": {"index": "5"},
        },
    }
    return _cfg(path, **{**base, **overrides})


def test_sparse_frames_land_on_stored_rows(tmp_path):
    # episode 0: 12 steps, frames at 0, 5, 10 and the final one (step 12)
    _write_sparse(tmp_path / "shard-000.npz", [(12, True), (7, True)], interval=5)
    dataset = build_dataset(_sparse_cfg(tmp_path))
    # decision points are multiples of 5 whose action window fits: 0, 5 (ep 0), 0 (ep 1)
    assert len(dataset) == 3
    second = dataset[1]                     # episode 0, t = 5
    assert second["qpos"][:, 0].tolist() == [4.0, 5.0]
    assert second["action"][:, 0].tolist() == [5.0, 6.0, 7.0, 8.0, 9.0]
    assert second["observation.images.front"][:, 0, 0, 0].tolist() == [0, 1]
    assert second["observation.images.front"].shape == (2, 3, 4, 6)
    assert second["observation.images.wrist"][:, 0, 0, 0].tolist() == [102]
    # episode 1 (7 steps, t = 0): its future frame at step 5 is a stored row, and the
    # history before step 0 clamps onto the first frame
    third = dataset[2]
    assert third["observation.images.front"][:, 0, 0, 0].tolist() == [10, 10]
    assert third["observation.images.wrist"][:, 0, 0, 0].tolist() == [111]


def test_a_future_frame_past_the_end_is_the_final_observation(tmp_path):
    _write_sparse(tmp_path / "shard-000.npz", [(7, True)], interval=5)
    dataset = build_dataset(_sparse_cfg(
        tmp_path,
        predict={"action": {"index": "0"}, "observation.images.wrist": {"index": "10"}},
        drop_boundary=False,
    ))
    first = dataset[0]
    assert first["observation.images.wrist"][:, 0, 0, 0].tolist() == [109]


def test_camera_offsets_must_be_multiples_of_the_frame_interval(tmp_path):
    _write_sparse(tmp_path / "shard-000.npz", [(12, True)], interval=5)
    with pytest.raises(ValueError, match="multiple"):
        build_dataset(_sparse_cfg(
            tmp_path, conditioning={"observation.images.front": {"index": "-1..0"}},
        ))


def test_sparse_frames_are_memory_mapped(tmp_path):
    _write_sparse(tmp_path / "shard-000.npz", [(12, True)], interval=5)
    dataset = build_dataset(_sparse_cfg(tmp_path))
    assert isinstance(dataset._frames[0], np.memmap)
    assert dataset.stats_columns(["qpos"])["qpos"].shape == (12, 4)


def test_max_episodes_caps_the_kept_episodes_before_the_split(data_dir):
    dataset = build_dataset(_cfg(data_dir, success_only=False, max_episodes=2))
    assert dataset.episodes.tolist() == [0, 1]
    with pytest.raises(ValueError, match="only 2 episodes"):
        build_dataset(_cfg(data_dir, max_episodes=3))
