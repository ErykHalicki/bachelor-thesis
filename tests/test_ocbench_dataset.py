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
    with pytest.raises(ValueError, match="sparse observations"):
        build_dataset(_cfg(tmp_path))
