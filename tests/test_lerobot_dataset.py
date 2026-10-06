import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("lerobot")

from thesis.datasets import build_dataset  # noqa: E402

JOINTS = ["shoulder_pan", "elbow_flex", "gripper"]
STATE_NAMES = [f"{j}.{s}" for s in ("pos", "torq", "vel") for j in JOINTS]
ACTION_NAMES = [f"{j}.pos" for j in JOINTS]


@pytest.fixture
def dataset_root(tmp_path):
    """A tiny video-free LeRobot dataset whose state column carries per-dimension feature
    names, laid out like a real robot recording (`shoulder_pan.pos`, `gripper.vel`, ...).
    Frame f of episode e holds state[d] = 100*e + f + d/100, so a sliced batch can be
    checked back against the dimensions it should have kept.
    """
    from lerobot.datasets import LeRobotDataset

    root = tmp_path / "fake_arm"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
    }
    ds = LeRobotDataset.create("test/fake_arm", fps=10, features=features, root=root)
    for episode in range(2):
        for frame in range(8):
            base = 100 * episode + frame
            ds.add_frame(
                {
                    "observation.state": np.array(
                        [base + d / 100 for d in range(len(STATE_NAMES))], dtype=np.float32
                    ),
                    "action": np.array(
                        [-(base + d / 100) for d in range(len(ACTION_NAMES))], dtype=np.float32
                    ),
                    "task": "fake",
                }
            )
        ds.save_episode()
    ds.finalize()
    return root


@pytest.fixture
def image_dataset_root(tmp_path):
    """A tiny LeRobot dataset with an `image` (PNG-backed) camera column, so the visual
    path is exercised without a video decoder. Every pixel of frame f of episode e is
    10*e + f, making a frame identifiable from any single pixel.
    """
    from lerobot.datasets import LeRobotDataset

    root = tmp_path / "fake_cam"
    features = {
        "observation.image": {
            "dtype": "image",
            "shape": (8, 10, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
    }
    ds = LeRobotDataset.create("test/fake_cam", fps=10, features=features, root=root)
    for episode in range(2):
        for frame in range(6):
            ds.add_frame(
                {
                    "observation.image": np.full((8, 10, 3), 10 * episode + frame, dtype=np.uint8),
                    "action": np.zeros(len(ACTION_NAMES), dtype=np.float32),
                    "task": "fake",
                }
            )
        ds.save_episode()
    ds.finalize()
    return root


def _cfg(dataset_root, **overrides):
    cfg = {
        "backend": "lerobot",
        "repo_id": "test/fake_arm",
        "root": str(dataset_root),
        "conditioning": {"observation.state": {"index": "-1..0"}},
        "predict": {"action": {"index": "0..1"}},
    }
    return OmegaConf.create({**cfg, **overrides})


def test_unsliced_columns_keep_every_dimension(dataset_root):
    batch = build_dataset(_cfg(dataset_root))[0]
    assert batch["observation.state"].shape == (2, len(STATE_NAMES))
    assert batch["action"].shape == (2, len(ACTION_NAMES))


def test_slice_keeps_named_dimensions_in_pattern_order(dataset_root):
    dataset = build_dataset(
        _cfg(dataset_root, slices={"observation.state": "gripper.vel, *.pos"})
    )
    full = build_dataset(_cfg(dataset_root))[3]["observation.state"]
    sliced = dataset[3]["observation.state"]
    assert sliced.shape == (2, 4)
    assert (sliced == full[:, [8, 0, 1, 2]]).all()


def test_overlapping_patterns_are_deduplicated(dataset_root):
    sliced = build_dataset(_cfg(dataset_root, slices={"action": ["*.pos", "gripper.pos"]}))[0]
    assert sliced["action"].shape == (2, len(ACTION_NAMES))


def test_one_column_splits_into_several_sliced_fields(dataset_root):
    dataset = build_dataset(
        _cfg(
            dataset_root,
            conditioning={
                "state_pos": {"index": "-1..0"},
                "state_vel": {"index": "-1..0"},
            },
            columns={"state_pos": "observation.state", "state_vel": "observation.state"},
            slices={"state_pos": "*.pos", "state_vel": "*.vel"},
        )
    )
    batch = dataset[3]
    full = build_dataset(_cfg(dataset_root))[3]["observation.state"]
    assert (batch["state_pos"] == full[:, 0:3]).all()
    assert (batch["state_vel"] == full[:, 6:9]).all()


def test_normalization_stats_follow_the_sliced_width(dataset_root):
    dataset = build_dataset(
        _cfg(
            dataset_root,
            slices={"observation.state": "*.torq"},
            normalize={"keys": ["observation.state"], "method": "min_max"},
        )
    )
    assert len(dataset.stats["observation.state"]["min"]) == 3
    assert dataset[0]["observation.state"].shape == (2, 3)


@pytest.mark.parametrize(
    ("slices", "message"),
    [
        ({"observation.state": "*.effort"}, "matches no dimension"),
        ({"observation.state": "gripper.pos, nope"}, "matches no dimension"),
        ({"observation.velocity": "*.vel"}, "no spec entry reads"),
    ],
)
def test_bad_slice_fails_at_build_time(dataset_root, slices, message):
    with pytest.raises(ValueError, match=message):
        build_dataset(_cfg(dataset_root, slices=slices))


def test_fields_on_one_column_may_use_different_windows(dataset_root):
    dataset = build_dataset(
        _cfg(
            dataset_root,
            conditioning={
                "state_pos": {"index": "-1..0"},
                "state_torq": {"index": "0"},
            },
            columns={"state_pos": "observation.state", "state_torq": "observation.state"},
            slices={"state_pos": "*.pos", "state_torq": "*.torq"},
        )
    )
    batch = dataset[3]
    full = build_dataset(_cfg(dataset_root))[3]["observation.state"]
    assert (batch["state_pos"] == full[:, 0:3]).all()
    assert batch["state_torq"].shape == (1, 3)
    assert (batch["state_torq"] == full[1:, 3:6]).all()


def _episode_final_state(offset):
    """Frame 7 (the last) of the episode starting at `offset`, per the fixture's layout."""
    return np.array([offset + 7 + d / 100 for d in range(len(STATE_NAMES))], dtype=np.float32)


def test_last_appends_the_episode_final_frame(dataset_root):
    dataset = build_dataset(
        _cfg(dataset_root, conditioning={"observation.state": {"index": "-1..0, last"}})
    )
    batch = dataset[1]
    assert batch["observation.state"].shape == (3, len(STATE_NAMES))
    assert (batch["observation.state"][:2] == build_dataset(_cfg(dataset_root))[1][
        "observation.state"
    ]).all()
    assert np.allclose(batch["observation.state"][2].numpy(), _episode_final_state(0))
    assert np.allclose(dataset[7]["observation.state"][2].numpy(), _episode_final_state(100))


def test_last_only_field_is_a_single_goal_row(dataset_root):
    dataset = build_dataset(
        _cfg(
            dataset_root,
            conditioning={
                "observation.state": {"index": "-1..0"},
                "goal_state": {"index": "last"},
            },
            columns={"goal_state": "observation.state"},
            slices={"goal_state": "*.pos"},
        )
    )
    batch = dataset[0]
    assert batch["observation.state"].shape == (2, len(STATE_NAMES))
    assert batch["goal_state"].shape == (1, 3)
    assert np.allclose(batch["goal_state"][0].numpy(), _episode_final_state(0)[0:3])


def test_goal_frame_joins_a_resized_visual_window(image_dataset_root):
    cfg = OmegaConf.create({
        "backend": "lerobot",
        "repo_id": "test/fake_cam",
        "root": str(image_dataset_root),
        "image_size": [4, 5],
        "conditioning": {"observation.image": {"index": "-1..0, last"}},
        "predict": {"action": {"index": "0..1"}},
    })
    frames = build_dataset(cfg)[1]["observation.image"]
    assert frames.shape == (3, 3, 4, 5)
    values = [round(float(frame.flatten()[0]) * 255) for frame in frames]
    assert values == [0, 1, 5]


def test_episode_start_pads_by_repeating_the_first_frame(image_dataset_root):
    # a rollout has no history on its first ticks and repeats the frame it opened
    # on, so the opening decision points are kept and padded the same way
    cfg = OmegaConf.create({
        "backend": "lerobot",
        "repo_id": "test/fake_cam",
        "root": str(image_dataset_root),
        "conditioning": {"observation.image": {"index": "-2..0"}},
        "predict": {"action": {"index": "0..1"}},
    })
    dataset = build_dataset(cfg)
    assert len(dataset) == 10

    def frames(idx):
        return [round(float(f.flatten()[0]) * 255) for f in dataset[idx]["observation.image"]]

    assert frames(0) == [0, 0, 0]
    assert frames(1) == [0, 0, 1]
    assert frames(2) == [0, 1, 2]
    # the pad clamps within the sample's own episode, never across the boundary
    assert frames(5) == [10, 10, 10]


def test_window_running_past_the_episode_end_is_still_dropped(image_dataset_root):
    cfg = OmegaConf.create({
        "backend": "lerobot",
        "repo_id": "test/fake_cam",
        "root": str(image_dataset_root),
        "conditioning": {"observation.image": {"index": "0"}},
        "predict": {"action": {"index": "0..2"}},
    })
    assert len(build_dataset(cfg)) == 8
    assert len(build_dataset(OmegaConf.create({**cfg, "drop_boundary": False}))) == 12


def test_raw_index_selects_the_rows_an_index_step_is_built_from(image_dataset_root):
    # 2 index steps built from 4 raw frames each: the backend loads raw_index, not index
    cfg = OmegaConf.create({
        "backend": "lerobot",
        "repo_id": "test/fake_cam",
        "root": str(image_dataset_root),
        "conditioning": {"observation.image": {"index": "-1, 0", "raw_index": "-3..0"}},
        "predict": {"action": {"index": "0..1"}},
    })
    dataset = build_dataset(cfg)
    assert len(dataset) == 10
    frames = dataset[3]["observation.image"]
    assert frames.shape == (4, 3, 8, 10)
    assert [round(float(f.flatten()[0]) * 255) for f in frames] == [0, 1, 2, 3]


def test_augment_wraps_a_real_backend_without_changing_the_batch_contract(image_dataset_root):
    cfg = {
        "backend": "lerobot",
        "repo_id": "test/fake_cam",
        "root": str(image_dataset_root),
        "conditioning": {"observation.image": {"index": "-1..0"}},
        "predict": {"action": {"index": "0..1"}},
    }
    # sample 1, not 0: episode 0 opens on an all-black frame, which crops and brightness
    # scales back to itself, so augmenting it is not visible in the batch
    plain = build_dataset(OmegaConf.create(cfg))[1]
    augmented = build_dataset(OmegaConf.create({
        **cfg,
        "augment": {
            "seed": 0,
            "streams": {
                "observation.*": {"random_crop": {"scale": 0.5},
                                  "color_jitter": {"brightness": 0.5}},
                "action": {"gaussian_noise": {"std": 0.1}},
            },
        },
    }))[1]

    assert set(augmented) == set(plain)
    for field in plain:
        if isinstance(plain[field], str):
            # the task string rides along unaugmented
            assert augmented[field] == plain[field]
            continue
        assert augmented[field].shape == plain[field].shape
        assert augmented[field].dtype == plain[field].dtype
        assert not np.array_equal(augmented[field].numpy(), plain[field].numpy())


def test_last_does_not_restrict_decision_points(dataset_root):
    without = build_dataset(_cfg(dataset_root, predict={"action": {"index": "0"}}))
    with_last = build_dataset(
        _cfg(
            dataset_root,
            predict={"action": {"index": "0"}},
            conditioning={"observation.state": {"index": "-1..0, last"}},
        )
    )
    assert len(with_last) == len(without)


def test_stats_come_from_the_data_not_the_shipped_file(dataset_root):
    """Stats come from the data, not meta/stats.json, whose quantiles are per-episode
    averages and so much narrower than the global quantile."""
    dataset = build_dataset(
        _cfg(
            dataset_root,
            slices={"observation.state": "*.torq"},
            normalize={"keys": ["action", "observation.state"], "method": "percentile",
                       "percentiles": [1, 99]},
        )
    )
    action = dataset.stats["action"]
    assert set(action) >= {"mean", "std", "min", "max", "q_lo", "q_hi"}
    # action[d] = -(100*e + f + d/100) over e in {0,1}, f in 0..7 -> dim 0 spans [-107, 0]
    assert action["min"][0] == -107.0 and action["max"][0] == 0.0
    # sliced field gets sliced stats, same width as the data
    assert len(dataset.stats["observation.state"]["q_lo"]) == 3


def test_stats_columns_avoid_the_item_path(dataset_root, monkeypatch):
    """The stats pass must read columns, not build windows: `__getitem__` decodes every
    video key in `delta_timestamps`, and the cameras are there whenever the algorithm
    uses them."""
    from thesis.datasets.lerobot import LeRobotSource

    def boom(self, idx):
        raise AssertionError("stats pass built a window instead of reading columns")

    monkeypatch.setattr(LeRobotSource, "__getitem__", boom)
    dataset = build_dataset(
        _cfg(
            dataset_root,
            normalize={"keys": ["action"], "method": "percentile", "percentiles": [1, 99]},
        )
    )
    assert len(dataset.stats["action"]["q_lo"]) == len(ACTION_NAMES)


def test_any_percentile_pair_is_available(dataset_root):
    """Any percentile pair works, not only the ones meta/stats.json ships."""
    dataset = build_dataset(
        _cfg(
            dataset_root,
            normalize={"keys": ["action"], "method": "percentile", "percentiles": [5, 95]},
        )
    )
    assert len(dataset.stats["action"]["q_lo"]) == len(ACTION_NAMES)
