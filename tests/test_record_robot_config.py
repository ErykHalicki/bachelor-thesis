"""Recording the arm settings a dataset was captured under, in meta/info.json.

robot_type says which arm; this says how it was driven. Gains, control mode and
gravity compensation decide what state a given action actually produces, so two
datasets off the same arm are only interchangeable if these match -- and a
resumed dataset whose settings have moved is quietly mixing regimes.

The round-trip matters as much as the capture: info.json is rewritten on every
metadata update, so a field that doesn't survive write/load would be erased by
the first saved episode.
"""

import json

import numpy as np
import pytest

pytest.importorskip("lerobot")
pytest.importorskip("motorbridge")
record = pytest.importorskip("thesis.scripts.b601.record")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.robots.rebot_b601_follower import (  # noqa: E402
    RebotB601Follower,
    RebotB601FollowerRobotConfig,
)
from lerobot.teleoperators.rebot_102_leader import (  # noqa: E402
    RebotArm102Leader,
    RebotArm102LeaderTeleopConfig,
)


@pytest.fixture
def settings():
    robot = RebotB601Follower(RebotB601FollowerRobotConfig(port="/dev/null", id="follower"))
    teleop = RebotArm102Leader(RebotArm102LeaderTeleopConfig(port="/dev/null", id="leader"))
    return record.recording_settings(robot, teleop)


@pytest.fixture
def dataset(tmp_path):
    """A real on-disk dataset, so info.json is written and re-read for real."""
    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }
    return LeRobotDataset.create(
        "test/robot_config",
        fps=30,
        root=tmp_path / "ds",
        features=features,
        robot_type="rebot_b601_follower",
        use_videos=False,
    )


def test_the_settings_are_json_writable(settings):
    """info.json is plain JSON, and configs hold dataclasses, Paths and tuples."""
    json.dumps(settings)


def test_the_settings_that_shape_the_data_are_captured(settings):
    robot = settings["robot"]
    assert robot["type"] == "rebot_b601_follower"
    for key in record.CONTROL_KEYS:
        assert key in robot, f"{key} decides what an action means but was not recorded"


def test_an_unset_config_leaves_info_json_clean(dataset):
    """Datasets that never set it must not grow a null key."""
    assert dataset.meta.robot_config is None
    written = json.loads((dataset.root / "meta" / "info.json").read_text())
    assert "robot_config" not in written


def test_the_config_survives_the_write_load_round_trip(dataset, settings):
    dataset.meta.robot_config = settings

    written = json.loads((dataset.root / "meta" / "info.json").read_text())
    assert written["robot_config"] == settings
    from lerobot.datasets.io_utils import load_info

    assert load_info(dataset.root).robot_config == settings


def test_the_config_is_not_erased_by_a_later_metadata_write(dataset, settings):
    """info.json is rewritten whenever counters move; the field has to persist."""
    dataset.meta.robot_config = settings
    dataset.meta.info.total_episodes += 1
    from lerobot.datasets.io_utils import write_info

    write_info(dataset.meta.info, dataset.meta.root)

    written = json.loads((dataset.root / "meta" / "info.json").read_text())
    assert written["robot_config"] == settings


def test_the_stored_config_is_a_copy(dataset, settings):
    dataset.meta.robot_config = settings
    got = dataset.meta.robot_config
    got["robot"]["control_mode"] = "mutated"
    assert dataset.meta.robot_config["robot"]["control_mode"] != "mutated"


@pytest.mark.parametrize(
    "key, value",
    [
        ("gravity_compensation", False),
        ("gravity_gain", 1.2),
        ("gravity_scale", {"elbow_flex": 1.3}),
        ("control_mode", "pos_vel"),
        ("mit_kp", [1.0] * 7),
        ("smoothing_time_constant_s", 0.2),
        ("joint_limits", {"shoulder_pan": [-10.0, 10.0]}),
    ],
)
def test_a_changed_control_setting_is_reported(settings, key, value):
    stored = json.loads(json.dumps(settings))
    stored["robot"][key] = value
    assert record.control_differences(stored, settings) == [key]


@pytest.mark.parametrize("key, value", [("port", "/dev/ttyACM9"), ("profile_log_path", "/tmp/x")])
def test_settings_that_dont_shape_the_data_are_ignored(settings, key, value):
    """Ports and log paths move between sessions without touching the episodes."""
    stored = json.loads(json.dumps(settings))
    stored["robot"][key] = value
    assert record.control_differences(stored, settings) == []


def test_identical_settings_report_no_difference(settings):
    assert record.control_differences(settings, settings) == []


def test_a_dataset_with_no_stored_config_reports_no_difference(settings):
    """Datasets recorded before this existed must stay appendable without a prompt."""
    assert record.control_differences(None, settings) == []
    assert record.control_differences({}, settings) == []


def test_jsonable_handles_what_configs_actually_contain():
    from dataclasses import dataclass
    from pathlib import Path

    @dataclass
    class Inner:
        limits: tuple[float, float]

    @dataclass
    class Outer:
        nested: dict[str, Inner]
        where: Path | None
        count: int

    got = record.jsonable(Outer(nested={"a": Inner((-1.0, 1.0))}, where=Path("/tmp/x"), count=3))
    assert got == {"nested": {"a": {"limits": [-1.0, 1.0]}}, "where": "/tmp/x", "count": 3}
    json.dumps(got)


def test_recorded_duration_counts_the_whole_dataset_not_one_episode():
    """num_frames is the dataset total, so resumed sessions keep accumulating."""
    import types

    assert record.recorded_duration(types.SimpleNamespace(num_frames=1800, fps=30)) == "00:01:00"
    assert record.recorded_duration(types.SimpleNamespace(num_frames=3600, fps=30)) == "00:02:00"


def test_jsonable_stringifies_rather_than_raising():
    """Failing to record the settings must never be what ends a recording."""
    got = record.jsonable({"odd": np.dtype("float32")})
    assert isinstance(got["odd"], str)
    json.dumps(got)
