"""LeRobotEval exercised without hardware: a fake robot supplies lerobot's Robot surface
(observation/action features, get_observation, send_action) and success tags come from a
scripted input(). Covers the obs->field mapping (state vector order, slices, image
permute/resize), chunked execution, human tagging, and the EvalResult shape.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.utils.errors import DeviceNotConnectedError
from omegaconf import OmegaConf

from thesis.experiments.eval import build_eval


class FakeRobot:
    observation_features = {"j1": float, "j2": float, "cam": (48, 64, 3)}
    action_features = {"j1": float, "j2": float}

    def __init__(self):
        self.sent = []
        self.connected = False

    @property
    def is_connected(self):
        return self.connected

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def get_observation(self):
        return {"j1": 1.0, "j2": 2.0, "cam": np.zeros((48, 64, 3), dtype=np.uint8)}

    def send_action(self, action):
        self.sent.append(action)
        return action


class FakePolicy(nn.Module):
    """Chunk policy surface the driver relies on: conditioning spec, obs_len,
    chunk_len/execute_len, which predicted stream holds the actions, and
    predict -> {"action": (B, chunk, dim)}.
    """

    obs_len = 2
    chunk_len = 4
    execute_len = 3
    action_stream = "action"
    action_field = "action"

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.conditioning = {
            "pixels": {"from": "observation.images.cam"},
            "state": {"from": "observation.state"},
        }
        self.seen = []

    def predict(self, obs):
        self.seen.append({k: v.shape for k, v in obs.items()})
        if "observation.images.cam" in obs:
            assert obs["observation.images.cam"].dtype == torch.uint8
        return {"action": torch.arange(self.chunk_len * 2, dtype=torch.float32).reshape(1, self.chunk_len, 2)}


def _cfg(**over):
    # recording is off by default: these tests exercise the driver, and the recorder
    # brings real video encoding with it (covered by its own test below)
    return OmegaConf.create({
        "backend": "lerobot",
        "robot": {"type": "fake"},
        "episodes": 2,
        "fps": 500,
        "max_episode_steps": 6,
        "image_size": [32, 32],
        "record_dataset": False,
        "quiet": True,
        **over,
    })


def panel_named(panels, name):
    """Panels are selected by name rather than position, so a test doesn't break every
    time the rollout grows another one."""
    return next(p for p in panels if p.name == name)


def run_eval(cfg, robot, model, monkeypatch, tags):
    ev = build_eval(cfg)
    monkeypatch.setattr(ev, "_make_robot", lambda: robot)
    answers = iter(tags)
    # under pytest stdin is not a tty, so the keyboard listener is absent by design
    monkeypatch.setattr("thesis.experiments.eval.lerobot.prompt", lambda *a: next(answers))
    return ev.run(model)


def test_rollout_maps_obs_and_tags_success(monkeypatch):
    robot, model, cfg = FakeRobot(), FakePolicy(), _cfg()
    result = run_eval(cfg, robot, model, monkeypatch, ["", "y", "", ""])

    assert result.metrics["success_rate"] == 50.0
    assert result.metrics["episodes"] == 2
    assert [r["success"] for r in result.episodes] == [True, False]

    shapes = model.seen[0]
    assert shapes["observation.state"] == (1, 2, 2)
    assert shapes["observation.images.cam"] == (1, 2, 3, 32, 32)

    assert len(robot.sent) == 12 and set(robot.sent[0]) == {"j1", "j2"}
    assert robot.sent[0] == {"j1": 0.0, "j2": 1.0}
    assert not robot.connected


def test_rollouts_recorded_as_lerobot_dataset(tmp_path, monkeypatch):
    """Kept episodes land on disk as a real LeRobotDataset and the per-camera mp4 comes
    back as a path in result.videos -- the contract log_eval relies on."""
    class BigCamRobot(FakeRobot):
        # SVT-AV1, the recorder's default codec, refuses frames smaller than 64x64
        observation_features = {"j1": float, "j2": float, "cam": (64, 64, 3)}

        def get_observation(self):
            return {"j1": 1.0, "j2": 2.0, "cam": np.zeros((64, 64, 3), dtype=np.uint8)}

    root = tmp_path / "rollouts"
    # fps stays under SVT-AV1's 240 fps ceiling (real control rates are ~30)
    cfg = _cfg(record_dataset=True, record_root=str(root), fps=200)
    result = run_eval(cfg, BigCamRobot(), FakePolicy(), monkeypatch, ["", "y", "", ""])

    assert result.metrics["episodes"] == 2
    video = Path(result.videos["cam"])
    assert video.suffix == ".mp4" and video.exists()
    assert list((root / "data").rglob("*.parquet"))


def test_resume_continues_an_interrupted_session(tmp_path, monkeypatch):
    """A session stopped part-way is picked up in the same rollout dir: its scored episodes
    count toward `episodes`, and the new ones append to the same dataset."""
    class BigCamRobot(FakeRobot):
        observation_features = {"j1": float, "j2": float, "cam": (64, 64, 3)}

        def get_observation(self):
            return {"j1": 1.0, "j2": 2.0, "cam": np.zeros((64, 64, 3), dtype=np.uint8)}

    root = tmp_path / "rollouts"
    common = dict(record_dataset=True, fps=200, episodes=3,
                  record_vcodec="h264", record_streaming_encoding=True)

    first = run_eval(_cfg(record_root=str(root), **common), BigCamRobot(), FakePolicy(),
                     monkeypatch, ["", "q"])
    assert first.metrics["episodes"] == 1
    assert json.loads((root / "eval_session.json").read_text())["episodes"] == first.episodes

    second = run_eval(_cfg(resume=str(root), **common), BigCamRobot(), FakePolicy(),
                      monkeypatch, ["", "y", "", "y"])

    assert second.metrics["episodes"] == 3
    assert [r["episode"] for r in second.episodes] == [0, 1, 2]
    assert [r["success"] for r in second.episodes] == [False, True, True]
    assert second.metrics["success_rate"] == pytest.approx(200 / 3)
    assert LeRobotDatasetMetadata("local/eval_rollouts", root=root).total_episodes == 3


def test_resume_without_a_dataset_is_a_clear_error(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="holds no rollout dataset"):
        run_eval(_cfg(record_dataset=True, resume=str(tmp_path)),
                 FakeRobot(), FakePolicy(), monkeypatch, [])


def test_q_tag_stops_early(monkeypatch):
    robot, model = FakeRobot(), FakePolicy()
    result = run_eval(_cfg(episodes=5), robot, model, monkeypatch, ["", "q"])
    assert result.metrics["episodes"] == 1 and result.metrics["success_rate"] == 0.0


def test_r_tag_discards_and_redoes_the_episode(monkeypatch):
    robot, model = FakeRobot(), FakePolicy()
    result = run_eval(_cfg(episodes=1), robot, model, monkeypatch, ["", "r", "", "y"])
    assert result.metrics["episodes"] == 1
    assert result.metrics["success_rate"] == 100.0
    assert len(robot.sent) == 12


class DroppingRobot(FakeRobot):
    """Loses its motor link part-way into the first episode, the way the b601's follower
    process stops on a CAN bus that went quiet, and rebuilds it on request."""

    def __init__(self, drop_at=3, recoverable=True):
        super().__init__()
        self.drop_at = drop_at
        self.recoverable = recoverable
        self.reads = 0
        self.restarts = 0

    def get_observation(self):
        self.reads += 1
        if self.reads == self.drop_at:
            self.connected = False
        if not self.connected:
            raise DeviceNotConnectedError("FakeRobot is not connected.")
        return super().get_observation()

    def restart_follower_process(self):
        self.restarts += 1
        self.connected = self.recoverable


def test_a_dropped_robot_is_reconnected_and_the_episode_redone(monkeypatch):
    robot, model = DroppingRobot(), FakePolicy()
    cfg = _cfg(episodes=1, robot_recovery_delay_s=0)
    result = run_eval(cfg, robot, model, monkeypatch, ["", "", "y"])

    assert robot.restarts == 1
    # the partial episode is not scored: only the one run on the robot that came back
    assert result.metrics["episodes"] == 1
    assert result.metrics["success_rate"] == 100.0


def test_a_robot_that_stays_down_ends_the_session_with_what_it_scored(monkeypatch):
    robot, model = DroppingRobot(drop_at=8, recoverable=False), FakePolicy()
    cfg = _cfg(episodes=2, robot_recovery_attempts=2, robot_recovery_delay_s=0)
    result = run_eval(cfg, robot, model, monkeypatch, ["", "y", ""])

    assert robot.restarts == 2
    assert result.metrics["episodes"] == 1
    assert result.metrics["success_rate"] == 100.0


def test_sparse_field_offsets_pick_matching_history_rows(monkeypatch):
    # a sparse conditioning window must receive the frames at those offsets, not
    # the oldest rows of the dense history buffer
    class CountingRobot(FakeRobot):
        def __init__(self):
            super().__init__()
            self.t = 0

        def get_observation(self):
            self.t += 1
            return {"j1": float(self.t), "j2": 0.0,
                    "cam": np.zeros((48, 64, 3), dtype=np.uint8)}

    class RecordingPolicy(FakePolicy):
        obs_len = 4

        def __init__(self):
            super().__init__()
            self.conditioning = {"state": {"from": "observation.state"}}
            self.field_offsets = {"observation.state": [-3, 0]}
            self.vals = []

        def predict(self, obs):
            self.vals.append({k: v.clone() for k, v in obs.items()})
            return super().predict(obs)

    robot, model = CountingRobot(), RecordingPolicy()
    run_eval(_cfg(episodes=1), robot, model, monkeypatch, ["", ""])

    assert model.vals[0]["observation.state"].shape == (1, 2, 2)
    # first replan at t=1 pads history by repetition; the second at t=4 has a full
    # buffer, so offsets -3 and 0 must surface j1 = 1.0 and 4.0
    assert model.vals[0]["observation.state"][0, :, 0].tolist() == [1.0, 1.0]
    assert model.vals[1]["observation.state"][0, :, 0].tolist() == [1.0, 4.0]


def test_slices_and_columns_remap(monkeypatch):
    robot, cfg = FakeRobot(), _cfg()
    cfg.columns = {"joints": "observation.state"}
    cfg.slices = {"joints": ["j2"]}
    model = FakePolicy()
    model.conditioning = {"state": {"from": "joints"}}
    result = run_eval(cfg, robot, model, monkeypatch, ["", "", "", ""])
    assert model.seen[0]["joints"] == (1, 2, 1)
    assert result.metrics["success_rate"] == 0.0


def test_unknown_column_is_a_clear_error(monkeypatch):
    robot, model = FakeRobot(), FakePolicy()
    model.conditioning = {"state": {"from": "observation.ghost"}}
    with pytest.raises(ValueError, match="observation.ghost"):
        run_eval(_cfg(), robot, model, monkeypatch, [""])


def test_camera_stream_starts_on_the_live_cameras_and_closes_before_disconnect(monkeypatch):
    """read_latest() raises once the cameras are disconnected, so the preview has to be
    down first."""
    order = []

    class StreamingRobot(FakeRobot):
        cameras = {"cam": object()}

        def disconnect(self):
            order.append("disconnect")
            super().disconnect()

    class FakeStream:
        url = "http://fake:8090/"

        def close(self):
            order.append("close")

    started = {}

    def fake_start(panels, port=None):
        started["panels"] = panels
        started["port"] = port
        return FakeStream()

    monkeypatch.setattr("thesis.utils.stream.start_stream", fake_start)
    robot = StreamingRobot()
    run_eval(_cfg(episodes=1), robot, FakePolicy(), monkeypatch, ["", "y"])

    camera = panel_named(started["panels"], "cameras")
    assert (camera.kind, camera.cameras) == ("video", StreamingRobot.cameras)
    assert (camera.height, camera.quality) == (360, 95)
    assert started["port"] == 8090
    assert order == ["close", "disconnect"]


def test_metrics_are_grouped_by_motor_field_and_subsampled(monkeypatch):
    """The plots follow the robot's own feature names -- `<motor>.<field>` becomes one
    plot per field with a line per motor -- so no series list has to be maintained."""
    class JointRobot(FakeRobot):
        observation_features = {"j1.pos": float, "j2.pos": float,
                                "j1.torq": float, "j2.torq": float}
        action_features = {"j1.pos": float, "j2.pos": float}
        cameras = {}

        def get_observation(self):
            return {"j1.pos": 1.0, "j2.pos": 2.0, "j1.torq": 0.5, "j2.torq": 0.25}

    captured = {}

    def fake_start(panels, port=None):
        captured["metrics"] = panel_named(panels, "observation")
        return None

    monkeypatch.setattr("thesis.utils.stream.start_stream", fake_start)
    model = FakePolicy()
    model.conditioning = {"state": {"from": "observation.state"}}
    cfg = _cfg(episodes=1, stream_metrics_fps=250)
    run_eval(cfg, JointRobot(), model, monkeypatch, ["", "y"])

    snap = captured["metrics"].snapshot()
    assert set(snap["series"]) == {
        "state.pos/j1", "state.pos/j2", "state.torq/j1", "state.torq/j2",
        "action.pos/j1", "action.pos/j2",
    }
    assert len(snap["t"]) == 3
    assert snap["series"]["state.torq/j1"] == [0.5, 0.5, 0.5]


def test_latency_panel_charts_each_replan(monkeypatch):
    """Replans happen every execute_len steps, so the latency panel samples on its own
    cadence rather than the observation panel's."""
    seen = {}

    def fake_start(panels, port=None):
        seen["latency"] = next(p for p in panels if p.name == "latency")
        return None

    monkeypatch.setattr("thesis.utils.stream.start_stream", fake_start)
    run_eval(_cfg(episodes=1), FakeRobot(), FakePolicy(), monkeypatch, ["", "y"])

    snap = seen["latency"].snapshot()
    assert list(snap["series"]) == ["replan.ms/compute"]
    assert len(snap["t"]) == 2
    assert all(v > 0 for v in snap["series"]["replan.ms/compute"])


def test_remote_driver_splits_the_round_trip_by_the_servers_own_timing(monkeypatch):
    """The server reports its compute, so the rest of the round trip is the wire. Both
    are durations on their own machine's clock, so no clock agreement is needed."""
    from thesis.experiments.eval.chunking import RemoteDriver

    driver = RemoteDriver.__new__(RemoteDriver)
    driver.visual = set()
    driver.quality = 95
    driver._latency = None
    monkeypatch.setattr(driver, "_request", lambda msg: {"actions": [[0.0]], "compute_s": 0.4})

    driver.predict_window([{"observation.state": np.zeros(1)}])
    spans = driver.take_latency()

    assert set(spans) == {"encode", "compute", "network"}
    assert spans["compute"] == 0.4
    # the request is instant here, so the server's 0.4s exceeds the round trip;
    # the wire share floors at zero rather than going negative
    assert spans["network"] == 0.0
    assert driver.take_latency() is None


def test_panels_are_opt_out_individually(monkeypatch):
    """Each panel is independent: turning one off must not take the others with it."""
    seen = {}

    def fake_start(panels, port=None):
        seen["names"] = [p.name for p in panels]
        return None

    monkeypatch.setattr("thesis.utils.stream.start_stream", fake_start)
    run_eval(_cfg(episodes=1), FakeRobot(), FakePolicy(), monkeypatch, ["", "y"])
    assert seen["names"] == ["observation", "latency"]

    run_eval(_cfg(episodes=1, stream_metrics=False), FakeRobot(), FakePolicy(),
             monkeypatch, ["", "y"])
    assert seen["names"] == ["latency"]

    run_eval(_cfg(episodes=1, stream_latency=False), FakeRobot(), FakePolicy(),
             monkeypatch, ["", "y"])
    assert seen["names"] == ["observation"]


def test_sampling_overrides_reach_the_model(monkeypatch):
    """A post-hoc eval rebuilds the model from the config its run stored, so the eval
    config is the only place left to change how it samples."""
    robot, model = FakeRobot(), FakePolicy()
    model.num_flow_steps, model.cfg_scale = 10, 1.0
    cfg = _cfg(num_flow_steps=2, cfg_scale=1.5)
    run_eval(cfg, robot, model, monkeypatch, ["", "", "", ""])
    assert (model.num_flow_steps, model.cfg_scale) == (2, 1.5)


def test_omitted_sampling_keeps_the_trained_value(monkeypatch):
    robot, model = FakeRobot(), FakePolicy()
    model.num_flow_steps = 10
    run_eval(_cfg(), robot, model, monkeypatch, ["", "", "", ""])
    assert model.num_flow_steps == 10


def test_zero_flow_steps_is_a_clear_error(monkeypatch):
    robot, model = FakeRobot(), FakePolicy()
    model.num_flow_steps = 10
    with pytest.raises(ValueError, match="num_flow_steps"):
        run_eval(_cfg(num_flow_steps=0), robot, model, monkeypatch, [""])
