"""Remote inference (eval `server:`) exercised end to end over a real loopback socket.

A background thread runs scripts/serve.py's connection handler with checkpoint loading
stubbed out, so the wire protocol, the JPEG round trip, and the client/server split of the
observation pipeline are all real. The bar throughout is that a remote rollout drives the
robot with exactly the actions a local one would, and that nothing the robot box lacks
(model, norm stats, column layout) is ever needed on its side.
"""

import socket
import threading

import numpy as np
import pytest
import torch

from thesis.experiments.eval import build_eval
from thesis.scripts import serve
from thesis.utils.smoothing import smooth_actions
from thesis.utils.wire import decode_jpeg, encode_jpeg

from test_lerobot_eval import FakePolicy, FakeRobot, _cfg


class RecordingPolicy(FakePolicy):
    """FakePolicy that keeps the tensors it was handed, not just their shapes."""

    def __init__(self):
        super().__init__()
        self.values = []

    def predict(self, obs):
        self.values.append({k: v.clone() for k, v in obs.items()})
        return super().predict(obs)


class Server:
    """A serve.py server on an ephemeral port, serving one model without touching wandb."""

    def __init__(self, model):
        self.model = model
        self.step = 4200
        self.runs = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.address = "127.0.0.1:%d" % self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def load_model(self, run_path, device, cache, overrides=()):
        self.runs.append(run_path)
        self.overrides = list(overrides)
        return self.model, self.step

    def _accept_loop(self):
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            try:
                serve.serve_client(conn, "cpu", {})
            finally:
                conn.close()

    def close(self):
        self._sock.close()
        self._thread.join(timeout=2)


@pytest.fixture
def server(monkeypatch):
    inst = Server(RecordingPolicy())
    monkeypatch.setattr(serve, "load_model", inst.load_model)
    try:
        yield inst
    finally:
        inst.close()


def remote_cfg(server, **over):
    return _cfg(server=server.address, run="me/proj/abcd1234", **over)


def run_eval(cfg, robot, model, monkeypatch, tags):
    ev = build_eval(cfg)
    monkeypatch.setattr(ev, "_make_robot", lambda: robot)
    answers = iter(tags)
    monkeypatch.setattr("thesis.experiments.eval.lerobot.prompt", lambda *a: next(answers))
    return ev, ev.run(model)


def test_remote_rollout_matches_local(server, monkeypatch):
    """Same robot, same policy, same tags: the remote path must send the same actions."""
    local_robot = FakeRobot()
    _, local = run_eval(_cfg(), local_robot, FakePolicy(), monkeypatch, ["", "y", "", ""])

    remote_robot = FakeRobot()
    _, result = run_eval(remote_cfg(server), remote_robot, None, monkeypatch, ["", "y", "", ""])

    assert remote_robot.sent == local_robot.sent
    assert result.metrics == local.metrics
    assert result.videos.keys() == local.videos.keys()
    assert server.runs == ["me/proj/abcd1234"]


def test_server_reports_the_checkpoint_step(server, monkeypatch):
    ev, _ = run_eval(remote_cfg(server, episodes=1), FakeRobot(), None, monkeypatch, ["", "y"])
    assert ev.checkpoint_step == 4200


def test_images_survive_the_wire_as_uint8_tensors(server, monkeypatch):
    """The server hands the model the same (1, T, C, H, W) uint8 batch the local driver
    would, decoded and resized on its side."""
    run_eval(remote_cfg(server, episodes=1), FakeRobot(), None, monkeypatch, ["", "y"])
    assert server.model.seen[0]["observation.images.cam"] == (1, 2, 3, 32, 32)
    assert server.model.seen[0]["observation.state"] == (1, 2, 2)


def test_jpeg_round_trip_preserves_rgb_channel_order():
    """Channels must not come back swapped: the encoder converts RGB->BGR for cv2 and the
    decoder has to undo it, or every remote rollout runs on colour-flipped frames."""
    rgb = np.zeros((16, 24, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    out = decode_jpeg(encode_jpeg(rgb, quality=100))
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert out[..., 0].mean() > 150 and out[..., 2].mean() < 50


def test_only_the_conditioned_offsets_cross_the_wire(server, monkeypatch):
    """A sparse conditioning window over a long history costs one frame per offset, not
    one per buffered tick -- the reason the server dictates the offsets."""
    server.model.obs_len = 8
    server.model.conditioning = {"pixels": {"from": "observation.images.cam"}}
    server.model.field_offsets = {"observation.images.cam": [-7, 0]}

    decoded = []
    original = serve.decode_jpeg

    def counting_decode(data):
        decoded.append(len(data))
        return original(data)

    monkeypatch.setattr(serve, "decode_jpeg", counting_decode)
    run_eval(remote_cfg(server, episodes=1), FakeRobot(), None, monkeypatch, ["", "y"])

    assert len(decoded) == 4
    assert server.model.seen[0]["observation.images.cam"] == (1, 2, 3, 32, 32)


def test_slices_are_applied_server_side(server, monkeypatch):
    """The client ships whole columns and the server slices, so a `slices:` change needs
    no agreement about vector layout across the link."""
    cfg = remote_cfg(server)
    cfg.columns = {"joints": "observation.state"}
    cfg.slices = {"joints": ["j2"]}
    server.model.conditioning = {"state": {"from": "joints"}}
    run_eval(cfg, FakeRobot(), None, monkeypatch, ["", "", "", ""])
    assert server.model.seen[0]["joints"] == (1, 2, 1)


def test_execute_len_override_crosses_the_wire(server, monkeypatch):
    """The server decides the horizon (it owns the model), so an eval-side override has to
    reach it -- and the client must buffer exactly that many actions between replans."""
    robot = FakeRobot()
    cfg = remote_cfg(server, episodes=1, max_episode_steps=FakePolicy.chunk_len)
    cfg.execute_len = FakePolicy.chunk_len
    run_eval(cfg, robot, None, monkeypatch, ["", "y"])

    assert len(robot.sent) == FakePolicy.chunk_len
    assert len(server.model.values) == 1


class SamplingPolicy(RecordingPolicy):
    """A policy carrying the inference-time sampling knobs, as a flow model does."""

    def __init__(self):
        super().__init__()
        self.num_flow_steps = 10
        self.cfg_scale = 1.0


def test_sampling_overrides_cross_the_wire(monkeypatch):
    """The model is rebuilt server-side from the config its run stored, so an eval-side
    sampling override only exists if it reaches the server's model."""
    inst = Server(SamplingPolicy())
    monkeypatch.setattr(serve, "load_model", inst.load_model)
    try:
        cfg = remote_cfg(inst, episodes=1, num_flow_steps=2, cfg_scale=1.5)
        ev, _ = run_eval(cfg, FakeRobot(), None, monkeypatch, ["", "y"])
        assert inst.model.num_flow_steps == 2
        assert inst.model.cfg_scale == 1.5
        assert ev.sampling == {"num_flow_steps": 2, "cfg_scale": 1.5}
    finally:
        inst.close()


def test_omitted_sampling_restores_the_trained_value(monkeypatch):
    """The server caches one model across sessions, so a knob the next client leaves unset
    must fall back to what the run trained with, not inherit the previous client's."""
    inst = Server(SamplingPolicy())
    monkeypatch.setattr(serve, "load_model", inst.load_model)
    try:
        run_eval(remote_cfg(inst, episodes=1, num_flow_steps=2), FakeRobot(), None,
                 monkeypatch, ["", "y"])
        assert inst.model.num_flow_steps == 2

        run_eval(remote_cfg(inst, episodes=1), FakeRobot(), None, monkeypatch, ["", "y"])
        assert inst.model.num_flow_steps == 10
    finally:
        inst.close()


def test_action_filter_is_applied_server_side(server, monkeypatch):
    """`action_filter:` is a client knob but the chunk only exists on the server, so it
    has to cross the wire -- an unplumbed knob would silently roll out unfiltered."""
    zigzag = torch.tensor([[0.0, 0.0], [3.0, -3.0], [0.0, 0.0], [3.0, -3.0]]).unsqueeze(0)
    server.model.predict = lambda obs: {"action": zigzag.clone()}

    raw, filtered = FakeRobot(), FakeRobot()
    run_eval(remote_cfg(server, episodes=1), raw, None, monkeypatch, ["", "y"])
    cfg = remote_cfg(server, episodes=1)
    cfg.action_filter = {"enabled": True, "window": 5, "poly_order": 2}
    run_eval(cfg, filtered, None, monkeypatch, ["", "y"])

    assert raw.sent != filtered.sent
    expected = smooth_actions(zigzag, window=5, poly_order=2)[0, : FakePolicy.execute_len]
    for sent, want in zip(filtered.sent, expected, strict=False):
        assert sent["j1"] == pytest.approx(float(want[0]), abs=1e-5)
        assert sent["j2"] == pytest.approx(float(want[1]), abs=1e-5)


def test_normalization_happens_on_the_server(server, monkeypatch):
    """Norm stats live in the checkpoint, so they are never needed on the robot box:
    observations go out raw and actions come back in raw units."""
    server.model.norm_stats = {
        "observation.state": {"mean": [1.0, 1.0], "std": [2.0, 2.0]},
        "action": {"mean": [10.0, 10.0], "std": [2.0, 2.0]},
    }
    server.model.norm_method = "mean_std"
    robot = FakeRobot()
    run_eval(remote_cfg(server, episodes=1), robot, None, monkeypatch, ["", "y"])

    seen = server.model.values[0]["observation.state"]
    assert torch.allclose(seen[0, -1], torch.tensor([0.0, 0.5]))
    assert robot.sent[0] == {"j1": 10.0, "j2": 12.0}


def test_server_error_reaches_the_client_with_its_message(server, monkeypatch):
    server.model.conditioning = {"state": {"from": "observation.ghost"}}
    with pytest.raises(RuntimeError, match="observation.ghost"):
        run_eval(remote_cfg(server), FakeRobot(), None, monkeypatch, [""])


def test_server_survives_a_failed_client_and_serves_the_next(server, monkeypatch):
    """A bad request costs that client an error, not the loaded model: an operator should
    be able to fix their config and reconnect."""
    broken = dict(server.model.conditioning)
    server.model.conditioning = {"state": {"from": "observation.ghost"}}
    with pytest.raises(RuntimeError):
        run_eval(remote_cfg(server), FakeRobot(), None, monkeypatch, [""])

    server.model.conditioning = broken
    _, result = run_eval(remote_cfg(server, episodes=1), FakeRobot(), None, monkeypatch, ["", "y"])
    assert result.metrics["success_rate"] == 100.0


def test_missing_run_is_a_clear_error(server, monkeypatch):
    with pytest.raises(ValueError, match="load="):
        run_eval(_cfg(server=server.address, run=None), FakeRobot(), None, monkeypatch, [""])


def test_model_overrides_cross_the_wire(monkeypatch):
    """`eval.model_overrides` (and the client's own algorithm.* hydra overrides) reach
    the server's load_model, which builds the model with them -- the server stays
    config-free while the client picks e.g. policy-only inference."""
    inst = Server(FakePolicy())
    monkeypatch.setattr(serve, "load_model", inst.load_model)
    try:
        cfg = remote_cfg(inst, episodes=1,
                         model_overrides=["algorithm.inference_streams=[action]"])
        run_eval(cfg, FakeRobot(), None, monkeypatch, ["", "y"])
        assert inst.overrides == ["algorithm.inference_streams=[action]"]
    finally:
        inst.close()
