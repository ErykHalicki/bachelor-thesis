"""Stream panels exercised without a robot: the grouping rule that decides which plots
exist, the ring buffer's eviction, the json shape the page reads, and the server's
panel dispatch -- which is what keeps a new kind of panel from touching the server.
"""

import json
import urllib.error
import urllib.request

import numpy as np
import pytest
from omegaconf import OmegaConf

from thesis.utils.stream.metrics import LeRobotObservationPanel, TimeSeriesPanel, group_key
from thesis.utils.stream.server import StreamPanel, start_stream


def test_group_key_splits_motor_from_field():
    assert group_key("action", "joint1.pos") == "action.pos/joint1"
    assert group_key("state", "joint7.torq") == "state.torq/joint7"
    assert group_key("state", "gripper") == "state/gripper"


def test_snapshot_transposes_rows_onto_a_shared_time_axis():
    m = TimeSeriesPanel()
    m.push({"a/x": 1.0, "a/y": 2.0}, t=10.0)
    m.push({"a/x": 3.0, "a/y": 4.0}, t=10.1)
    snap = m.snapshot()

    assert snap["t"] == [0.0, 0.1]
    assert snap["series"] == {"a/x": [1.0, 3.0], "a/y": [2.0, 4.0]}


def test_series_appearing_late_is_padded_with_nulls():
    m = TimeSeriesPanel()
    m.push({"a/x": 1.0}, t=0.0)
    m.push({"a/x": 2.0, "a/y": 9.0}, t=0.1)
    assert m.snapshot()["series"] == {"a/x": [1.0, 2.0], "a/y": [None, 9.0]}


def test_non_finite_values_are_dropped_not_raised():
    m = TimeSeriesPanel()
    m.push({"a/x": float("nan"), "a/y": float("inf"), "a/z": 1.0, "a/w": "no"}, t=0.0)
    # json.dumps would emit a bare NaN, which no browser's JSON.parse accepts
    assert m.snapshot()["series"] == {"a/z": [1.0]}
    json.loads(m.json())


def test_window_evicts_the_oldest_rows():
    m = TimeSeriesPanel(window=3)
    for i in range(5):
        m.push({"a/x": float(i)}, t=float(i))
    snap = m.snapshot()
    assert snap["series"]["a/x"] == [2.0, 3.0, 4.0]
    assert snap["t"] == [0.0, 1.0, 2.0]


def test_charts_carry_their_own_uplot_spec():
    """A ChartPanel defines its plots in python -- series, labels, colours -- so the
    page renders them without a renderer per panel."""
    m = LeRobotObservationPanel()
    m.push({group_key("state", "joint2.pos"): 5.0,
            group_key("state", "joint10.pos"): 6.0,
            group_key("state", "joint2.torq"): 0.5}, t=0.0)
    charts = {c["name"]: c for c in m.charts()}

    assert sorted(charts) == ["state.pos", "state.torq"]
    pos = charts["state.pos"]
    assert [s.get("label") for s in pos["spec"]["series"]] == [None, "joint2", "joint10"]
    assert pos["spec"]["series"][1]["stroke"] != pos["spec"]["series"][2]["stroke"]
    assert pos["data"] == [[0.0], [5.0], [6.0]]
    assert (pos["spec"]["precision"], charts["state.torq"]["spec"]["precision"]) == (1, 2)


def test_a_metric_without_a_field_gets_its_own_chart():
    m = TimeSeriesPanel()
    m.push({"loss": 0.25}, t=0.0)
    (chart,) = m.charts()
    assert chart["name"] == "loss"
    assert chart["data"] == [[0.0], [0.25]]


def test_observation_panel_names_its_charts_from_lerobot_features():
    """The charts follow the robot's own feature names, so an embodiment that reports
    different motors or fields plots differently with nothing to reconfigure."""
    panel = LeRobotObservationPanel()
    panel.track_features(
        {"observation.state": {"names": ["j1.pos", "j2.pos", "j1.torq", "j2.torq"]}},
        ["j1.pos", "j2.pos"],
    )
    panel.push_step({"observation.state": [1.0, 2.0, 0.5, 0.25]}, [9.0, 8.0], t=0.0)

    assert [c["name"] for c in panel.charts()] == ["action.pos", "state.pos", "state.torq"]
    assert panel.snapshot()["series"]["action.pos/j1"] == [9.0]
    assert panel.snapshot()["series"]["state.torq/j2"] == [0.25]


def test_push_raw_takes_teleop_dicts_and_drops_camera_frames():
    """record.py sees the robot's and teleoperator's own dicts rather than the dataset
    frames a rollout builds, so those need no features tracked first."""
    panel = LeRobotObservationPanel()
    panel.push_raw(
        {"j1.pos": 1.0, "j1.torq": 0.5, "front": np.zeros((4, 4, 3), dtype=np.uint8)},
        {"j1.pos": 1.2},
        t=0.0,
    )
    assert sorted(panel.snapshot()["series"]) == [
        "action.pos/j1", "state.pos/j1", "state.torq/j1",
    ]


def test_dry_run_returns_before_the_dataset_root_is_touched(tmp_path, monkeypatch):
    """A rehearsal must not create, resume or archive a dataset folder -- the real
    session that follows has to find exactly what it expected to."""
    import thesis.scripts.b601.record as record

    called = {}
    for name in ("LeRobotDataset", "LeRobotDatasetMetadata", "archive_dataset_root"):
        monkeypatch.setattr(record, name, _forbidden(name, called))
    monkeypatch.setattr(record, "build_cameras", lambda cams: {})
    monkeypatch.setattr(record, "build_arms", lambda cfg, cams: ("robot", "teleop"))
    monkeypatch.setattr(record, "make_default_processors", lambda: (None, None, None))
    monkeypatch.setattr(record, "teleoperate", lambda *a, **k: called.setdefault("teleop", True))

    cfg = OmegaConf.structured(record.RecordConfig)
    cfg.dry_run = True
    cfg.repo_id = "someone/should-not-be-created"
    cfg.root = str(tmp_path / "never")
    cfg.stream_metrics = False
    record.main.__wrapped__(cfg)

    assert called == {"teleop": True}
    assert not (tmp_path / "never").exists()


def _forbidden(name, called):
    def fail(*args, **kwargs):
        called[name] = True
        raise AssertionError(f"dry run reached {name}")
    return fail


def test_record_tap_passes_values_through_untouched():
    """record_loop only calls its processors, so tapping one needs no hook in lerobot --
    but the tapped value must reach it unchanged."""
    from thesis.scripts.b601.record import tap

    seen = []
    processor = tap(lambda pair: {"j1.pos": pair[0]["j1.pos"] * 2}, seen.append)
    out = processor(({"j1.pos": 3.0}, "ignored"))

    assert out == {"j1.pos": 6.0}
    assert seen == [out]


def test_observation_panel_is_inert_until_features_are_known():
    # the panel is built before the rollout knows the robot's features; pushing in
    # that window must not raise
    panel = LeRobotObservationPanel()
    panel.push_step({"observation.state": [1.0]}, [2.0])
    assert panel.charts() == []


class NotePanel(StreamPanel):
    """A panel kind the server has never heard of, which is the point: a new thing to
    watch should need nothing but its own class and a renderer on the page."""

    kind = "note"

    def serve(self, handler):
        handler.send_bytes(b"hello", "text/plain")

    def close(self):
        self.closed = True


@pytest.fixture
def served():
    """A server on an ephemeral port, torn down after the test."""
    servers = []

    def start(panels):
        # port 0 lets the OS pick a free one, so tests never collide with a real stream
        server = start_stream(panels, port=0)
        servers.append(server)
        return server, f"http://127.0.0.1:{server.server.server_address[1]}"

    yield start
    for server in servers:
        server.close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.read()


def test_manifest_describes_every_panel_and_routes_to_it(served):
    metrics = TimeSeriesPanel(name="m")
    metrics.push({"a/x": 1.0}, t=0.0)
    _, base = served([NotePanel("n"), metrics])

    _, body = get(f"{base}/panels.json")
    assert json.loads(body)["panels"] == [
        {"name": "n", "kind": "note", "src": "/panel/n", "spec": None},
        {"name": "m", "kind": "chart", "src": "/panel/m", "spec": None},
    ]
    assert get(f"{base}/panel/n")[1] == b"hello"
    assert json.loads(get(f"{base}/panel/m")[1])["charts"][0]["data"] == [[0.0], [1.0]]
    assert get(f"{base}/")[1].startswith(b"<!doctype html>")


def test_external_assets_are_served_and_nothing_else_is(served):
    _, base = served([NotePanel("n")])
    status, body = get(f"{base}/external/uplot.js")
    assert status == 200 and b"uPlot" in body
    assert get(f"{base}/external/uplot.css")[0] == 200
    for path in ("/external/../page.py", "/external/uplot.LICENSE", "/external/"):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(f"{base}{path}")
        assert e.value.code == 404


def test_unknown_route_is_404(served):
    _, base = served([NotePanel("n")])
    with pytest.raises(urllib.error.HTTPError) as e:
        get(f"{base}/panel/nope")
    assert e.value.code == 404


def test_close_closes_every_panel(served):
    panel = NotePanel("n")
    server, _ = served([panel])
    server.close()
    assert panel.closed
