"""Contract tests for the B601 follower, against a stand-in CAN bus.

The follower hands its serial device to a separate process once connected, so
anything that has to talk to the motors directly afterwards must take it back
first. Calibration is the case that matters: lerobot-calibrate calls
connect(calibrate=False) and *then* calibrate(), by which point the direct
connection is gone.
"""

import builtins
import itertools
import types

import numpy as np
import pytest

pytest.importorskip("lerobot")
pytest.importorskip("motorbridge")
follower_module = pytest.importorskip(
    "lerobot.robots.rebot_b601_follower.rebot_b601_follower"
)
gm = pytest.importorskip("lerobot.robots.rebot_b601_follower.gravity_model")

from lerobot.robots.rebot_b601_follower import (  # noqa: E402
    RebotB601Follower,
    RebotB601FollowerRobotConfig,
)


class FakeMotor:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def set_zero_position(self):
        self.log.append(("zeroed", self.name))

    def close(self):
        pass

    def clear_error(self):
        pass

    def ensure_mode(self, mode):
        pass


class FakeBus:
    def __init__(self, log):
        self.log = log
        log.append(("bus_open", None))

    def add_damiao_motor(self, send_id, recv_id, model):
        return FakeMotor(f"{send_id:#x}", self.log)

    def disable_all(self):
        self.log.append(("disable_all", None))

    def close(self):
        self.log.append(("bus_close", None))


@pytest.fixture
def robot(monkeypatch):
    """A follower whose bus, follower process and calibration file are all fakes.

    Returns (robot, log) where log is the ordered record of what it did.
    """
    log: list[tuple] = []
    monkeypatch.setattr(
        follower_module,
        "MotorBridgeController",
        types.SimpleNamespace(from_dm_serial=lambda serial_port, baud: FakeBus(log)),
    )
    monkeypatch.setattr(follower_module.time, "sleep", lambda *_: None)
    monkeypatch.setattr(builtins, "input", lambda *_: "")

    bot = RebotB601Follower(RebotB601FollowerRobotConfig(port="/dev/null", id="test"))
    monkeypatch.setattr(bot, "_start_follower_process", lambda: log.append(("proc_start", None)))
    monkeypatch.setattr(bot, "_stop_follower_process", lambda: log.append(("proc_stop", None)))
    monkeypatch.setattr(bot, "_save_calibration", lambda: log.append(("saved", None)))
    return bot, log


def steps(log) -> list[str]:
    return [entry[0] for entry in log]


def zeroed(log) -> list[str]:
    return [name for kind, name in log if kind == "zeroed"]


def test_calibrating_during_connect_zeroes_every_motor(robot):
    bot, log = robot
    bot.connect(calibrate=True)
    assert zeroed(log) == [f"{i:#x}" for i in range(1, 8)]


def test_calibrating_after_connect_zeroes_every_motor(robot):
    """lerobot-calibrate's order. This used to raise AttributeError on a None
    bus -- and had it got past that, self.motors was empty too, so it would have
    saved a calibration file having zeroed nothing at all."""
    bot, log = robot
    bot.connect(calibrate=False)
    bot._follower_process = types.SimpleNamespace(is_alive=lambda: True, pid=1)
    log.clear()

    bot.calibrate()

    assert zeroed(log) == [f"{i:#x}" for i in range(1, 8)]
    order = steps(log)
    assert order.index("proc_stop") < order.index("bus_open") < order.index("zeroed")
    assert order.index("zeroed") < order.index("bus_close") < order.index("proc_start")
    assert order[-1] == "saved"


def test_calibration_covers_every_configured_joint(robot):
    bot, log = robot
    bot.connect(calibrate=True)
    assert set(bot.calibration) == set(bot.config.motor_can_ids)
    for name, entry in bot.calibration.items():
        low, high = bot.config.joint_limits[name]
        assert (entry.range_min, entry.range_max) == (int(low), int(high))
        assert entry.homing_offset == 0


def test_the_direct_connection_is_returned_even_if_calibration_raises(robot):
    bot, log = robot
    bot.connect(calibrate=False)
    bot._follower_process = types.SimpleNamespace(is_alive=lambda: True, pid=1)
    log.clear()

    def boom(*_):
        raise KeyboardInterrupt("user gave up at the prompt")

    with pytest.raises(KeyboardInterrupt):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(builtins, "input", boom)
            bot.calibrate()

    assert bot.bus is None and bot.motors == {}
    assert steps(log)[-1] == "proc_start", "the follower process must come back"


def test_connect_hands_the_device_to_the_follower_process(robot):
    bot, log = robot
    bot.connect(calibrate=True)
    order = steps(log)
    assert order.index("bus_close") < order.index("proc_start")
    assert bot.bus is None and bot.motors == {}


GRIPPER = follower_module.GRIPPER_MOTOR
WRIST = follower_module.WRIST_MOTOR


@pytest.fixture
def homing_ramp():
    """The home trajectory with short legs, plus every goal the follower
    process would take from it, tick by tick.

    The close keeps enough ticks (0.2s at send_rate_hz) to see the wrist descend
    gradually rather than in a couple of jumps; the real default is 2s.
    """
    config = RebotB601FollowerRobotConfig(
        port="/dev/null", id="test", home_duration_s=0.05, gripper_close_duration_s=0.2
    )
    motors = list(config.motor_can_ids)
    ramp = follower_module._HomeRamp(config, motors, dict.fromkeys(motors, 0.0))
    tick = 1.0 / config.send_rate_hz
    ticks = [(i * tick, ramp.target(i * tick)) for i in range(int(ramp.duration / tick) + 1)]
    return config, ramp, ticks


def leg(ticks, start, end):
    """One leg's samples, including the boundary ticks that begin and end it --
    those are the ones that reach the pose the leg was aiming for."""
    eps = 1e-9
    return [goal for elapsed, goal in ticks if start - eps <= elapsed <= end + eps]


def test_homing_keeps_commanding_the_final_pose_after_the_ramp(homing_ramp):
    """The gripper and wrist are still travelling when their goals stop moving,
    and disconnect() kills the follower process the moment go_home returns --
    so the ramp has to be followed by a hold, or they are cut off part-way."""
    config, ramp, ticks = homing_ramp
    settled = dict.fromkeys(config.motor_can_ids, 0.0)

    tail = list(itertools.takewhile(lambda t: t[1] == settled, reversed(ticks)))
    assert len(tail) > 10, f"only {len(tail)} ticks held the final pose"
    assert len(tail) / config.send_rate_hz >= follower_module._GRIPPER_SETTLE_SEC * 0.8


def test_homing_lifts_the_wrist_clear_while_the_arm_comes_in(homing_ramp):
    """Coming in flat swings whatever is on the end into the shoulder at ramp
    speed. The wrist holds clear until the arm has arrived."""
    config, ramp, ticks = homing_ramp
    approach = leg(ticks, 0.0, config.home_duration_s)
    lift = config.home_wrist_flex_deg
    assert lift < 0, "negative wrist_flex is what points the gripper up on this arm"
    assert ramp.target(config.home_duration_s)[WRIST] == pytest.approx(lift)
    assert min(goal[WRIST] for goal in approach) == pytest.approx(lift)
    low, high = config.joint_limits[WRIST]
    assert low <= lift <= high


def test_the_wrist_lowers_over_the_whole_gripper_close(homing_ramp):
    """Set down across the full close rather than dropped at the end of it."""
    config, ramp, ticks = homing_ramp
    close = leg(
        ticks, config.home_duration_s, config.home_duration_s + config.gripper_close_duration_s
    )
    wrist = [goal[WRIST] for goal in close]
    gripper = [goal[GRIPPER] for goal in close]
    assert wrist[0] == pytest.approx(config.home_wrist_flex_deg)
    assert wrist[-1] == pytest.approx(0.0)
    partway = [w for w in wrist if config.home_wrist_flex_deg * 0.9 < w < -1.0]
    assert len(partway) > 5, "the wrist should descend over the close, not jump"
    assert gripper[0] < gripper[-1] == pytest.approx(0.0), "gripper closes over the same span"


def test_homing_still_ends_with_every_joint_at_zero(homing_ramp):
    config, ramp, ticks = homing_ramp

    arrived = ramp.target(config.home_duration_s)
    for name in config.motor_can_ids:
        if name not in (GRIPPER, WRIST):
            assert arrived[name] == pytest.approx(0.0, abs=1e-9)
    assert arrived[GRIPPER] == pytest.approx(config.joint_limits[GRIPPER][0])
    assert ticks[-1][1] == dict.fromkeys(config.motor_can_ids, 0.0)


def test_the_ramp_arrives_on_schedule_however_it_is_sampled(homing_ramp):
    """Sampled by elapsed time, not stepped: a tick that lands late resumes
    where the clock says instead of stretching the ramp out behind it."""
    config, ramp, ticks = homing_ramp
    stalled = ramp.target(ramp.duration * 0.75)
    assert stalled == next(goal for elapsed, goal in ticks if elapsed >= ramp.duration * 0.75)
    assert ramp.target(ramp.duration * 10) == dict.fromkeys(config.motor_can_ids, 0.0)


def homing_bot(monkeypatch, alive=True):
    bot = RebotB601Follower(RebotB601FollowerRobotConfig(port="/dev/null", id="t"))
    monkeypatch.setattr(
        bot, "send_action", lambda action: pytest.fail("the caller must not drive the ramp")
    )
    bot._follower_process = types.SimpleNamespace(is_alive=lambda: alive, pid=1)
    return bot


def test_go_home_hands_the_ramp_to_the_follower_process(monkeypatch):
    """A 100 Hz ramp driven from the caller stalls whenever anything holds that
    process's GIL for a moment -- the dataset writer flushing a video file between
    episodes -- freezing the goal mid-travel and then jumping it."""
    bot = homing_bot(monkeypatch)
    commands = []
    bot._command_queue = types.SimpleNamespace(
        put=lambda cmd: (commands.append(cmd), bot._home_done.set())
    )

    bot.go_home()

    assert commands == ["home"]


def test_go_home_refuses_when_the_follower_process_is_already_gone(monkeypatch):
    """Nothing will ever set the done event, and a caller blocked forever here
    is a caller that never disconnects the arm."""
    bot = homing_bot(monkeypatch, alive=False)
    bot._command_queue = types.SimpleNamespace(
        put=lambda cmd: pytest.fail("nothing is left to read the command")
    )

    with pytest.raises(ConnectionError, match="not running"):
        bot.go_home()


def test_go_home_gives_up_if_the_follower_process_dies_mid_ramp(monkeypatch):
    """The same deadlock one step later: alive at the hand-off, gone before it
    reports back."""
    bot = homing_bot(monkeypatch)
    alive = itertools.chain([True], itertools.repeat(False))
    bot._follower_process = types.SimpleNamespace(is_alive=lambda: next(alive), pid=1)
    bot._command_queue = types.SimpleNamespace(put=lambda cmd: None)

    with pytest.raises(ConnectionError, match="stopped while homing"):
        bot.go_home()

    assert bot._homing_flag.value == 0


def test_an_intermittent_can_failure_does_not_stop_the_follower():
    """The dm-serial write that times out once and then works: the arm must keep its
    goal and keep running, since dying on one costs the whole session."""
    errors = follower_module._BusErrors(grace_s=0.5)
    t = 0.0
    for tick in range(400):
        t += 0.01
        if tick % 10 == 0:
            errors.failed("feedback request to elbow_flex", RuntimeError("timed out"))
        assert not errors.expired(t)


def test_a_bus_that_stops_answering_stops_the_follower_after_the_grace_window():
    errors = follower_module._BusErrors(grace_s=0.5)
    t = 0.0
    while t < 0.45:
        t += 0.01
        errors.failed("send to gripper", RuntimeError("timed out"))
        assert not errors.expired(t)
    t += 0.1
    errors.failed("send to gripper", RuntimeError("timed out"))
    assert errors.expired(t)
    assert "send to gripper" in errors.last_error


MOTORS = list(gm.JOINT_NAMES) + ["gripper"]
REACHING = [0.0, -64.0, -40.0, 24.0, 0.0, 0.0, -30.0]


def compensator(**overrides):
    config = RebotB601FollowerRobotConfig(port="/dev/null", id="t", **overrides)
    return follower_module._GravityCompensator(config, MOTORS), config


def test_compensation_is_on_by_default_and_matches_the_model():
    """The teleoperation case: without this a joint sits g(q)/kp below its
    target to make its own holding torque, which is a standing tracking error."""
    comp, config = compensator()
    assert comp.enabled
    torques = comp.torques(REACHING)

    trim = np.array([config.gravity_scale.get(n, 1.0) for n in gm.JOINT_NAMES])
    expected = gm.gravity_torque(np.radians(REACHING[:6])) * trim
    np.testing.assert_allclose(torques[:6], expected, atol=1e-9)
    assert torques[-1] == 0.0, "the gripper is not in the model"


def test_the_gripper_is_never_compensated():
    comp, _ = compensator()
    for gripper_angle in (-270.0, -120.0, 0.0):
        assert comp.torques(REACHING[:6] + [gripper_angle])[-1] == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"gravity_compensation": False},
        {"control_mode": "pos_vel"},
    ],
)
def test_compensation_yields_zeros_when_it_cannot_apply(overrides):
    comp, _ = compensator(**overrides)
    assert not comp.enabled
    assert not any(comp.torques(REACHING))


def test_an_unrecognised_motor_layout_disables_rather_than_crashes():
    config = RebotB601FollowerRobotConfig(port="/dev/null", id="t", motor_can_ids={"a": (1, 17)})
    comp = follower_module._GravityCompensator(config, ["a"])
    assert not comp.enabled
    assert not any(comp.torques([0.0]))


def test_the_trim_knobs_reach_the_torque():
    base = compensator()[0].torques(REACHING)
    scaled = compensator(gravity_gain=0.5)[0].torques(REACHING)
    np.testing.assert_allclose(scaled[:6], np.array(base[:6]) * 0.5, atol=1e-9)

    elbow = gm.JOINT_NAMES.index("elbow_flex")
    one_joint = compensator(gravity_scale={"elbow_flex": 2.0}, gravity_gain=1.0)[0].torques(REACHING)
    assert one_joint[elbow] == pytest.approx(
        gm.gravity_torque(np.radians(REACHING[:6]))[elbow] * 2.0
    )


def test_a_payload_loads_the_joints_that_carry_it():
    bare = np.array(compensator()[0].torques(REACHING)[:6])
    loaded = np.array(compensator(gravity_payload_kg=1.0)[0].torques(REACHING)[:6])
    # everything upstream of the end effector takes more; the wrist roll axis
    # the payload sits on takes none
    assert np.abs(loaded[1:4]).max() > np.abs(bare[1:4]).max()
    assert loaded[-1] == pytest.approx(bare[-1])


def test_the_ceiling_binds_only_on_the_high_torque_joints():
    comp, _ = compensator(gravity_max_torque=2.0)
    torques = comp.torques(REACHING)
    assert abs(torques[gm.JOINT_NAMES.index("shoulder_lift")]) == pytest.approx(2.0)
    assert abs(torques[gm.JOINT_NAMES.index("wrist_flex")]) < 2.0


def test_an_unknown_trim_joint_is_rejected_at_construction():
    with pytest.raises(ValueError, match="not arm joints"):
        compensator(gravity_scale={"nope": 2.0})


def test_gravity_torques_reports_what_the_follower_is_applying(monkeypatch):
    """The parent-side view has to agree with the follower process, or the
    dry-run readout would be a second opinion rather than the truth."""
    bot = RebotB601Follower(RebotB601FollowerRobotConfig(port="/dev/null", id="t"))
    monkeypatch.setattr(
        bot, "_present_pos", lambda: dict(zip(MOTORS, REACHING, strict=True))
    )
    reported = bot.gravity_torques()
    expected = follower_module._GravityCompensator(bot.config, bot.motor_names).torques(REACHING)
    assert reported == dict(zip(MOTORS, expected, strict=True))


def test_the_smoother_settles_within_the_gripper_hold():
    """The hold is only worth having if the smoothing filter can converge inside
    it. Worst case is a full-travel step, which the real close ramp never is."""
    tau = RebotB601FollowerRobotConfig(port="/dev/null", id="t").smoothing_time_constant_s
    dt = 1.0 / RebotB601FollowerRobotConfig(port="/dev/null", id="t").send_rate_hz
    axis = follower_module._SCurveAxis(tau_s=tau)
    travel = 270.0
    axis.step(-travel, dt)
    for _ in range(200):
        axis.step(-travel, dt)

    pos = travel
    for _ in range(int(follower_module._GRIPPER_SETTLE_SEC / dt)):
        pos, _ = axis.step(0.0, dt)
    assert abs(pos) < 0.5, f"{abs(pos):.2f}deg of travel left when the hold ends"
