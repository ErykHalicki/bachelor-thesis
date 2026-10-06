"""The gravity compensation session, driven against a stand-in follower.

The follower carries the arm's weight itself now, on every session including
teleoperation, so this script only decides how stiff the joints are. What is
left to get wrong is ordering: the position gains have to be back before
anything ramps the arm home, because a kp=0 joint ignores its goal entirely.
The fake follower asserts that from the inside.

Whether the torque itself is right belongs to the follower -- see
test_b601_follower.py for the compensator, and test_gravity_model.py for the
model under it.
"""

import math
import threading

import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("lerobot")
pytest.importorskip("motorbridge")
gc = pytest.importorskip("thesis.scripts.b601.gravity_compensation")
gm = pytest.importorskip("lerobot.robots.rebot_b601_follower.gravity_model")

MOTOR_NAMES = list(gm.JOINT_NAMES) + ["gripper"]
CONFIGURED_KP = 45.0
CONFIGURED_KD = 12.0
# reaching out, so the shoulder and elbow torques are large enough that a sign
# or scaling error cannot hide in the noise
POSE_DEG = {"shoulder_lift": -40.0, "elbow_flex": -70.0}


class FakeFollower:
    """Stands in for RebotB601Follower, recording what the session asked of it
    and refusing the moves that would drop a real arm."""

    def __init__(self, config):
        self.config = config
        self.motor_names = MOTOR_NAMES
        self.is_connected = False
        self.pos_deg = {name: POSE_DEG.get(name, 0.0) for name in MOTOR_NAMES}
        self.kp = dict.fromkeys(MOTOR_NAMES, CONFIGURED_KP)
        self.gain_calls: list[tuple] = []
        self.homed = 0
        self.disconnects = 0
        self.actions = 0

    def connect(self):
        self.is_connected = True

    def get_observation(self):
        obs = {}
        for name in MOTOR_NAMES:
            obs[f"{name}.pos"] = self.pos_deg[name]
            obs[f"{name}.vel"] = 0.0
            obs[f"{name}.torq"] = 0.0
        return obs

    def gravity_torques(self):
        q = np.radians([self.pos_deg[name] for name in gm.JOINT_NAMES])
        tau = dict(zip(gm.JOINT_NAMES, gm.gravity_torque(q), strict=True))
        return {name: tau.get(name, 0.0) for name in MOTOR_NAMES}

    def send_action(self, action):
        self.actions += 1
        return action

    def set_mit_gains(self, kp=None, kd=None):
        self.gain_calls.append(
            (
                dict(kp) if isinstance(kp, dict) else kp,
                dict(kd) if isinstance(kd, dict) else kd,
            )
        )
        if kp is None:
            self.kp = dict.fromkeys(MOTOR_NAMES, CONFIGURED_KP)
        elif isinstance(kp, dict):
            self.kp.update(kp)

    def get_mit_gains(self):
        return dict(self.kp), dict.fromkeys(MOTOR_NAMES, CONFIGURED_KD)

    def go_home(self):
        assert all(self.kp[n] > 0 for n in gm.JOINT_NAMES), (
            "homing a kp=0 joint does nothing"
        )
        self.homed += 1

    def disconnect(self):
        assert all(self.kp[n] > 0 for n in gm.JOINT_NAMES), (
            "disconnect homes; stiffen back first"
        )
        self.disconnects += 1
        self.is_connected = False


@pytest.fixture
def session(monkeypatch):
    """Run gc.main()'s body against a FakeFollower, returning it for inspection.

    Calls the undecorated function so the config is built here instead of
    through Hydra's CLI; the config schema is the real one either way.
    """
    built: list[FakeFollower] = []

    def make(config):
        follower = FakeFollower(config)
        built.append(follower)
        return follower

    monkeypatch.setattr(gc, "RebotB601Follower", make)

    def run(
        stop_after: float = 0.4,
        lock_at: float | None = None,
        unlock_at: float | None = None,
        **overrides,
    ):
        """`lock_at`/`unlock_at` press those keys that many seconds in, as a user would."""
        timers = []

        def fake_listener(events):
            if lock_at is not None:
                timers.append(threading.Timer(lock_at, lambda: events.update(locked=True)))
            if unlock_at is not None:
                timers.append(threading.Timer(unlock_at, lambda: events.update(locked=False)))
            if stop_after:
                timers.append(
                    threading.Timer(stop_after, lambda: events.update(stop_recording=True))
                )
            for timer in timers:
                timer.start()
            return None

        monkeypatch.setattr(gc, "make_key_listener", fake_listener)
        cfg = OmegaConf.structured(gc.GravityCompConfig)
        cfg.follower_port = "/dev/fake"
        cfg.log_every = 0
        for key, value in overrides.items():
            setattr(cfg, key, value)
        try:
            gc.main.__wrapped__(cfg)
        finally:
            for timer in timers:
                timer.cancel()
        return built[-1]

    return run


def gain_dicts(follower: FakeFollower) -> list[tuple[dict, dict]]:
    return [call for call in follower.gain_calls if isinstance(call[0], dict)]


def floated_gains(follower: FakeFollower) -> tuple[dict, dict]:
    """The softest gains reached, i.e. the floating end of the crossfade."""
    return min(gain_dicts(follower), key=lambda call: sum(call[0].values()))


def final_gains(follower: FakeFollower) -> tuple[dict, dict]:
    """The gains the arm was left on, after stiffening back on the way out."""
    return gain_dicts(follower)[-1]


def test_the_follower_is_told_to_compensate(session):
    follower = session(stop_after=0, dry_run=True)
    assert follower.config.gravity_compensation is True
    assert follower.config.control_mode == "mit", "feedforward exists only in MIT mode"


def test_trim_is_left_to_the_follower_unless_overridden(session):
    """Unset knobs must not shadow the follower's own defaults, or the arm would
    behave differently here than it does under record.py."""
    from lerobot.robots.rebot_b601_follower import RebotB601FollowerRobotConfig

    stock = RebotB601FollowerRobotConfig(port="/dev/null", id="x")
    follower = session(stop_after=0, dry_run=True)
    for field in ("gravity_gain", "gravity_scale", "gravity_payload_kg", "gravity_max_torque"):
        assert getattr(follower.config, field) == getattr(stock, field)


@pytest.mark.parametrize(
    "override, field, expected",
    [
        ({"gain": 0.9}, "gravity_gain", 0.9),
        ({"scale": {"elbow_flex": 1.2}}, "gravity_scale", {"elbow_flex": 1.2}),
        ({"payload_kg": 1.5}, "gravity_payload_kg", 1.5),
        ({"max_torque": 12.0}, "gravity_max_torque", 12.0),
    ],
)
def test_overrides_reach_the_follower_config(session, override, field, expected):
    follower = session(stop_after=0, dry_run=True, **override)
    assert getattr(follower.config, field) == expected


def test_dry_run_never_touches_the_arm(session):
    follower = session(stop_after=0, dry_run=True)
    assert follower.homed == 0
    assert follower.gain_calls == []
    assert follower.config.return_home_on_disconnect is False
    assert follower.disconnects == 1


def test_arm_goes_soft_but_the_gripper_keeps_its_grip(session):
    follower = session()
    kp, _ = floated_gains(follower)
    assert kp == pytest.approx(dict.fromkeys(gm.JOINT_NAMES, 0.0))
    assert "gripper" not in kp, "the gripper is never softened, so it keeps holding"


def test_gains_are_restored_on_the_way_out(session):
    follower = session()
    assert final_gains(follower)[0] == pytest.approx(
        dict.fromkeys(gm.JOINT_NAMES, CONFIGURED_KP)
    )
    assert follower.disconnects == 1


def test_it_homes_before_softening_the_arm(session):
    follower = session()
    assert follower.homed == 1


def test_home_on_start_can_be_skipped(session):
    follower = session(home_on_start=False)
    assert follower.homed == 0


def test_only_the_listed_joints_are_floated(session):
    follower = session(joints=["shoulder_lift", "elbow_flex"])
    kp, _ = floated_gains(follower)
    assert kp["shoulder_lift"] == pytest.approx(0.0)
    assert kp["elbow_flex"] == pytest.approx(0.0)
    assert kp["shoulder_pan"] == pytest.approx(CONFIGURED_KP)
    assert kp["wrist_flex"] == pytest.approx(CONFIGURED_KP)


def test_hold_swaps_in_a_position_spring(session):
    follower = session(hold=True)
    kp, kd = floated_gains(follower)
    assert kp == pytest.approx(dict.fromkeys(gm.JOINT_NAMES, gc.GravityCompConfig.hold_kp))
    assert kd == pytest.approx(dict.fromkeys(gm.JOINT_NAMES, gc.GravityCompConfig.hold_kd))


def test_goal_keeps_tracking_the_arm_so_homing_has_no_gap(session):
    # with kp=0 the goal does nothing, but a stale goal is a lurch when disconnect
    # ramps home from it
    follower = session()
    assert follower.actions > 1


def test_locking_restores_the_gains(session):
    follower = session(stop_after=0.6, lock_at=0.3)
    assert final_gains(follower)[0] == pytest.approx(
        dict.fromkeys(gm.JOINT_NAMES, CONFIGURED_KP)
    )


def test_gains_switch_outright_on_each_transition(session):
    """Nothing is faded: the follower carries the weight at every gain setting,
    so there is no partway state to pass through -- and one write per transition
    rather than one per tick keeps the loop off the shared arrays."""
    follower = session(stop_after=0.6, lock_at=0.3, rate=100)
    kps = [call[0]["elbow_flex"] for call in gain_dicts(follower)]
    assert set(kps) == {0.0, CONFIGURED_KP}, f"intermediate stiffness reached: {sorted(set(kps))}"
    # no third write on the way out: locking already put the configured gains back
    assert len(kps) == 2, f"expected one write per transition, got {len(kps)}"


def test_unlocking_returns_to_floating(session):
    follower = session(stop_after=0.9, lock_at=0.3, unlock_at=0.6)
    assert floated_gains(follower)[0] == pytest.approx(dict.fromkeys(gm.JOINT_NAMES, 0.0))


def test_start_locked_never_softens(session):
    follower = session(stop_after=0.4, start_locked=True)
    assert floated_gains(follower)[0] == pytest.approx(
        dict.fromkeys(gm.JOINT_NAMES, CONFIGURED_KP)
    )


def test_unknown_joint_names_are_rejected(session):
    with pytest.raises(ValueError, match="not arm joints"):
        session(stop_after=0, joints=["nope"])


def test_runaway_velocity_stops_the_session(session, monkeypatch):
    real = gc.read_arm_state

    def fast(robot):
        pos, vel, torq = real(robot)
        vel[gm.JOINT_NAMES.index("elbow_flex")] = 1e4
        return pos, vel, torq

    monkeypatch.setattr(gc, "read_arm_state", fast)
    follower = session(stop_after=0, rate=50)
    assert follower.actions <= 2
    assert all(value == CONFIGURED_KP for value in follower.kp.values())


SOFT_LIMITS = {
    "shoulder_pan": (-150.0, 150.0),
    "shoulder_lift": (-200.0, 1.0),
    "elbow_flex": (-200.0, 1.0),
    "wrist_flex": (-80.0, 90.0),
    "wrist_yaw": (-90.0, 90.0),
    "wrist_roll": (-90.0, 90.0),
}


def test_the_zero_pose_is_flagged_as_resting_on_its_stops():
    # shoulder_lift and elbow_flex both have zero at the top of their range and
    # gravity pushes both that way, so the stops carry them
    q = np.radians([1.169, -0.055, -0.339, -0.142, -0.863, -2.131])
    resting = gc.resting_on_stops(SOFT_LIMITS, q, gm.gravity_torque(q))
    assert resting == ["shoulder_lift", "elbow_flex"]


def test_a_mid_range_pose_is_not_flagged():
    q = np.radians([0.0, -45.0, -60.0, 0.0, 0.0, 0.0])
    assert gc.resting_on_stops(SOFT_LIMITS, q, gm.gravity_torque(q)) == []


def test_unloaded_joints_are_never_called_stopped():
    q = np.radians([0.0, -45.0, -60.0, 0.0, 0.0, 90.0])
    assert "wrist_roll" not in gc.resting_on_stops(SOFT_LIMITS, q, gm.gravity_torque(q))


def test_a_joint_held_off_its_stop_by_gravity_is_not_flagged():
    q = np.radians([0.0, -45.0, -60.0, 0.0, 0.0, 0.0])
    away = -gm.gravity_torque(q)
    assert "elbow_flex" not in gc.resting_on_stops(SOFT_LIMITS, q, away)


def test_read_arm_state_converts_degrees_to_radians():
    follower = FakeFollower(config=None)
    pos, vel, torq = gc.read_arm_state(follower)
    expected = [math.radians(POSE_DEG.get(name, 0.0)) for name in gm.JOINT_NAMES]
    np.testing.assert_allclose(pos, expected)
    assert pos.shape == vel.shape == torq.shape == (gm.NUM_JOINTS,)
