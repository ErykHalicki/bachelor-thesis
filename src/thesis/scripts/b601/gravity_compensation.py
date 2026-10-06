#!/usr/bin/env python
"""Hold the reBot B601-DM follower against its own weight so it can be moved by
hand. Invoked by src/thesis/scripts/b601/gravity_compensation.sh, which resolves
the arm port and resets the follower's control mode first.

The follower already carries the arm's own weight on every session -- see
RebotB601FollowerConfig.gravity_compensation, which adds tau = g(q) to every MIT
command. All this script does is take the position gains away: with kp at zero
the arm is left holding itself up on that torque plus kd damping, so it feels
weightless and can be pushed anywhere. Raise kp (or use `hold=true`) to have it
spring back to wherever it was let go.

Everything hardware-side goes through lerobot's RebotB601Follower, the same
driver record.py teleoperates: its follower process keeps commanding the motors
at a fixed rate whatever this loop does, so a stall here holds the arm rather
than dropping it, and homing / Ctrl-C handling / disconnect are the shared
helpers in b601/common.py rather than a second implementation.

Joint angles are raw motor angles, so the model assumes the motor zeros were set
at the URDF zero pose -- the pose lerobot-calibrate homes to. Check with
`dry_run=true` before letting the arm take its own weight: if the printed
torques don't match the direction the arm actually wants to fall, the zeros are
wrong and nothing below will behave.

Controls (same keyboard backend as record.py -- arrow keys, or the letters over
a laggy SSH link, where escape sequences get split):
  Up / Right / l  -> lock: hold the angles the arm is at right now
  Down / Left / f -> unlock: float again
  Esc / q         -> stop and ramp the arm home

Only the gains ever move here; the follower carries the weight throughout,
locked or floating or homing. That is the difference between locking and
sagging: a position loop with no feedforward has to sit g(q)/kp below its target
to generate the holding torque, 8deg at this arm's elbow and 10deg at the wrist.
Feed the weight forward and the error needed is nothing, so a locked joint holds
where it actually is.
"""

import logging
import math
import time
from dataclasses import dataclass, field

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from lerobot.robots.rebot_b601_follower import (
    JOINT_NAMES,
    RebotB601Follower,
    RebotB601FollowerRobotConfig,
)
from lerobot.robots.rebot_b601_follower.gravity_model import NUM_JOINTS
from lerobot.utils.keyboard_input import create_key_listener
from lerobot.utils.utils import log_say
from thesis.scripts.b601.common import (
    CYAN,
    GREEN,
    YELLOW,
    announce,
    format_row,
    install_graceful_sigint_handler,
    print_temperature_table,
    quiet_console_logging,
    start_homing,
)
from omegaconf import MISSING

logger = logging.getLogger(__name__)


@dataclass
class GravityCompConfig:
    """Schema + defaults for gravity_compensation.py, registered with Hydra as
    `gravity_schema`. follower_port is filled in by gravity_compensation.sh."""

    # kp=0 is what makes the arm free to push around: with no position spring, only
    # the gravity feedforward and kd damping act on it
    kp: float = 0.0
    kd: float = 0.8
    hold: bool = False
    hold_kp: float = 2.0
    hold_kd: float = 1.0

    # overrides for the follower's own gravity_* trim, for this run only; unset
    # they defer to RebotB601FollowerConfig
    gain: float | None = None
    scale: dict[str, float] | None = None
    payload_kg: float | None = None
    payload_com: tuple[float, float, float] | None = None
    max_torque: float | None = None

    joints: list[str] = field(default_factory=list)  # subset to float; empty = all
    dry_run: bool = False  # print the torques the model asks for, then stop

    max_vel_deg_s: float = 180.0  # runaway cutoff
    start_locked: bool = False  # begin holding position; unlock with the down/left key
    rate: int = 100  # how often this loop recomputes g(q), Hz
    log_every: int = 50  # print the torque row every N ticks; 0 to silence

    # costs about an eighth of a core in the follower process, since the estimate
    # is refitted there
    debug_temp: bool = False

    quiet: bool = False  # disable spoken announcements
    home_on_start: bool = True
    home_duration_s: float = 1.5

    follower_port: str = MISSING
    follower_id: str = "b601_follower"


ConfigStore.instance().store(name="gravity_schema", node=GravityCompConfig)


def resolve_active(cfg: GravityCompConfig) -> list[str]:
    """Which joints get floated. The rest keep the follower's configured gains,
    so they hold position rather than going limp."""
    if not cfg.joints:
        return list(JOINT_NAMES)
    unknown = set(cfg.joints) - set(JOINT_NAMES)
    if unknown:
        raise ValueError(
            f"joints {sorted(unknown)} are not arm joints; pick from {list(JOINT_NAMES)}"
        )
    return [name for name in JOINT_NAMES if name in cfg.joints]


def read_arm_state(
    robot: RebotB601Follower,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arm joint positions (radians), velocities (deg/s) and measured torques
    (N.m), in JOINT_NAMES order, from the follower process's latest reading.

    The follower reports raw motor angles in degrees; the gravity model wants
    the same angles in radians, which is what makes the URDF zero pose and the
    motor zero pose the same thing."""
    obs = robot.get_observation()
    pos = np.array([math.radians(obs[f"{name}.pos"]) for name in JOINT_NAMES])
    vel = np.array([obs[f"{name}.vel"] for name in JOINT_NAMES])
    torq = np.array([obs[f"{name}.torq"] for name in JOINT_NAMES])
    return pos, vel, torq


def make_key_listener(events: dict):
    """Start the shared keyboard listener wired to this script's controls.

    Arrow keys and letter equivalents both, for the same reason record.py takes
    n/r/q: over a laggy SSH or VNC link an arrow's escape sequence can arrive
    split or be swallowed by the terminal, while a single byte cannot."""

    def on_key(name: str) -> None:
        key = name.lower()
        if key in ("up", "right", "l"):
            events["locked"] = True
        elif key in ("down", "left", "f"):
            events["locked"] = False
        elif key in ("esc", "q"):
            events["stop_recording"] = True

    return create_key_listener(
        on_key, controls_help="Up/Right=lock, Down/Left=float, Esc=stop; or l / f / q"
    )


def resting_on_stops(
    joint_limits: dict[str, tuple[float, float]],
    pos: np.ndarray,
    tau: np.ndarray,
    slack_deg: float = 2.0,
) -> list[str]:
    """Joints sitting against the soft limit that gravity is pushing them into.

    The mechanical stop is carrying those joints, not the motor, so their
    measured torque reads near zero however much the pose actually needs. The
    calibration zero pose is one of these -- shoulder_lift and elbow_flex both
    have zero at the top of their range, and gravity pushes both that way -- so
    it is the one pose where measured-vs-model proves nothing.

    tau is what the motor must exert, so gravity pushes the joint the other way:
    tau < 0 means gravity drives q up, toward the upper limit.
    """
    resting = []
    for i, name in enumerate(JOINT_NAMES):
        if name not in joint_limits or abs(tau[i]) < 1e-3:
            continue
        lower, upper = joint_limits[name]
        degrees = math.degrees(pos[i])
        if tau[i] < 0 and degrees > upper - slack_deg:
            resting.append(name)
        elif tau[i] > 0 and degrees < lower + slack_deg:
            resting.append(name)
    return resting


def apply_gains(robot: RebotB601Follower, kp: np.ndarray, kd: np.ndarray) -> None:
    """Set the arm's position gains outright.

    Nothing is faded, because there is nothing to fade: the follower carries the
    arm's weight at every gain setting, so removing the position spring leaves a
    balanced arm and restoring it engages against a goal already at the current
    pose. A joint only lurches when the two hand over unevenly, and they never do
    here. It is also why a locked joint holds its angle *at* its angle -- an
    unassisted position loop would sit g(q)/kp below target, 8deg at this elbow.
    """
    robot.set_mit_gains(
        kp={name: float(kp[i]) for i, name in enumerate(JOINT_NAMES)},
        kd={name: float(kd[i]) for i, name in enumerate(JOINT_NAMES)},
    )


def print_torque_table(pos: np.ndarray, tau: np.ndarray, measured: np.ndarray) -> None:
    """What the follower is applying next to what the motors report, so a bad
    zero or a mis-set payload is visible before the arm carries itself."""
    print(format_row("joint", np.arange(1, NUM_JOINTS + 1)))
    print(format_row("q (deg)", np.degrees(pos)))
    print(format_row("tau (N.m)", tau))
    print(format_row("measured", measured))


@hydra.main(
    version_base=None, config_path="configs", config_name="gravity_compensation"
)
def main(cfg: GravityCompConfig) -> None:
    quiet_console_logging()
    play_sounds = not cfg.quiet

    active = resolve_active(cfg)
    active_mask = np.array([name in active for name in JOINT_NAMES])
    kp = cfg.hold_kp if cfg.hold else cfg.kp
    kd = cfg.hold_kd if cfg.hold else cfg.kd
    gravity_overrides = {
        key: value
        for key, value in (
            ("gravity_gain", cfg.gain),
            ("gravity_scale", dict(cfg.scale) if cfg.scale is not None else None),
            ("gravity_payload_kg", cfg.payload_kg),
            ("gravity_payload_com", tuple(cfg.payload_com) if cfg.payload_com else None),
            ("gravity_max_torque", cfg.max_torque),
        )
        if value is not None
    }

    robot = RebotB601Follower(
        RebotB601FollowerRobotConfig(
            port=cfg.follower_port,
            id=cfg.follower_id,
            can_adapter="damiao",
            control_mode="mit",
            home_duration_s=cfg.home_duration_s,
            # the goal position is meaningless once kp is 0, and smoothing it would make
            # the commanded velocity chase the arm instead of sitting at zero
            enable_trajectory_smoothing=False,
            # a dry run reports what the model asks for and must not move the arm
            return_home_on_disconnect=not cfg.dry_run,
            gravity_compensation=True,
            temp_debug=cfg.debug_temp,
            **gravity_overrides,
        )
    )

    listener = None
    softened = False
    try:
        robot.connect()
        events = {
            "locked": cfg.start_locked,
            "exit_early": False,
            "stop_recording": False,
        }
        listener = make_key_listener(events)
        install_graceful_sigint_handler(events)
        pos, _, measured = read_arm_state(robot)
        # straight from the follower, so this is what it is actually applying
        applied = robot.gravity_torques()
        tau = np.array([applied[name] for name in JOINT_NAMES])
        print()
        print_torque_table(pos, tau, measured)

        if cfg.dry_run:
            # nothing has called send_action yet, so the follower has no goal to dispatch
            announce(
                "\ndry_run: the arm was never floated, and nothing is commanding the motors "
                "yet, so `measured` reads near zero whatever the pose needs.",
                YELLOW,
            )
            resting = resting_on_stops(robot.config.joint_limits, pos, tau)
            if resting:
                announce(
                    f"{', '.join(resting)} are against the soft limit gravity pushes them into, "
                    "so the stops are carrying them here -- this pose cannot confirm the model. "
                    "Hold the arm somewhere mid-range and re-run to see torques it must earn.",
                    YELLOW,
                )
            return

        configured_kp, configured_kd = robot.get_mit_gains()
        held_kp = np.array([configured_kp[name] for name in JOINT_NAMES])
        held_kd = np.array([configured_kd[name] for name in JOINT_NAMES])
        float_kp = np.array(
            [kp if name in active else held_kp[i] for i, name in enumerate(JOINT_NAMES)]
        )
        float_kd = np.array(
            [kd if name in active else held_kd[i] for i, name in enumerate(JOINT_NAMES)]
        )
        softened = False

        # the follower keeps carrying the arm through the ramp, so homing does not
        # have to be waited out before the gains can move
        homing = start_homing(robot, play_sounds) if cfg.home_on_start else None

        announce(
            f"{len(active)} joint(s) under gravity compensation (float: kp={kp}, kd={kd}).",
            CYAN,
            speak="Gravity compensation on",
            play_sounds=play_sounds,
            bell=True,
        )
        announce(
            "  [up / right / l] lock   [down / left / f] float   [q / esc] stop", CYAN
        )
        if cfg.debug_temp:
            announce(
                "  temp columns: "
                + "  ".join(f"{i + 1}={n}" for i, n in enumerate(robot.motor_names)),
                CYAN,
            )

        interval = 1.0 / cfg.rate
        next_tick = time.perf_counter()
        started = time.perf_counter()
        next_temp_print = started
        ticks = 0
        pos, _, _ = read_arm_state(robot)
        lock_pose = pos.copy()
        was_locked = None

        while not events["stop_recording"]:
            pos, vel, measured = read_arm_state(robot)
            if np.abs(vel[active_mask]).max(initial=0.0) > cfg.max_vel_deg_s:
                announce(
                    f"\nJoint velocity exceeded {cfg.max_vel_deg_s:.0f} deg/s; stopping.",
                    YELLOW,
                )
                break

            if homing is not None and not homing.done:
                # the home ramp owns the goal and needs the position gains it started with
                lock_pose = pos.copy()
            else:
                if homing is not None:
                    homing()
                    homing = None

                locked = events["locked"]
                if locked and locked != was_locked:
                    lock_pose = pos.copy()

                # a locked joint still has its spring, so the goal must be the pose it holds;
                # floating it is inert (kp=0) but keeps tracking, so homing cannot snap to a stale goal
                goal = lock_pose if locked else pos
                robot.send_action(
                    {
                        f"{n}.pos": math.degrees(goal[i])
                        for i, n in enumerate(JOINT_NAMES)
                    }
                )

                if locked != was_locked:
                    # after the goal, never before: a spring that engages first pulls toward
                    # whatever the last goal happened to be
                    apply_gains(
                        robot,
                        held_kp if locked else float_kp,
                        held_kd if locked else float_kd,
                    )
                    softened = not locked
                    was_locked = locked
                    announce(
                        f"\n{'Locked' if locked else 'Floating'} at "
                        + ", ".join(f"{v:.1f}" for v in np.degrees(goal))
                        + " deg",
                        YELLOW if locked else GREEN,
                    )

                ticks += 1
                if cfg.log_every and ticks % cfg.log_every == 0:
                    # a joint that consistently reads heavier than it was told to pull needs
                    # its `scale` raised
                    applied = robot.gravity_torques()
                    tau = np.array([applied[name] for name in JOINT_NAMES])
                    state = "LOCK" if locked else "float"
                    print(
                        f"{format_row(f'{state} tau', tau)}\n"
                        f"{format_row('measured', measured)}",
                        # rewritten in place, except under debug_temp, where a cursor jumping back
                        # would clip the temperature block scrolling past
                        end="\n" if cfg.debug_temp else "\033[1A\r",
                        flush=True,
                    )

            if cfg.debug_temp and time.perf_counter() >= next_temp_print:
                next_temp_print += 1.0
                print_temperature_table(robot, time.perf_counter() - started)

            # clear it so it cannot be mistaken for a stop later
            events["exit_early"] = False

            next_tick += interval
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.perf_counter()
    finally:
        print()
        announce("Stopping.", YELLOW)
        log_say("Stopping gravity compensation", play_sounds, blocking=True)
        if softened and robot.is_connected:
            # a kp=0 joint ignores its goal, so disconnect()'s home ramp would do nothing
            # to it: restore the configured gains before handing the arm over
            apply_gains(robot, held_kp, held_kd)
        if robot.is_connected:
            robot.disconnect()
        if listener is not None:
            listener.stop()
        print("Done.")


if __name__ == "__main__":
    main()
