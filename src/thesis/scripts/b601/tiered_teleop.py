#!/usr/bin/env python
"""Tiered wrist_flex jitter probe, outside lerobot.

Drives the B601 follower directly through motorbridge and reads the reBot 102
leader directly through motorbridge_smart_servo, with no lerobot pipeline in
between. Each tier adds one layer of complexity so we can find exactly which
layer makes the follower wrist_flex jitter.

Tiers:
  1  code       follower wrist_flex tracks a code-generated triangle wave.
                No leader is opened. Reproduces "direct motor control".
  2  one2one    open leader, read ONLY servo 3 each tick, command ONLY
                follower wrist_flex. Minimal one-servo teleop.
  3  read7cmd1  read ALL 7 leader servos each tick (full sync_monitor cost),
                still command ONLY follower wrist_flex.
  4  read7cmd7  read all 7 leader servos, command all 7 follower motors.
                Full teleop command traffic, no follower feedback read.
  5  full       tier 4 plus a follower feedback read every tick, matching the
                lerobot get_observation -> send_action loop.

Run one tier, move the leader's wrist_flex by hand, and watch the follower.
Every run writes a CSV trace next to this script.

Usage:
  python scripts/b601/tiered_teleop.py --tier 2 --seconds 30
  python scripts/b601/tiered_teleop.py --tier 1 --amp 40 --speed 20
  python scripts/b601/tiered_teleop.py --tier 5 --readback
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

from motorbridge import Controller, MODE_MIT, RID_CTRL_MODE
from motorbridge_smart_servo import FashionStarServo


def _env_port(var_name: str) -> str:
    """Read a port from src/thesis/scripts/b601/.env (same file record.sh /
    teleop.sh source), so this script tracks whatever port record.sh's last
    auto-detection found instead of a hardcoded, machine-specific path."""
    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(f'export {var_name}="') and line.endswith('"'):
                return line[len(f'export {var_name}="'):-1]
    value = os.environ.get(var_name)
    if not value:
        raise SystemExit(
            f"{var_name} not found in {env_path} or the environment. "
            "Run record.sh or teleop.sh once to auto-detect and persist ports."
        )
    return value


FOLLOWER_PORT = _env_port("THESIS_B601_FOLLOWER_PORT")
FOLLOWER_BAUD = 921600
LEADER_PORT = _env_port("THESIS_B601_LEADER_PORT")
LEADER_BAUD = 1_000_000

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
]
CAN_IDS = {
    "shoulder_pan": (0x01, 0x11),
    "shoulder_lift": (0x02, 0x12),
    "elbow_flex": (0x03, 0x13),
    "wrist_flex": (0x04, 0x14),
    "wrist_yaw": (0x05, 0x15),
    "wrist_roll": (0x06, 0x16),
    "gripper": (0x07, 0x17),
}
MODELS = {
    "shoulder_pan": "4340P",
    "shoulder_lift": "4340P",
    "elbow_flex": "4340P",
    "wrist_flex": "4310",
    "wrist_yaw": "4310",
    "wrist_roll": "4310",
    "gripper": "4310",
}
MIT_KP = {
    "shoulder_pan": 45.0,
    "shoulder_lift": 45.0,
    "elbow_flex": 45.0,
    "wrist_flex": 8.0,
    "wrist_yaw": 9.0,
    "wrist_roll": 8.0,
    "gripper": 8.0,
}
MIT_KD = {
    "shoulder_pan": 12.0,
    "shoulder_lift": 12.0,
    "elbow_flex": 12.0,
    "wrist_flex": 1.0,
    "wrist_yaw": 1.0,
    "wrist_roll": 1.0,
    "gripper": 1.0,
}
LEADER_IDS = {
    "shoulder_pan": 0,
    "shoulder_lift": 1,
    "elbow_flex": 2,
    "wrist_flex": 3,
    "wrist_yaw": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
LEADER_DIR = {
    "shoulder_pan": -1,
    "shoulder_lift": -1,
    "elbow_flex": 1,
    "wrist_flex": 1,
    "wrist_yaw": 1,
    "wrist_roll": -1,
    "gripper": -6,
}
JOINT_RANGE = {
    "shoulder_pan": (-150.0, 150.0),
    "shoulder_lift": (-200.0, 1.0),
    "elbow_flex": (-200.0, 1.0),
    "wrist_flex": (-80.0, 90.0),
    "wrist_yaw": (-90.0, 90.0),
    "wrist_roll": (-90.0, 90.0),
    "gripper": (-270.0, 0.0),
}


# sync_monitor() stalls ~100ms intermittently past 4 servo ids in one call (a
# batch-size limit in the sync command). Mirrors RebotArm102Leader's chunking.
_MAX_SYNC_MONITOR_IDS = 4


def sync_monitor_chunked(leader: FashionStarServo, ids: list[int]) -> dict:
    result: dict = {}
    for i in range(0, len(ids), _MAX_SYNC_MONITOR_IDS):
        result.update(leader.sync_monitor(ids[i : i + _MAX_SYNC_MONITOR_IDS]))
    return result


def leader_to_target(name: str, raw_deg: float) -> float:
    """Replicate the lerobot leader get_action math for one joint."""
    range_min, range_max = JOINT_RANGE[name]
    direction = LEADER_DIR[name]
    sign = 1.0 if direction >= 0 else -1.0
    lo, hi = range_min * sign, range_max * sign
    center = (lo + hi) / 2.0
    turns = round((raw_deg - center) / 360.0)
    unwrapped = raw_deg - turns * 360.0
    position = unwrapped * direction
    return max(range_min, min(range_max, position))


def ensure_mit_mode(motor, name: str) -> None:
    """Force the motor into MIT mode and persist it. Does NOT enable -- see the
    note in main() about why enabling is done in a separate, tight loop across
    all configured motors instead of one-by-one here.

    ensure_mode's runtime CTRL_MODE write does not stick when the motor has a
    different mode saved, so write the register directly and store_parameters.
    """
    motor.disable()
    time.sleep(0.5)
    mode = None
    for _ in range(5):
        try:
            mode = motor.damiao_get_param_u32(RID_CTRL_MODE)
        except Exception:
            mode = None
        if mode == MODE_MIT:
            break
        motor.disable()
        time.sleep(0.3)
        motor.damiao_write_param_u32(RID_CTRL_MODE, MODE_MIT)
        motor.store_parameters()
        time.sleep(0.4)
    if mode != MODE_MIT:
        raise RuntimeError(f"{name}: could not set MIT mode (stuck at {mode})")
    print(f"  {name}: MIT mode set")


def read_wf_state(controller, motor):
    """Fresh single-motor feedback read for wrist_flex, in degrees. Also returns
    the raw status_code so a live fault (e.g. under-voltage, comm-loss) shows up
    in the trace instead of just silently-unmoving position/velocity."""
    motor.request_feedback()
    try:
        controller.poll_feedback_once()
    except Exception:
        pass
    st = motor.get_state()
    if st is None:
        return math.nan, math.nan, None
    return math.degrees(st.pos), math.degrees(st.vel), st.status_code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, required=True, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--amp", type=float, default=30.0, help="tier 1 triangle amplitude (deg)")
    ap.add_argument("--speed", type=float, default=20.0, help="tier 1 triangle speed (deg/s)")
    ap.add_argument("--readback", action="store_true",
                    help="also read+log fresh wrist_flex feedback each tick")
    args = ap.parse_args()

    uses_leader = args.tier >= 2
    cmd_all = args.tier >= 4
    feedback_all = args.tier >= 5
    cmd_names = MOTOR_NAMES if cmd_all else ["wrist_flex"]
    read_ids = (
        list(LEADER_IDS.values()) if args.tier >= 3
        else [LEADER_IDS["wrist_flex"]] if uses_leader
        else []
    )

    print(f"Tier {args.tier}: command {cmd_names}, "
          f"leader reads {len(read_ids)} servo(s), feedback_all={feedback_all}")

    controller = Controller.from_dm_serial(serial_port=FOLLOWER_PORT, baud=FOLLOWER_BAUD)
    motors = {}
    for name in cmd_names:
        send_id, recv_id = CAN_IDS[name]
        motors[name] = controller.add_damiao_motor(send_id, recv_id, MODELS[name])

    controller.disable_all()
    time.sleep(0.5)

    # leader connect/unlock/reset runs BEFORE arming the follower: it is a
    # multi-round-trip sequence that sends the follower no traffic, and after
    # enable() it could outlast the follower's RID_TIMEOUT and trip a comm-loss fault
    leader = None
    if uses_leader:
        leader = FashionStarServo(LEADER_PORT, baudrate=LEADER_BAUD)
        for name in MOTOR_NAMES:
            leader.unlock(LEADER_IDS[name])
            time.sleep(0.01)
        for name in MOTOR_NAMES:
            leader.reset_multi_turn(LEADER_IDS[name])
        print("Leader connected, unlocked, multi-turn reset.")

    print("Arming follower motors (each takes ~2s to accept MIT mode)...")
    for name in cmd_names:
        ensure_mit_mode(motors[name], name)

    # enabling happens in its own tight loop right before the control loop starts:
    # once enabled a motor expects continuous commands, and with ~1-2s each to
    # mode-set, an early-enabled motor would sit silent past its comm-timeout
    for name in cmd_names:
        motors[name].enable()
    print("Follower motors enabled.")

    out_path = Path(__file__).with_name(
        f"tier{args.tier}_{time.strftime('%H%M%S')}.csv"
    )
    cols = ["t_s", "dt_ms", "leader_raw_wf", "cmd_wf"]
    if args.readback:
        cols += ["fb_pos_wf", "fb_vel_wf", "fb_status"]
    f = open(out_path, "w", newline="", buffering=1)
    writer = csv.writer(f)
    writer.writerow(cols)

    dt_target = 1.0 / args.hz
    wf_motor = motors["wrist_flex"]

    print(f"\nRunning {args.seconds:.0f}s at {args.hz:.0f} Hz. Move the leader wrist_flex. Ctrl-C to stop.")
    print(f"Logging to {out_path}\n")

    t0 = time.perf_counter()
    prev = t0
    try:
        while True:
            now = time.perf_counter()
            t = now - t0
            if t >= args.seconds:
                break
            dt_ms = (now - prev) * 1e3
            prev = now

            leader_raw_wf = math.nan
            cmd_wf = math.nan

            if feedback_all:
                for name in cmd_names:
                    motors[name].request_feedback()
                try:
                    controller.poll_feedback_once()
                except Exception:
                    pass

            if uses_leader:
                assert leader is not None
                mons = sync_monitor_chunked(leader, read_ids)
                raws = {}
                for name in MOTOR_NAMES:
                    lid = LEADER_IDS[name]
                    if lid in mons and mons[lid] is not None:
                        raws[name] = mons[lid].angle_deg
                leader_raw_wf = raws.get("wrist_flex", math.nan)
                for name in cmd_names:
                    if name not in raws:
                        continue
                    target = leader_to_target(name, raws[name])
                    if name == "wrist_flex":
                        cmd_wf = target
                    motors[name].send_mit(
                        math.radians(target), 0.0, MIT_KP[name], MIT_KD[name], 0.0
                    )
            else:
                phase = (args.speed * t) % (2.0 * args.amp)
                tri = phase if phase <= args.amp else (2.0 * args.amp - phase)
                target = tri - args.amp / 2.0
                cmd_wf = target
                wf_motor.send_mit(
                    math.radians(target), 0.0, MIT_KP["wrist_flex"], MIT_KD["wrist_flex"], 0.0
                )

            row = [f"{t:.4f}", f"{dt_ms:.2f}", f"{leader_raw_wf:.3f}", f"{cmd_wf:.3f}"]
            if args.readback:
                pos, vel, status = read_wf_state(controller, wf_motor)
                row += [f"{pos:.3f}", f"{vel:.3f}", "" if status is None else status]
            writer.writerow(row)

            sleep = dt_target - (time.perf_counter() - now)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        f.close()
        try:
            controller.disable_all()
        except Exception:
            pass
        for motor in motors.values():
            try:
                motor.disable()
                motor.clear_error()
                motor.close()
            except Exception:
                pass
        try:
            controller.close()
        except Exception:
            pass
        if leader is not None:
            try:
                leader.close()
            except Exception:
                pass
        print(f"Done. Trace at {out_path}")


if __name__ == "__main__":
    main()
