"""Shared scaffolding for the b601 Python entry points (record.py,
gravity_compensation.py), the Python counterpart to common.sh.

The session-generic pieces (milestone printing, prompts, graceful Ctrl-C) live in
thesis.utils.session, shared with the lerobot eval backend, and are re-exported here so
the b601 scripts keep one import surface. What stays in this module is b601-specific:
how the arm is walked home on a background thread, kept in one place so the scripts
can't drift on the parts that decide whether the arm is left in a safe state.
"""

from __future__ import annotations

import threading

from lerobot.robots.rebot_b601_follower import RebotB601Follower

from thesis.utils.session import (  # noqa: F401 - re-exported for the b601 scripts
    BLUE,
    CYAN,
    GREEN,
    MAGENTA,
    RESET_COLOR,
    YELLOW,
    announce,
    confirm,
    confirm_phrase,
    flush_pending_stdin,
    install_graceful_sigint_handler,
    prompt,
    quiet_console_logging,
)


class Homing:
    """Handle for a home ramp running on its own thread.

    Call it to wait for the ramp and re-raise anything it hit; check `done` to
    poll instead, for a caller that has to keep servicing the arm meanwhile.
    """

    def __init__(self, thread: threading.Thread, failure: list):
        self._thread = thread
        self._failure = failure

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()

    def __call__(self) -> None:
        self._thread.join()
        if self._failure:
            raise self._failure[0]


def start_homing(robot: RebotB601Follower, play_sounds: bool) -> Homing:
    """Begin ramping the follower back to its calibration zero pose (gripper
    opened on the way, closed at the end) and return a callable that waits for
    it to finish.

    The ramp runs in the follower process; go_home() only waits on it. This
    thread is so the caller can get on with whatever comes next (video
    encoding, in record.py) instead of waiting too -- the ramp itself does not
    care what this process is doing. The caller must not write the goal
    meanwhile.

    The ramp itself is the one disconnect() uses (an eased interpolation at the
    follower's own send rate, so distance doesn't become speed) rather than a
    second implementation that could drift from it."""
    announce("Homing the arm...", BLUE, speak="Homing", play_sounds=play_sounds)
    failure = []

    def ramp() -> None:
        try:
            robot.go_home()
        except Exception as e:
            failure.append(e)

    thread = threading.Thread(target=ramp, name="b601-homing", daemon=True)
    thread.start()
    return Homing(thread, failure)


def format_row(label: str, values) -> str:
    return f"{label:>12s}  " + "  ".join(f"{v:+8.3f}" for v in values)


def format_text_row(label: str, values) -> str:
    """format_row's columns for values that are already text, so the
    temperature table lines up with the torque table above it."""
    return f"{label:>12s}  " + "  ".join(f"{v:>8s}" for v in values)


def format_eta(seconds: float | None) -> str:
    """A time-to-overheat estimate as HH:MM:SS, or "-" where there is none:
    too little history yet, or a motor that is steady or cooling."""
    if seconds is None:
        return "-"
    total = max(0, int(round(seconds)))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def print_temperature_table(robot: RebotB601Follower, elapsed: float) -> None:
    """Both temperatures of every motor and how long the follower process
    reckons each has before it trips temp_max_c.

    The estimates are the follower's own, not recomputed here, so what this
    prints is what the thermal protection is actually acting on.
    """
    temps = robot.motor_temperatures()
    etas = robot.motor_overheat_etas()
    names = robot.motor_names
    print(f"\n[temp] t={format_eta(elapsed)}")
    print(format_text_row("motor", [str(i + 1) for i in range(len(names))]))
    print(format_text_row("mosfet", [f"{temps[n]['mosfet']:.0f}C" for n in names]))
    print(format_text_row("rotor", [f"{temps[n]['rotor']:.0f}C" for n in names]))
    print(format_text_row("overheat", [format_eta(etas[n]) for n in names]))
