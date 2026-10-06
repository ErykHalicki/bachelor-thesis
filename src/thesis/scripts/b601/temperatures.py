#!/usr/bin/env python
"""Read the reBot B601-DM's motor temperatures -- the same numbers its thermal
protection acts on. Invoked by src/thesis/scripts/b601/temperatures.sh, which
resolves the arm port first.

Every Damiao feedback frame carries the MOSFET and rotor temperature alongside
position, so this connects RebotB601Follower for its feedback stream and reads
nothing else. The ThermalMonitor inside the follower process runs here exactly
as it does under teleop or a rollout: its 57C/61C warnings print on this
terminal, its 65C shutdown disconnects the arm, and the `overheat` row is that
same monitor's estimate rather than a second one fitted here.

Nothing is ever commanded and torque is dropped right after connecting, so the
arm hangs on its own stops throughout -- exactly as limp as it was before the
script started, and no warmer for having been checked. Rest it somewhere it can
sit unpowered before running this.
"""

import logging
import time
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from lerobot.robots.rebot_b601_follower import (
    RebotB601Follower,
    RebotB601FollowerRobotConfig,
)
from thesis.scripts.b601.common import (
    CYAN,
    YELLOW,
    announce,
    install_graceful_sigint_handler,
    print_temperature_table,
    quiet_console_logging,
)
from omegaconf import MISSING

logger = logging.getLogger(__name__)


@dataclass
class TemperaturesConfig:
    """Schema + defaults for temperatures.py, registered with Hydra as
    `temperatures_schema`. follower_port is filled in by temperatures.sh."""

    watch: bool = False  # keep printing until Ctrl-C instead of reading once
    interval: float = 2.0  # seconds between tables while watching
    duration: float = 0.0  # stop watching after this many seconds; 0 = until Ctrl-C

    # The overheat fit needs 20 samples at temp_sample_hz before it reports
    # anything, so a shorter settle prints a table with no estimates on it.
    settle: float = 2.5

    follower_port: str = MISSING
    follower_id: str = "b601_follower"


ConfigStore.instance().store(name="temperatures_schema", node=TemperaturesConfig)


def warn_on_silent_motors(robot: RebotB601Follower) -> None:
    """A motor that never answered keeps the 0C its shared slot started at, and
    0C would otherwise read as the coldest joint on the arm."""
    temps = robot.motor_temperatures()
    silent = [name for name, t in temps.items() if t["mosfet"] == 0.0 and t["rotor"] == 0.0]
    if silent:
        announce(
            f"{', '.join(silent)} reported no temperature; 0C above means no answer, not cold.",
            YELLOW,
        )


@hydra.main(version_base=None, config_path="configs", config_name="temperatures")
def main(cfg: TemperaturesConfig) -> None:
    quiet_console_logging()

    robot = RebotB601Follower(
        RebotB601FollowerRobotConfig(
            port=cfg.follower_port,
            id=cfg.follower_id,
            can_adapter="damiao",
            control_mode="mit",
            # nothing is commanded here, so there is no pose to walk home from and
            # no weight being carried: connecting must not become a reason to move
            return_home_on_disconnect=False,
            gravity_compensation=False,
            # publishes motor_overheat_etas(); this script is the something-watching
            # that makes the refit worth its eighth of a core
            temp_debug=True,
        )
    )

    events = {"exit_early": False, "stop_recording": False}
    install_graceful_sigint_handler(events)
    try:
        # calibration only matters to code that converts angles; temperatures are
        # reported the same whatever the zero pose is
        robot.connect(calibrate=False)
        # an enabled-but-uncommanded motor faults on its own comm timeout, and a
        # passive read has nothing to command
        robot.disable_torque()
        sleep_until(time.perf_counter() + cfg.settle, events)

        announce(
            "temp columns: " + "  ".join(f"{i + 1}={n}" for i, n in enumerate(robot.motor_names)),
            CYAN,
        )
        # A fit over 20-odd samples of 1C-quantised readings will occasionally
        # read noise on an idle motor as a climb, and print a minutes-away
        # estimate for a joint sitting at 32C. Watch it for a while before
        # believing one.
        announce(
            "overheat is the follower's own fit over its last temp_history_s of readings: "
            "trustworthy on a motor that is actually heating, flickery on an idle one.",
            CYAN,
        )
        # after the connect, so `t=` and `duration` both measure time watched
        started = time.perf_counter()
        while True:
            print_temperature_table(robot, time.perf_counter() - started)
            warn_on_silent_motors(robot)
            elapsed = time.perf_counter() - started
            if not cfg.watch or events["exit_early"]:
                break
            if cfg.duration and elapsed >= cfg.duration:
                break
            sleep_until(time.perf_counter() + cfg.interval, events)
    finally:
        if robot.is_connected:
            robot.disconnect()


def sleep_until(deadline: float, events: dict) -> None:
    """Wait, but give up as soon as Ctrl-C has asked the script to stop."""
    while time.perf_counter() < deadline and not events["exit_early"]:
        time.sleep(0.05)


if __name__ == "__main__":
    main()
