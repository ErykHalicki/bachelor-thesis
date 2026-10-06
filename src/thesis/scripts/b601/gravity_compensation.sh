#!/usr/bin/env bash
# Float the reBot B601-DM follower on its own gravity model so it can be pushed
# around by hand. Wrapper around src/thesis/scripts/b601/gravity_compensation.py:
# this handles USB port detection/persistence and control-mode reset (same as
# teleop.sh / record.sh), then hands off to the Python script for the control
# loop.
#
# gravity_compensation.py is a Hydra app (config:
# src/thesis/scripts/b601/configs/gravity_compensation.yaml), so anything in
# GravityCompConfig is settable as a key=value override:
#   ./src/thesis/scripts/b601/gravity_compensation.sh [key=value ...]
#
# Run with --help to see the full resolved config.
#
# Check the model before letting the arm carry itself -- this reads the pose and
# prints the torques without ever moving the arm:
#   ./src/thesis/scripts/b601/gravity_compensation.sh dry_run=true
#
# --recalibrate  Re-run zero-pose calibration for the follower first. The model
#                reads raw motor angles, so a wrong zero means wrong torques.
# --update-mode  Reset the follower's control mode before starting. Only needed
#                after a motor was left in the wrong mode by an earlier
#                run/crash; gravity compensation requires MIT mode.
set -euo pipefail

RECALIBRATE=0
UPDATE_MODE=0
# Hydra's CLI requires its own flags before any key=value override, or it errors
# with "unrecognized arguments"
HYDRA_FLAGS=()
OVERRIDE_ARGS=()
args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
    arg="${args[$i]}"
    case "$arg" in
        --recalibrate) RECALIBRATE=1 ;;
        --update-mode) UPDATE_MODE=1 ;;
        --config-name)
            i=$((i + 1))
            HYDRA_FLAGS+=("$arg" "${args[$i]}")
            ;;
        --*) HYDRA_FLAGS+=("$arg") ;;
        *) OVERRIDE_ARGS+=("$arg") ;;
    esac
    i=$((i + 1))
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

ENV_FILE="$REPO_ROOT/src/thesis/scripts/b601/.env"
CALLER_NAME="gravity_compensation.sh"
source "$REPO_ROOT/src/thesis/scripts/b601/common.sh"

FOLLOWER_ID="${FOLLOWER_ID:-b601_follower}"

# shared with teleop.sh/record.sh, so it resolves the leader too; only the
# follower is used here
resolve_ports

echo
echo "Follower: $FOLLOWER_PORT (id=$FOLLOWER_ID)"
echo

# lerobot's ensure_mode does not persist a CTRL_MODE change, so a motor left in
# the wrong mode by an earlier run makes connect() fail. Feedforward needs MIT.
if [[ "$UPDATE_MODE" -eq 1 ]]; then
    echo "Resetting follower control modes..."
    python src/thesis/scripts/b601/set_control_mode.py --port "$FOLLOWER_PORT" --mode mit
    echo
fi

if [[ "$RECALIBRATE" -eq 1 ]]; then
    echo "=== Recalibration requested ==="
    recalibrate_device robot rebot_b601_follower robots \
        "$FOLLOWER_PORT" "$FOLLOWER_ID" \
        "--robot.can_adapter=damiao --robot.control_mode=mit"
    echo
fi

python src/thesis/scripts/b601/gravity_compensation.py \
    "${HYDRA_FLAGS[@]}" \
    follower_port="$FOLLOWER_PORT" \
    follower_id="$FOLLOWER_ID" \
    "${OVERRIDE_ARGS[@]}"
