#!/usr/bin/env bash
# Record a LeRobot dataset by teleoperating the reBot B601-DM follower with the
# reBot Arm 102 leader, cameras included. Task-agnostic wrapper around
# src/thesis/scripts/b601/record.py: this handles USB port detection/persistence and
# control-mode reset (same as teleop.sh), then hands off to the Python script
# for the actual recording loop / task-switching / dataset writing.
#
# record.py is a Hydra app. Each dataset gets its own config file under
# src/thesis/scripts/b601/configs/ (see configs/example_task.yaml for the
# template), selected with --config-name, so recording settings stay fixed
# across sessions of the same dataset instead of being re-typed as CLI
# overrides:
#   ./src/thesis/scripts/b601/record.sh --config-name=<dataset> [key=value ...]
#
# Run with --help to see the full resolved config.
#
# --recalibrate  Re-run zero-pose calibration for the follower and the leader
#                before recording starts. Overwrites the saved calibration files.
# --update-mode  Reset the follower's control mode (see below) before recording
#                starts. Only needed after a motor was left in the wrong mode by
#                an earlier run/crash, skip it on routine recording runs.
# --dry-run      Teleoperate under this config and write nothing: same cameras,
#                control mode and stream, no dataset and no upload. For checking a
#                rig -- framing, tracking, control rate -- before collecting on it:
#                  ./src/thesis/scripts/b601/record.sh --config-name=<dataset> --dry-run
set -euo pipefail

RECALIBRATE=0
UPDATE_MODE=0
# Hydra's CLI requires its own flags before any key=value override, or it errors
# with "unrecognized arguments". Split them so callers can type either order;
# a space-separated `--config-name foo` keeps its value glued to it.
HYDRA_FLAGS=()
OVERRIDE_ARGS=()
args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
    arg="${args[$i]}"
    case "$arg" in
        --recalibrate) RECALIBRATE=1 ;;
        --update-mode) UPDATE_MODE=1 ;;
        # passed to Hydra as an override, since its CLI rejects any --flag it does not define
        --dry-run) OVERRIDE_ARGS+=("dry_run=true") ;;
        --config-name)
            i=$((i + 1))
            HYDRA_FLAGS+=("$arg" "${args[$i]}")
            ;;
        --*) HYDRA_FLAGS+=("$arg") ;;
        *) OVERRIDE_ARGS+=("$arg") ;;
    esac
    i=$((i + 1))
done

# fail fast before touching the venv, ports or motors: repo_id has no default
HAS_CONFIG=0
for arg in "${HYDRA_FLAGS[@]}" "${OVERRIDE_ARGS[@]}"; do
    case "$arg" in
        --config-name=*|--config-name|repo_id=*) HAS_CONFIG=1 ;;
    esac
done
if [[ "$HAS_CONFIG" -eq 0 ]]; then
    echo "Missing dataset config: pass --config-name=<dataset> (see src/thesis/scripts/b601/configs/) or repo_id=<hf_username>/<dataset_name> directly." >&2
    echo "Usage: ./src/thesis/scripts/b601/record.sh --config-name=<dataset> [key=value ...]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

# read by `datasets` at import time, and it outranks disable_progress_bars(),
# which the Hub client undoes while resuming a dataset
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

ENV_FILE="$REPO_ROOT/src/thesis/scripts/b601/.env"
CALLER_NAME="record.sh"
source "$REPO_ROOT/src/thesis/scripts/b601/common.sh"

FOLLOWER_ID="${FOLLOWER_ID:-b601_follower}"
LEADER_ID="${LEADER_ID:-b601_leader}"
CONTROL_MODE="${CONTROL_MODE:-mit}"

resolve_ports

echo
echo "Follower: $FOLLOWER_PORT (id=$FOLLOWER_ID)"
echo "Leader:   $LEADER_PORT (id=$LEADER_ID)"
echo "Control mode: $CONTROL_MODE"
echo

# lerobot's ensure_mode does not persist a CTRL_MODE change, so a motor left in
# the wrong mode by an earlier run makes connect() fail
if [[ "$UPDATE_MODE" -eq 1 ]]; then
    echo "Resetting follower control modes..."
    python src/thesis/scripts/b601/set_control_mode.py --port "$FOLLOWER_PORT" --mode "$CONTROL_MODE"
    echo
fi

if [[ "$RECALIBRATE" -eq 1 ]]; then
    echo "=== Recalibration requested ==="
    recalibrate_device robot rebot_b601_follower robots \
        "$FOLLOWER_PORT" "$FOLLOWER_ID" \
        "--robot.can_adapter=damiao --robot.control_mode=$CONTROL_MODE"
    recalibrate_device teleop rebot_102_leader teleoperators \
        "$LEADER_PORT" "$LEADER_ID" ""
    echo
fi

python src/thesis/scripts/b601/record.py \
    "${HYDRA_FLAGS[@]}" \
    follower_port="$FOLLOWER_PORT" \
    leader_port="$LEADER_PORT" \
    follower_id="$FOLLOWER_ID" \
    leader_id="$LEADER_ID" \
    control_mode="$CONTROL_MODE" \
    "${OVERRIDE_ARGS[@]}"
