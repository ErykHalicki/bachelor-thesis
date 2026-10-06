#!/usr/bin/env bash
# Print the reBot B601-DM's motor temperatures and how long its thermal
# protection reckons each motor has left. Wrapper around
# src/thesis/scripts/b601/temperatures.py: this handles USB port
# detection/persistence (same as teleop.sh / record.sh), then hands off.
#
# temperatures.py is a Hydra app (config:
# src/thesis/scripts/b601/configs/temperatures.yaml), so anything in
# TemperaturesConfig is settable as a key=value override:
#   ./src/thesis/scripts/b601/temperatures.sh [key=value ...]
#   ./src/thesis/scripts/b601/temperatures.sh watch=true interval=5
#
# The arm is never commanded and never moves; it is limp throughout, as it is
# before the script starts.
#
# --update-mode  Reset the follower's control mode before connecting. Only
#                needed after a motor was left in the wrong mode by an earlier
#                run/crash, which makes connect() fail.
set -euo pipefail

UPDATE_MODE=0
# Hydra's CLI requires its own flags before any key=value override, or it errors
# with "unrecognized arguments"
HYDRA_FLAGS=()
OVERRIDE_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --update-mode) UPDATE_MODE=1 ;;
        --*) HYDRA_FLAGS+=("$arg") ;;
        *) OVERRIDE_ARGS+=("$arg") ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

ENV_FILE="$REPO_ROOT/src/thesis/scripts/b601/.env"
CALLER_NAME="temperatures.sh"
source "$REPO_ROOT/src/thesis/scripts/b601/common.sh"

FOLLOWER_ID="${FOLLOWER_ID:-b601_follower}"

# shared with teleop.sh/record.sh, so it resolves the leader too; only the
# follower is used here
resolve_ports

echo
echo "Follower: $FOLLOWER_PORT (id=$FOLLOWER_ID)"
echo

if [[ "$UPDATE_MODE" -eq 1 ]]; then
    echo "Resetting follower control modes..."
    python src/thesis/scripts/b601/set_control_mode.py --port "$FOLLOWER_PORT" --mode mit
    echo
fi

python src/thesis/scripts/b601/temperatures.py \
    "${HYDRA_FLAGS[@]}" \
    follower_port="$FOLLOWER_PORT" \
    follower_id="$FOLLOWER_ID" \
    "${OVERRIDE_ARGS[@]}"
