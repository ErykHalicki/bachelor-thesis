#!/bin/bash
#SBATCH --job-name=oxe-rlds
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=outputs/slurm/%x-%j.out
# Convert Open-X RLDS datasets, straight from gs://gresearch/robotics, into LeRobot v3 with
# NVENC on half the V100 (a shard, so training can share the card):
#   sbatch scripts/local/oxe_rlds.sbatch jaco_play:10 viola:20 ...     (name:control Hz)
# Submit from the repo root. OUT and TMP override the output and shard staging directories.
set -euo pipefail
OUT=${OUT:-$HOME/data/oxe/rlds}
TMP=${TMP:-$HOME/data/oxe/_rlds_tmp}
VENV=${VENV:-$HOME/School/thesis/bachelor-thesis/.venv}
export PYTHONPATH=$PWD/src
export LD_LIBRARY_PATH=$(ls -d $VENV/lib/python*/site-packages/nvidia/*/lib | tr "\n" ":")${LD_LIBRARY_PATH:-}
for spec in "$@"; do
  "$VENV/bin/python" -u -m thesis.datasets.mixture.rlds "${spec%%:*}" --fps "${spec##*:}" \
    --out "$OUT" --tmp "$TMP" --overwrite
done
