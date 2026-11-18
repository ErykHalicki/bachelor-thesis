#!/bin/bash
#SBATCH --job-name=oxe-download
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --output=outputs/slurm/%x-%j.out
# Download every Open-X dataset in configs/mixture/rlds_datasets.yaml (40 real-robot datasets,
# DROID included, ~5.3 TB of RLDS) straight from gs://gresearch/robotics and convert each into
# LeRobot v3 under $OUT: RGB fit inside 320x240 as h264 (NVENC on half the V100, so training
# can share the card), every state and action field kept as recorded.
#
#   sbatch scripts/local/oxe_download_all.sbatch                     # everything
#   sbatch scripts/local/oxe_download_all.sbatch --only jaco_play viola
#
# Submit from the repo root. Safe to resubmit after any interruption: finished datasets are
# skipped and a partial one resumes from its last checkpoint. Progress: $LOGS/summary.json and
# $LOGS/<dataset>.log. OUT, TMP and LOGS override the default locations (all on the NVMe).
set -euo pipefail
OUT=${OUT:-$HOME/data/oxe/rlds}
TMP=${TMP:-$HOME/data/oxe/_rlds_tmp}
LOGS=${LOGS:-outputs/oxe_rlds}
VENV=${VENV:-$HOME/School/thesis/bachelor-thesis/.venv}
export PYTHONPATH=$PWD/src
export LD_LIBRARY_PATH=$(ls -d $VENV/lib/python*/site-packages/nvidia/*/lib | tr "\n" ":")${LD_LIBRARY_PATH:-}
exec "$VENV/bin/python" -u scripts/oxe/convert_rlds_all.py \
  --list configs/mixture/rlds_datasets.yaml --out "$OUT" --tmp "$TMP" --logs "$LOGS" \
  --lanes 2 --decode-threads 4 "$@"
