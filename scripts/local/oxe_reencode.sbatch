#!/bin/bash
#SBATCH --job-name=oxe-reencode
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --output=outputs/slurm/%x-%j.out
# Re-encode raw LeRobot v3.0 datasets to GOP-2 h264 with NVENC on the V100 (224x224 by
# default; pass --fit 320x240 to keep aspect ratios, with OUT=~/data/oxe/fit320x240).
#   sbatch scripts/local/oxe_reencode.sbatch taco_play jaco_play ...
# Submit from the repo root. RAW and OUT override the default data locations.
set -euo pipefail
RAW=${RAW:-$HOME/data/oxe/raw}
OUT=${OUT:-$HOME/data/oxe/224}
VENV=${VENV:-$HOME/School/thesis/bachelor-thesis/.venv}
export PYTHONPATH=$PWD/src
exec "$VENV/bin/python" -m thesis.datasets.mixture.reencode --raw "$RAW" --out "$OUT" "$@"
