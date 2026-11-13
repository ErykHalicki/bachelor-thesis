#!/bin/bash
#SBATCH --job-name=wm
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --output=outputs/slurm/%x-%j.out
# Train the mixed-embodiment world model on eryk-pc's V100. Submit from the repo root:
#   sbatch scripts/local/wm_train.sbatch configs/wm/oxe_small.yaml [key=value ...]
set -euo pipefail
VENV=${VENV:-$HOME/School/thesis/bachelor-thesis/.venv}
export LD_LIBRARY_PATH=$(ls -d "$VENV"/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PWD/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$VENV/bin/python" -m thesis.wm.train --config "$@"
