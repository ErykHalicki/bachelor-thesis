#!/bin/bash
# One training run on eryk-pc's V100 through the local Slurm:
#
#   sbatch scripts/local/train.sbatch run=<arm> [more hydra overrides...]
#
# The job sees only its card, as device 0 (CUDA_DEVICE_ORDER=PCI_BUS_ID from the prolog).
#SBATCH --partition=main
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=22000
#SBATCH --job-name=thesis-train
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
# torchcodec needs the NPP/nvJPEG libs the cu126 torch wheel does not bundle (docs/v100_setup.md)
export LD_LIBRARY_PATH=$(ls -d "$PWD"/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
echo "job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
export MUJOCO_GL=egl
srun .venv/bin/python main.py experiment.num_workers=$((SLURM_CPUS_PER_TASK - 1)) "$@"
