#!/bin/bash
# One training run on a Sockeye V100:
#
#   sbatch scripts/sockeye/train.sbatch run=<arm> [more hydra overrides...]
#
# The dataset ($DATA, default ocbench_visual_256) is copied from scratch to the node's local
# NVMe first: training reads ~1,500 random frames a step from memory-mapped shards, and over
# Lustre that took ~10 s a step against ~1 s of compute; the job's memory limit also caps how
# much of the dataset the page cache can hold, so it never warms up.
#
# Run from the repo checkout on scratch (compute nodes can write only there). Compute nodes
# have no internet: wandb logs offline into scratch -- sync it from the login node with
# `.venv/bin/wandb sync <WANDB_DIR>/wandb/offline-run-*` -- and anything downloaded at
# build time (ImageNet weights) must already sit in ~/.cache on the shared home.
#SBATCH --account=st-dushan20-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=thesis-train
#SBATCH --output=/scratch/st-dushan20-1/ehalicki/logs/%x-%j.out

set -euo pipefail
SCRATCH=/scratch/st-dushan20-1/ehalicki
# unbuffered: a job's stdout is a file, so prints otherwise sit in an 8 KB buffer for ~30 min
export MUJOCO_GL=egl WANDB_MODE=offline PYTHONUNBUFFERED=1
export WANDB_DIR=$SCRATCH/wandb/run WANDB_CACHE_DIR=$SCRATCH/wandb/cache WANDB_DATA_DIR=$SCRATCH/wandb/data
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"

DATA=${DATA:-ocbench_visual_256}
LOCAL=${SLURM_TMPDIR:-/tmp}/thesis-data-$SLURM_JOB_ID
trap 'rm -rf "$LOCAL"' EXIT
mkdir -p "$LOCAL"
start=$(date +%s)
cp -r "$SCRATCH/data/$DATA" "$LOCAL/"
echo "staged $DATA to $LOCAL in $(( $(date +%s) - start ))s ($(du -sh "$LOCAL/$DATA" | cut -f1))"

echo "job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
srun .venv/bin/python main.py \
    dataset.dataset_path=$LOCAL/$DATA \
    experiment.batch_size=auto experiment.effective_batch=256 \
    experiment.num_workers=$((SLURM_CPUS_PER_TASK - 1)) \
    wandb.mode=offline \
    "$@"
