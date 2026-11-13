#!/bin/bash
# Two training runs sharing one Sockeye V100, like the MPS shards on eryk-pc:
#
#   STAGE="ocbench_official ocbench_multi8/hanoi-single-task1" \
#     sbatch scripts/sockeye/train_pair.sbatch [shared overrides] -- [run A overrides] -- [run B overrides]
#
# Each STAGE entry (relative to $SCRATCH/data) is copied to the node's local NVMe under
# the same relative path, and exported as STAGED, so overrides point at it with
# '${oc.env:STAGED}/<entry>'. Only the task dirs a pair reads are staged, not all of multi8.
#
# Two processes on one GPU time-slice it; under MPS their kernels overlap instead (on
# eryk-pc: 1.62x the throughput of one run, against 1.33x plain). Sockeye does not offer
# MPS as a gres, so the job starts its own daemon when the node has the binary.
#SBATCH --account=st-dushan20-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=thesis-train-pair
#SBATCH --output=/scratch/st-dushan20-1/ehalicki/logs/%x-%j.out

set -euo pipefail
SCRATCH=/scratch/st-dushan20-1/ehalicki
export MUJOCO_GL=egl WANDB_MODE=offline PYTHONUNBUFFERED=1
export WARP_CACHE_PATH=$SCRATCH/warp_cache
export WANDB_DIR=$SCRATCH/wandb/run WANDB_CACHE_DIR=$SCRATCH/wandb/cache WANDB_DATA_DIR=$SCRATCH/wandb/data
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"

shared=() run_a=() run_b=() part=0
for arg in "$@"; do
    if [[ $arg == -- ]]; then part=$((part + 1)); continue; fi
    case $part in
        0) shared+=("$arg") ;;
        1) run_a+=("$arg") ;;
        2) run_b+=("$arg") ;;
        *) echo "more than two runs given" >&2; exit 1 ;;
    esac
done
[[ $part == 2 ]] || { echo "usage: [shared] -- [run A] -- [run B]" >&2; exit 1; }

export STAGED=${SLURM_TMPDIR:-/tmp}/thesis-data-$SLURM_JOB_ID
export CUDA_MPS_PIPE_DIRECTORY=$STAGED/mps-pipe CUDA_MPS_LOG_DIRECTORY=$STAGED/mps-log
mps=0
cleanup() {
    if [[ $mps == 1 ]]; then echo quit | nvidia-cuda-mps-control || true; fi
    rm -rf "$STAGED"
}
trap cleanup EXIT
mkdir -p "$STAGED"
start=$(date +%s)
(cd "$SCRATCH/data" && cp -r --parents ${STAGE:?set STAGE to the data dirs to copy} "$STAGED/")
echo "staged $STAGE to $STAGED in $(( $(date +%s) - start ))s ($(du -sh "$STAGED" | cut -f1))"

echo "job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name,memory.total,compute_mode --format=csv,noheader)"
if command -v nvidia-cuda-mps-control >/dev/null; then
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if nvidia-cuda-mps-control -d; then mps=1; fi
fi
echo "mps: $mps"

workers=$(( SLURM_CPUS_PER_TASK / 2 - 1 ))
launch() {
    .venv/bin/python main.py \
        experiment.batch_size=auto experiment.effective_batch=256 \
        experiment.num_workers=$workers \
        wandb.mode=offline \
        "${shared[@]}" "$@"
}
# hydra names its output dir by the second, so the two must not start together
launch "${run_a[@]}" > >(sed -u 's/^/[A] /') 2>&1 &
pid_a=$!
sleep 5
launch "${run_b[@]}" > >(sed -u 's/^/[B] /') 2>&1 &
pid_b=$!

status=0
wait $pid_a || status=$?
wait $pid_b || status=$?
exit $status
