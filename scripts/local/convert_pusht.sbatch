#!/bin/bash
# Decompress the LeWM Push-T train file and convert it to the LeRobot dataset the hjepa_pusht
# run reads (configs/dataset/pusht_expert.yaml). CPU only:
#
#   sbatch scripts/local/convert_pusht.sbatch [/mnt/hdd/pusht]
#SBATCH --partition=main
#SBATCH --cpus-per-task=10
#SBATCH --mem=20000
#SBATCH --job-name=convert-pusht
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
SRC=${1:-/mnt/hdd/pusht}
if [ ! -f "$SRC/pusht_expert_train.h5" ]; then
  unzstd -T0 "$SRC/pusht_expert_train.h5.zst" -o "$SRC/pusht_expert_train.h5"
fi
.venv/bin/python scripts/convert_pusht_to_lerobot.py --src "$SRC/pusht_expert_train.h5" \
  --root /mnt/hdd/lerobot/pusht_expert --repo_id local/pusht_expert --workers $SLURM_CPUS_PER_TASK
