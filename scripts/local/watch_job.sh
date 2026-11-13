#!/bin/bash
# Timestamped GPU utilization (1 s), per-process memory (10 s) and log lines for one Slurm job:
#   scripts/local/watch_job.sh <jobid> <job log>   ->   outputs/slurm/watch-<jobid>-{gpu,mem,log}.txt
J=$1; LOG=$2; OUT=outputs/slurm/watch-$J
( while squeue -h -j $J | grep -q .; do
    echo "$(date +%s) $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | grep '^1,' | cut -d, -f2,3)"
    sleep 1; done ) > $OUT-gpu.txt &
( while squeue -h -j $J | grep -q .; do
    echo "== $(date +%s) free_mb=$(free -m | awk '/Mem:/{print $7}')"
    ps -u $USER -o rss=,comm= --sort=-rss | head -15 | awk '{printf "%d %s\n", $1/1024, $2}'
    sleep 10; done ) > $OUT-mem.txt &
( until [ -f "$LOG" ]; do sleep 1; done
  tail -n +1 -F "$LOG" 2>/dev/null | grep --line-buffered -E "^step|dataset:|auto batch|eval|Traceback|Error" | while read -r l; do echo "$(date +%s) $l"; done ) > $OUT-log.txt &
wait
