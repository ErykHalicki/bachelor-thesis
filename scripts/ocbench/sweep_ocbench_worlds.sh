#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
for n in "$@"; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 -lms 500 > /tmp/ocb_mem_$n.txt &
  smi=$!
  echo "=== nworld=$n"
  .venv/bin/python scripts/ocbench/bench_ocbench_demos.py --num_demos $n --repeats 1 2>&1 | grep -E "warmup|\[run|Error|error|out of memory" | grep -v "To disable"
  kill $smi
  echo "peak V100 mem: $(sort -n /tmp/ocb_mem_$n.txt | tail -1) MiB"
done
