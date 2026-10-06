"""Throughput of OCBench scripted demo collection on MJWarp (nothing is saved)."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

OCBENCH = Path(__file__).resolve().parents[1] / "external" / "ocbench"
sys.path.insert(0, str(OCBENCH / "impls"))

import ocbench
import warp as wp
from envs.streaming_data import _generate_mjwarp_dataset_batch, make_collection_seeds

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="block-single-task1-v0")
parser.add_argument("--num_demos", type=int, default=100)
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()

print("warp device:", wp.get_device().name)
family, backend, env_kwargs, max_steps = ocbench.parse_env_spec(args.env_name)
assert backend == "mjwarp"


def run(collection_idx, n=args.num_demos):
    reset_seeds = make_collection_seeds(0, collection_idx, "train_reset", n)
    oracle_seeds = make_collection_seeds(0, collection_idx, "train_oracle", n)
    wp.synchronize()
    t0 = time.perf_counter()
    _, lengths, successes, _, info = _generate_mjwarp_dataset_batch(
        family, env_kwargs, max_steps, reset_seeds, oracle_seeds
    )
    wp.synchronize()
    return time.perf_counter() - t0, lengths, successes, info


t, *_ = run(999, n=8)
print(f"warmup (8 worlds): {t:.2f}s")

for i in range(args.repeats):
    t, lengths, successes, info = run(i)
    healthy = np.array([x["healthy"] for x in info])
    transitions = int(lengths.sum())
    print(
        f"[run {i}] {args.num_demos} demos in {t:.2f}s | "
        f"{args.num_demos / t:.2f} demos/s | "
        f"{transitions / t:,.0f} demo steps/s ({transitions} steps, mean len {lengths.mean():.0f}) | "
        f"{args.num_demos * max_steps / t:,.0f} world-steps/s (incl. parked worlds, {max_steps} steps/episode cap) | "
        f"success {successes.mean():.2f}, healthy {healthy.mean():.2f}"
    )
