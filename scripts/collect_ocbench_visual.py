"""Collect OCBench scripted demos with dense state and sparse camera frames, shard by shard.

Runs OCBench's own MJWarp collector (impls/envs/streaming_data.py) on a visual env whose
cameras and resolution are overridden, with `observation_interval` frames per stored image:
OCBench's sparse layout keeps a frame at steps 0, K, 2K, ... plus each episode's final one.
Every shard is written as

    <out>/shard-NNN.npz         actions, rewards, masks, terminals, qpos, qvel,
                                observation_interval (OCBench's own keys, dense per step)
    <out>/shard-NNN-pixels.npy  the sparse frames, (rows, cameras, H, W, 3) uint8

The frames live in their own uncompressed .npy so the dataset backend can memory-map them:
at 256 px a 10k-episode dataset holds ~150 GB of frames, far more than fits in RAM.

Shards are collected until the successful episodes reach --num_successes; failed episodes
are kept in the shards and dropped by the dataset backend's `success_only`.

Needs ocbench's `impls` dependencies (jax, flax), which the repo's `.venv` carries (see
docs/v100_setup.md), e.g. on eryk-pc:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl JAX_PLATFORMS=cpu \
      .venv/bin/python scripts/collect_ocbench_visual.py --out ~/data/ocbench_visual_256
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

OCBENCH = Path(__file__).resolve().parents[1] / "external" / "ocbench"
sys.path.insert(0, str(OCBENCH / "impls"))

import ocbench  # noqa: E402
from envs.streaming_data import _generate_mjwarp_dataset_chunks, make_collection_seeds  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="visual-block-single-task1-v0")
parser.add_argument("--out", required=True)
parser.add_argument("--num_successes", type=int, default=10500)
# a shard's frames sit in RAM until written, twice over while they are joined: ~15 GB at
# 300 episodes and 256 px. Collection is seeded per shard, so keep this equal across machines
# that should hold the same data
parser.add_argument("--shard_episodes", type=int, default=300)
parser.add_argument("--max_parallel", type=int, default=300)
parser.add_argument("--observation_interval", type=int, default=25)
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--cameras", default="front,ur5e/wrist")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

family, backend, env_kwargs, max_steps = ocbench.parse_env_spec(args.env_name)
if env_kwargs.get("ob_type") != "pixels":
    raise ValueError(f"{args.env_name} is not a visual env")
env_kwargs = dict(env_kwargs, width=args.size, height=args.size,
                  pixel_cameras=tuple(args.cameras.split(",")))
out = Path(args.out).expanduser()
out.mkdir(parents=True, exist_ok=True)

successes, shard = 0, 0
while successes < args.num_successes:
    path = out / f"shard-{shard:03d}.npz"
    if path.exists():
        with np.load(path) as data:
            ends = np.flatnonzero(data["terminals"]) + 1
            rewards = data["rewards"]
        successes += sum(rewards[s:e].max() == 1 for s, e in zip(np.r_[0, ends[:-1]], ends))
        print(f"{path.name} exists, {successes} successes so far", flush=True)
        shard += 1
        continue
    started = time.time()
    reset_seeds = make_collection_seeds(args.seed, shard, "train_reset", args.shard_episodes)
    oracle_seeds = make_collection_seeds(args.seed, shard, "train_oracle", args.shard_episodes)
    dataset, lengths, shard_success, _, _ = _generate_mjwarp_dataset_chunks(
        family, env_kwargs, max_steps, reset_seeds, oracle_seeds, args.max_parallel,
        observation_interval=args.observation_interval,
    )
    np.save(out / f"shard-{shard:03d}-pixels.npy", dataset.pop("observations"))
    np.savez(path, **dataset)
    successes += int(shard_success.sum())
    print(f"{path.name}: {len(lengths)} episodes, {shard_success.mean():.3f} success, "
          f"{int(lengths.sum())} steps, {time.time() - started:.0f}s; "
          f"{successes}/{args.num_successes} successes", flush=True)
    shard += 1
