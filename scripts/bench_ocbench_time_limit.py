"""Scripted-demo success rate as a function of episode time limit (MJWarp, nothing recorded)."""
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
from ocbench.mjwarp import make_env
from ocbench.mjwarp.metadata import record_episode_outcomes
from envs.streaming_data import _make_mjwarp_controller, make_collection_seeds

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="block-single-task1-v0")
parser.add_argument("--nworld", type=int, default=8000)
parser.add_argument("--max_steps", type=int, default=5000)
args = parser.parse_args()

family, backend, env_kwargs, default_limit = ocbench.parse_env_spec(args.env_name)
reset_seeds = make_collection_seeds(0, 0, "train_reset", args.nworld)
oracle_seeds = make_collection_seeds(0, 0, "train_oracle", args.nworld)

env = make_env(family, env_kwargs, args.nworld)
env.reset(seeds=reset_seeds)
env._ensure_ik_buffers()
controller = _make_mjwarp_controller(family, env, oracle_seeds, args.max_steps)
controller.reset()
wp = env._wp
wp.synchronize()
assert not controller.done.numpy().any(), "some worlds terminal at reset"
device = env.data.qpos.device
episode_healthy = wp.ones(args.nworld, dtype=wp.int32, device=device)
episode_failure = wp.zeros(args.nworld, dtype=wp.int32, device=device)
no_failure = wp.zeros(args.nworld, dtype=wp.int32, device=device)

t0 = time.perf_counter()
for step in range(args.max_steps):
    target_pos, target_xmat, target_gripper = controller.make_targets()
    action = env.ee_target_arrays_to_joint_actions_gpu(target_pos, target_xmat, target_gripper)
    env.step_joint_actions_gpu(action, controller.done)
    wp.launch(
        record_episode_outcomes,
        dim=args.nworld,
        inputs=[controller.done, env._gpu_healthy, no_failure, episode_healthy, episode_failure],
        device=device,
    )
    controller.update_done()
    if (step + 1) % 250 == 0:
        done = controller.done.numpy().astype(bool)
        print(f"step {step + 1}: {done.mean():.4f} done, {time.perf_counter() - t0:.0f}s", flush=True)
        if done.all():
            break
wp.synchronize()
elapsed = time.perf_counter() - t0

lengths = controller.episode_length.numpy()
success = controller.episode_success.numpy().astype(bool)
healthy = episode_healthy.numpy().astype(bool)
print(f"\n{args.nworld} worlds, {args.env_name} (default limit {default_limit}), {elapsed:.0f}s")
print(f"unhealthy worlds: {(~healthy).sum()} (of which flagged success: {(success & ~healthy).sum()})")
print("time_limit  success_rate  success_rate_healthy_only")
for limit in [500, 750, 1000, default_limit, 1500, 1750, 2000, 2500, 3000, 4000, args.max_steps]:
    if limit <= args.max_steps:
        hit = success & (lengths <= limit)
        print(f"{limit:10d}  {hit.mean():.4f}        {hit[healthy].mean():.4f}")
if success.any():
    print("success length percentiles (50/90/99/max):", np.percentile(lengths[success], [50, 90, 99, 100]).astype(int))
np.savez("/tmp/ocb_time_limit.npz", lengths=lengths, success=success, healthy=healthy)
