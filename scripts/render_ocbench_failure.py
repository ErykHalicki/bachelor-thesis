"""Collect MJWarp scripted demos with a given time limit and render one failed episode to mp4."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np

OCBENCH = Path(__file__).resolve().parents[1] / "external" / "ocbench"
sys.path.insert(0, str(OCBENCH / "impls"))

import ocbench
from envs.streaming_data import _generate_mjwarp_dataset_batch, make_collection_seeds

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="block-single-task1-v0")
parser.add_argument("--nworld", type=int, default=400)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--failure_idx", type=int, default=0)
parser.add_argument("--frame_skip", type=int, default=4)
parser.add_argument("--camera", default="front")
parser.add_argument("--out", default=os.path.expanduser("~/ocb_bench/failure.mp4"))
args = parser.parse_args()

family, _, env_kwargs, _ = ocbench.parse_env_spec(args.env_name)
reset_seeds = make_collection_seeds(0, 0, "train_reset", args.nworld)
oracle_seeds = make_collection_seeds(0, 0, "train_oracle", args.nworld)
dataset, lengths, successes, _, info = _generate_mjwarp_dataset_batch(
    family, env_kwargs, args.max_steps, reset_seeds, oracle_seeds, return_metadata=True
)
healthy = np.array([x["healthy"] for x in info])
failed = np.flatnonzero(~successes & healthy)
print(f"success {successes.mean():.4f}, {len(failed)} healthy failures out of {args.nworld}")
ep = failed[args.failure_idx]
offsets = np.concatenate([[0], np.cumsum(lengths)])
qpos = dataset["qpos"][offsets[ep] : offsets[ep + 1]]
print(f"episode {ep}: length {lengths[ep]}, mistakes p={info[ep].get("p_mistake")}")
for seg in info[ep].get("segments", []):
    print("  segment", {k: v for k, v in seg.items()})

cpu_name = args.env_name.replace(f"{family}-", f"{family}-cpu-", 1)
cpu_env = ocbench.make(cpu_name).unwrapped
cpu_env.reset()
model, data = cpu_env._model, cpu_env._data
assert model.nq == qpos.shape[1], (model.nq, qpos.shape)
model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
renderer = mujoco.Renderer(model, height=480, width=640)
fps = 1.0 / (cpu_env._control_timestep * args.frame_skip)
writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8)
for t in range(0, len(qpos), args.frame_skip):
    data.qpos[:] = qpos[t]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=args.camera)
    writer.append_data(renderer.render())
writer.close()
print(f"wrote {args.out}: {len(qpos) // args.frame_skip} frames at {fps:.1f} fps (real time)")
