"""Check an RLDS -> LeRobot conversion of jaco_play against lerobot/jaco_play (v3.0).

    python scripts/oxe/crosscheck_jaco.py ~/data/oxe/rlds/jaco_play ~/data/oxe/raw/jaco_play

Episodes are paired by length and first end-effector pose (the two copies order episodes
differently); paired trajectories must be identical, and frames are compared by PSNR. On
2026-11-13 all 1,085 episodes paired with identical end-effector trajectories and the two
cameras agreed at 33-38 dB (their AV1 vs our h264, both from the source PNGs). lerobot's
`observation.state` is the source end-effector pose with a zero column inserted at index 6,
and its gripper action maps the source's -1/0/+1 commands to a 0/1 signal.
"""
import sys

import numpy as np

from thesis.datasets.mixture.source import LeRobotV3Source

ours, ref = LeRobotV3Source(sys.argv[1]), LeRobotV3Source(sys.argv[2])
print("episodes", ours.num_episodes, ref.num_episodes, "frames", ours.total_frames, ref.total_frames)
ee_o = ours.column("observation.end_effector_cartesian_pos")
ee_r = ref.column("observation.state")[:, [0, 1, 2, 3, 4, 5, 7]]
index = {(int(ref.lengths[e]),) + tuple(np.round(ee_r[ref.first_row[e]], 5)): e
         for e in range(ref.num_episodes)}
pairs = []
for e in range(ours.num_episodes):
    k = (int(ours.lengths[e]),) + tuple(np.round(ee_o[ours.first_row[e]], 5))
    if k in index:
        pairs.append((e, index[k]))
differ = sum(not np.array_equal(ee_o[ours.first_row[e]:ours.first_row[e] + ours.lengths[e]],
                                ee_r[ref.first_row[q]:ref.first_row[q] + ref.lengths[q]])
             for e, q in pairs)
print(f"paired {len(pairs)}/{ours.num_episodes}; paired episodes whose trajectory differs: {differ}")
rng = np.random.default_rng(0)
for cam in ("observation.images.image", "observation.images.image_wrist"):
    ps = []
    for i in rng.choice(len(pairs), min(8, len(pairs)), replace=False):
        e, q = pairs[i]
        T = int(ours.lengths[e])
        ix = [0, T // 2, T - 1]
        x = ours.frames(cam, e, ix).float().numpy()
        y = ref.frames(cam, q, ix).float().numpy()
        ps += list(10 * np.log10(255 ** 2 / np.maximum(((x - y) ** 2).mean(axis=(1, 2, 3)), 1e-6)))
    print(f"{cam}: PSNR min {min(ps):.1f} mean {np.mean(ps):.1f} dB over {len(ps)} frames")
