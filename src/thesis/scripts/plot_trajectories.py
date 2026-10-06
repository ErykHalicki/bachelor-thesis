#!/usr/bin/env python3
"""Every episode's end-effector path in one interactive 3D plot.

The spatial half of what the frame overlays show: where the gripper actually went, one
line per episode, rotatable with the mouse. Tight bundles mean repeatable demonstrations;
a fan means the arm took a different route each time; distinct separated strands mean
the task is multimodal, which is what decides whether a unimodal regression policy can
imitate it at all or will average two valid paths into one invalid one.

Joint angles come from the dataset and go through the follower's own URDF forward
kinematics, so the positions are the arm's real geometry rather than a proxy -- the same
`link_frames` the gravity compensator uses, and `end_link` is the tool frame.

    python src/thesis/scripts/plot_trajectories.py ehalicki/b601_pusht_pick_and_place
    python src/thesis/scripts/plot_trajectories.py <repo_id> --field action --save paths.png

`--field state` (the default) is where the arm was, `--field action` is where the leader
told it to go; the gap between them is teleoperation tracking error. Nothing here decodes
video, so it reads only the parquet columns and takes seconds.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo_id", help="lerobot dataset, e.g. ehalicki/b601_pusht_pick_and_place")
    p.add_argument("--root", default=None, help="local dataset root (default: the HF cache)")
    p.add_argument("--field", default="state", choices=("state", "action"),
                   help="joint angles to plot (default: %(default)s)")
    p.add_argument("--episodes", type=int, default=None,
                   help="plot only the first N episodes (default: all)")
    p.add_argument("--stride", type=int, default=1,
                   help="keep every Nth frame along each path (default: %(default)s)")
    p.add_argument("--mode", default="lines", choices=("lines", "heat"),
                   help="one line per episode, or a voxel density cloud (default: %(default)s)")
    p.add_argument("--voxel", type=float, default=0.01,
                   help="heat mode: voxel edge in metres (default: %(default)s)")
    p.add_argument("--heat-by", default="episodes", choices=("episodes", "frames"),
                   help="heat mode: count distinct episodes visiting a voxel, or frames "
                        "spent in it (default: %(default)s)")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="heat mode: gaussian blur in voxels, turning the discrete bins "
                        "into a continuous field; 0 keeps the raw counts (default: %(default)s)")
    p.add_argument("--floor", type=float, default=0.02,
                   help="heat mode with --smooth: drop cells below this fraction of the "
                        "peak density (default: %(default)s)")
    p.add_argument("--save", default=None,
                   help="write a PNG here instead of opening a window")
    return p.parse_args()


def dataset_root(repo_id, root):
    """A directory holding this dataset's `data/` parquet.

    A dataset recorded on this machine is already there. One that only lives on the hub
    is pulled down column-side only: `LeRobotDatasetMetadata` alone fetches `meta/` and
    leaves the parquet to be materialized lazily per row by `LeRobotDataset`, which this
    script deliberately does not use. Videos are excluded from the fetch -- nothing here
    reads a frame, and they are the whole weight of the repo.
    """
    if root:
        return Path(root)

    from lerobot.utils.constants import HF_LEROBOT_HOME

    local = HF_LEROBOT_HOME / repo_id
    if (local / "data").exists():
        return local

    from huggingface_hub import snapshot_download

    print(f"  no local copy of {repo_id}; fetching its data columns (no videos)")
    return Path(snapshot_download(
        repo_id, repo_type="dataset", allow_patterns=["meta/*", "data/**"],
    ))


def read_episodes(root, field, limit, stride):
    """{episode index: (T, 6) joint angles in degrees}, straight from the parquet.

    Reads the columns rather than going through LeRobotDataset, which would decode a
    video frame per row for a plot that uses none of them.
    """
    import pandas as pd

    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        sys.exit(f"no parquet under {root / 'data'}")
    column = "observation.state" if field == "state" else "action"
    frame = pd.concat(
        [pd.read_parquet(f, columns=[column, "episode_index", "frame_index"]) for f in files],
        ignore_index=True,
    )

    episodes = {}
    for index in sorted(frame["episode_index"].unique())[:limit]:
        rows = frame[frame["episode_index"] == index].sort_values("frame_index")
        # both columns start with the six arm joints; `state` continues into torque and
        # velocity, and its seventh entry is the gripper -- neither is part of the chain
        angles = np.stack(rows[column].to_numpy())[:, :6]
        episodes[int(index)] = angles[::stride]
    return episodes


def forward_kinematics(angles_deg):
    """(T, 6) joint angles in degrees -> (T, 3) tool positions in metres."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "external/lerobot/src"))
    from lerobot.robots.rebot_b601_follower.gravity_model import link_frames

    # link_frames takes radians; end_link (the last frame) is the tool
    return np.array([link_frames(np.radians(q))[-1][1] for q in angles_deg])


def smoothed_density(paths, voxel, heat_by, sigma, floor):
    """A continuous density field: the same voxel counts blurred by a gaussian, sampled
    back at cell centres.

    Discrete counts are spiky because a path either clips a 1 cm cell or misses it, so
    neighbouring cells can read 1 and 40. Convolving with a gaussian of `sigma` cells
    turns that into a smooth field where a near miss still contributes, which is what
    makes the cloud read as continuous rather than as scattered dice.

    Cells below `floor` of the peak are dropped rather than drawn at zero alpha -- they
    are the overwhelming majority of a bounding box that the arm never enters, and
    keeping them makes rotation crawl for no visible difference.
    """
    from scipy.ndimage import gaussian_filter

    points = np.concatenate([
        np.unique(np.floor(p / voxel).astype(np.int64), axis=0) if heat_by == "episodes"
        else np.floor(p / voxel).astype(np.int64)
        for p in paths.values()
    ])
    low = points.min(axis=0)
    shape = points.max(axis=0) - low + 1
    grid = np.zeros(shape, dtype=float)
    np.add.at(grid, tuple((points - low).T), 1.0)

    grid = gaussian_filter(grid, sigma=sigma, mode="constant")
    occupied = np.argwhere(grid > floor * grid.max())
    return (occupied + low + 0.5) * voxel, grid[tuple(occupied.T)]


def voxel_density(paths, voxel, heat_by):
    """Tool positions binned onto a voxel grid -> (centres, counts).

    `episodes` counts how many distinct episodes ever entered a voxel, which is the
    question "how much do the paths agree": a voxel at 48 is on every run's route, one
    at 1 belongs to a single episode. `frames` counts samples instead, so it measures
    dwell time and lights up wherever the arm moved slowly -- a grasp reads as hot even
    if every episode passes through the same small volume quickly.
    """
    counts = {}
    for path in paths.values():
        index = np.floor(path / voxel).astype(np.int64)
        if heat_by == "episodes":
            index = np.unique(index, axis=0)
        for key in map(tuple, index):
            counts[key] = counts.get(key, 0) + 1
    keys = np.array(list(counts))
    return (keys + 0.5) * voxel, np.array(list(counts.values()), dtype=float)


def main():
    args = parse_args()
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    root = dataset_root(args.repo_id, args.root)
    episodes = read_episodes(root, args.field, args.episodes, args.stride)
    if not episodes:
        sys.exit(f"{args.repo_id} has no episodes")
    paths = {index: forward_kinematics(angles) for index, angles in episodes.items()}

    figure = plt.figure(figsize=(11, 9))
    ax = figure.add_subplot(projection="3d")
    starts = np.array([p[0] for p in paths.values()])
    ends = np.array([p[-1] for p in paths.values()])

    if args.mode == "heat":
        if args.smooth:
            centres, counts = smoothed_density(
                paths, args.voxel, args.heat_by, args.smooth, args.floor)
        else:
            centres, counts = voxel_density(paths, args.voxel, args.heat_by)
        # log scaled: nearly every voxel is visited by one episode and a handful by all,
        # so a linear ramp renders as blank paper. The colourbar carries the same norm.
        norm = LogNorm(vmin=max(counts.min(), counts.max() / 100), vmax=counts.max())
        weight = np.asarray(norm(counts))
        # sampled off the colormap's faint end so the sparsest voxels recede instead
        # of disappearing
        colours = plt.get_cmap("Blues")(0.25 + 0.75 * weight)
        colours[:, 3] = 0.2 + 0.8 * weight
        dots = ax.scatter(*centres.T, c=colours, s=8 + 40 * weight, depthshade=False,
                          edgecolors="none")
        dots.set(array=counts, cmap="Blues", norm=norm)
        bar = figure.colorbar(dots, ax=ax, shrink=0.6, pad=0.1)
        unit = "episodes visiting" if args.heat_by == "episodes" else "frames spent in"
        density = " (gaussian smoothed)" if args.smooth else ""
        bar.set_label(f"{unit} a {args.voxel * 100:g} cm voxel{density}")
    else:
        for colour, (index, path) in zip(
            plt.get_cmap("viridis")(np.linspace(0, 1, len(paths))),
            sorted(paths.items()), strict=True,
        ):
            ax.plot(*path.T, color=colour, linewidth=0.9, alpha=0.7, label=f"ep {index}")
        ax.scatter(*starts.T, color="tab:green", s=28, depthshade=False, label="start")
        ax.scatter(*ends.T, color="tab:red", s=28, depthshade=False, label="end")

    # equal aspect from the data's own extent: with free aspect a tight bundle
    # looks like a fan
    span = np.concatenate(list(paths.values()))
    ax.set_box_aspect(np.ptp(span, axis=0))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"{args.repo_id}  --  {len(paths)} episodes, {args.field} tool path"
        + ("" if args.mode == "lines" else f"  ({args.heat_by} per voxel)"))
    if args.mode == "lines":
        ax.legend(handles=ax.collections, loc="upper right")

    spread = np.linalg.norm(starts - starts.mean(0), axis=1)
    print(f"  {len(paths)} episodes, {sum(len(p) for p in paths.values())} points")
    print(f"  start spread  {spread.mean() * 1000:6.1f} mm mean from centroid")
    print(f"  end spread    {np.linalg.norm(ends - ends.mean(0), axis=1).mean() * 1000:6.1f} mm")

    if args.save:
        figure.savefig(args.save, dpi=130, bbox_inches="tight")
        print(f"  wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
