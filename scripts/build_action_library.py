"""Sample recorded action segments from a LeRobot dataset's training episodes into a .npy
library for the `demo_shooting` planner, normalized like the model's actions (mean_std).

    .venv/bin/python scripts/build_action_library.py --run hjepa_ocbench_fused \\
        --steps 150 --count 20000 --out ~/data/lerobot/ocbench_action_library_150.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="run config whose dataset and normalization to use")
p.add_argument("--steps", type=int, required=True)
p.add_argument("--count", type=int, default=20000)
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

from lerobot.datasets import LeRobotDataset  # noqa: E402
from test_configs import _compose  # noqa: E402
from thesis.datasets.lerobot import _arrow_column, split_episodes  # noqa: E402

cfg = _compose(("base", a.run)).dataset
ds = LeRobotDataset(cfg.repo_id, root=cfg.get("root"))
lengths = np.asarray(ds.meta.episodes["length"])
starts = np.asarray(ds.meta.episodes["dataset_from_index"])
train = split_episodes(range(len(lengths)), cfg.get("validation_split", 0.0), "train",
                       seed=int(cfg.get("validation_split_seed", 0)))
actions = _arrow_column(ds.reader.hf_dataset, "action")
train_rows = np.concatenate([np.arange(starts[e], starts[e] + lengths[e]) for e in train])
mean, std = actions[train_rows].mean(0), actions[train_rows].std(0)
rng = np.random.default_rng(a.seed)
eps = rng.choice([e for e in train if lengths[e] >= a.steps], size=a.count)
seg = np.stack([actions[starts[e] + (t := rng.integers(0, lengths[e] - a.steps + 1)): starts[e] + t + a.steps]
                for e in eps])
lib = ((seg - mean) / std).astype(np.float32)
Path(a.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
np.save(Path(a.out).expanduser(), lib)
print(f"{lib.shape} segments -> {a.out}")
