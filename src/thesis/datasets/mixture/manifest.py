"""Offline index and statistics for one (source, embodiment) pair.

    python -m thesis.datasets.mixture.manifest --config configs/wm/oxe_small.yaml

writes `<data_root>/<source>/manifests/<embodiment>.json` for every member of the mixture.
Training reads manifests only: there is no statistics pass at startup. A manifest does not
depend on the clock. Statistics are per action/state dimension over every native frame,
which is exactly what the model sees when chunks are native actions; the anchors a clock
allows are arithmetic on the episode lengths stored here and are derived at load time.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .embodiment import load_embodiment
from .source import LeRobotV3Source

VERSION = 1


def fingerprint(source, embodiment, val_fraction, seed):
    info = source.info
    payload = json.dumps([VERSION, embodiment.text, info["total_frames"],
                          info["total_episodes"], info["fps"], val_fraction, seed])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _stats(array):
    if array.shape[1] == 0:
        return {k: [] for k in ("q01", "q99", "mean", "std")}
    q01, q99 = np.quantile(array, [0.01, 0.99], axis=0)
    return {"q01": q01.tolist(), "q99": q99.tolist(),
            "mean": array.mean(0).tolist(), "std": array.std(0).tolist()}


def manifest_path(data_root, embodiment):
    return Path(data_root).expanduser() / embodiment.source / "manifests" / f"{embodiment.name}.json"


def build_manifest(data_root, embodiment, val_fraction=0.05, seed=0):
    source = LeRobotV3Source(Path(data_root).expanduser() / embodiment.source)
    missing = {c.column for c in embodiment.cameras} - set(source.video_keys)
    if missing:
        raise ValueError(f"{embodiment.name}: no such videos {sorted(missing)} in {source.root}")
    action, action_kinds = embodiment.assemble(source, embodiment.action)
    state, _ = embodiment.assemble(source, embodiment.state)

    order = np.random.default_rng(seed).permutation(source.num_episodes)
    n_val = max(1, int(round(val_fraction * source.num_episodes)))
    frames = int(source.lengths.sum())
    manifest = {
        "version": VERSION,
        "fingerprint": fingerprint(source, embodiment, val_fraction, seed),
        "embodiment": embodiment.name,
        "source": embodiment.source,
        "fps": source.fps,
        "episodes": source.num_episodes,
        "frames": frames,
        "hours": frames / source.fps / 3600,
        "action_dim": action.shape[1],
        "action_kinds": action_kinds,
        "state_dim": state.shape[1],
        "cameras": [{"name": c.name, "role": c.role} for c in embodiment.cameras],
        "val_fraction": val_fraction,
        "seed": seed,
        "val_episodes": sorted(order[:n_val].tolist()),
        "stats": {"action": _stats(action), "state": _stats(state)},
    }
    path = manifest_path(data_root, embodiment)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))
    return manifest


def load_manifest(data_root, embodiment, source=None):
    """Read a manifest. Given the open `source`, also refuse one built from a different
    embodiment file or dataset than the ones in hand."""
    path = manifest_path(data_root, embodiment)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `python -m thesis.datasets.mixture.manifest`")
    manifest = json.loads(path.read_text())
    if source is not None:
        expected = fingerprint(source, embodiment, manifest["val_fraction"], manifest["seed"])
        if manifest["fingerprint"] != expected:
            raise ValueError(f"{path} is stale: rebuild it with `python -m thesis.datasets.mixture.manifest`")
    return manifest


def main():
    from omegaconf import OmegaConf

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    for name in cfg.mixture.members:
        emb = load_embodiment(Path(cfg.embodiments_dir) / f"{name}.yaml")
        m = build_manifest(cfg.data_root, emb, cfg.val.fraction, cfg.seed)
        print(f"{m['embodiment']:32s} {m['fps']:4.0f} fps  {m['episodes']:5d} episodes  "
              f"{m['hours']:6.2f} h  action {m['action_dim']}  state {m['state_dim']}  "
              f"cameras {[c['name'] for c in m['cameras']]}")


if __name__ == "__main__":
    main()
