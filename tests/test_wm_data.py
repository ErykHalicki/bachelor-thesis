"""The multi-embodiment data layer and world model on two tiny synthetic datasets with
different frame rates, action widths and camera counts. Every frame's brightness encodes
its global frame index, so decoding the wrong frame is visible."""

import json
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("torchcodec")

from thesis.datasets.mixture.batch import collate
from thesis.datasets.mixture.clock import Clock
from thesis.datasets.mixture.embodiment import load_embodiment
from thesis.datasets.mixture.manifest import build_manifest, load_manifest
from thesis.datasets.mixture.stream import (CameraPolicy, FixedWindows, MixtureStream, build_members,
                                mixture_spec, mixture_weights)

SIZE = 32


def brightness(i):
    return 20 + (7 * i) % 210


def write_dataset(root, fps, lengths, action_dim, cameras):
    total = sum(lengths)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    rows = np.arange(total)
    action = np.stack([rows * 10 + d for d in range(action_dim)], 1).astype(np.float32)
    pq.write_table(pa.table({
        "action": pa.array(action.tolist(), pa.list_(pa.float32())),
        "observation.state": pa.array((action[:, :2] + 0.5).tolist(), pa.list_(pa.float32())),
        "index": rows,
    }), root / "data" / "chunk-000" / "file-000.parquet")

    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    episodes = {"episode_index": np.arange(len(lengths)), "length": np.array(lengths),
                "dataset_from_index": starts, "dataset_to_index": starts + np.array(lengths)}
    features = {"action": {"dtype": "float32", "shape": [action_dim]},
                "observation.state": {"dtype": "float32", "shape": [2]}}
    for cam in cameras:
        key = f"observation.images.{cam}"
        features[key] = {"dtype": "video", "shape": [SIZE, SIZE, 3]}
        episodes[f"videos/{key}/chunk_index"] = np.zeros(len(lengths), np.int64)
        episodes[f"videos/{key}/file_index"] = np.zeros(len(lengths), np.int64)
        episodes[f"videos/{key}/from_timestamp"] = starts / fps
        episodes[f"videos/{key}/to_timestamp"] = (starts + np.array(lengths)) / fps
        path = root / "videos" / key / "chunk-000" / "file-000.mp4"
        path.parent.mkdir(parents=True)
        frames = np.stack([np.full((SIZE, SIZE, 3), brightness(i), np.uint8) for i in range(total)])
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{SIZE}x{SIZE}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-g", "2",
             "-bf", "0", "-crf", "10", "-pix_fmt", "yuv420p", str(path)],
            input=frames.tobytes(), check=True)
    pq.write_table(pa.table(episodes), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": fps, "total_frames": total,
        "total_episodes": len(lengths), "features": features,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}))


@pytest.fixture(scope="module")
def mixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("mixture")
    write_dataset(root / "data" / "slow", 5, [20, 30, 25, 5], 3, ["front"])
    write_dataset(root / "data" / "fast", 10, [50, 40, 60], 2, ["front", "wrist"])
    emb = root / "embodiments"
    emb.mkdir()
    (emb / "slow_arm.yaml").write_text(
        "name: slow_arm\nsource: slow\n"
        "action:\n  - {from: action, dims: [0, 1], kind: delta}\n"
        "  - {from: action, dims: [2], kind: discrete}\n"
        "state:\n  - {from: observation.state}\n"
        "cameras:\n  - {name: front, from: observation.images.front, role: exterior}\n")
    (emb / "fast_arm.yaml").write_text(
        "name: fast_arm\nsource: fast\n"
        "action:\n  - {from: action, kind: delta}\n"
        "cameras:\n  - {name: front, from: observation.images.front, role: exterior}\n"
        "  - {name: wrist, from: observation.images.wrist, role: wrist}\n")
    names = ["slow_arm", "fast_arm"]
    for name in names:
        build_manifest(root / "data", load_embodiment(emb / f"{name}.yaml"), val_fraction=0.3)
    clock = Clock(f_wm=2.5, steps=4)
    members = build_members(root / "data", emb, names, clock, "train")
    return root, emb, names, clock, members


def test_window_is_the_right_frames_and_actions(mixture):
    *_, clock, members = mixture
    slow, fast = members
    episode = int(fast.episodes[-1])
    row = int(fast.source.first_row[episode])
    sample = fast.window(episode, start=5, cams=[0, 1])
    assert sample["frames"].shape == (2, 4, 3, SIZE, SIZE)
    shown = row + 5 + np.array([0, 4, 8, 12])
    expected = np.array([brightness(i) for i in shown])
    for view in sample["frames"]:
        np.testing.assert_allclose(view.float().mean((1, 2, 3)).numpy(), expected, atol=2.5)
    assert sample["action"].shape == (3, 4, 2)
    np.testing.assert_array_equal(sample["action"].reshape(12, 2)[:, 0], (row + 5 + np.arange(12)) * 10)
    assert sample["state"].shape == (4, 0)

    sample = slow.window(int(slow.episodes[0]), start=0, cams=[0])
    assert sample["action"].shape == (3, 2, 3) and sample["state"].shape == (4, 2)


def test_short_episodes_and_val_episodes_never_reach_training(mixture):
    root, emb, names, clock, members = mixture
    slow = members[0]
    assert 3 not in slow.episodes and slow.dropped + len(slow.episodes) < 4
    val = build_members(root / "data", emb, names, clock, "val")
    for train_member, val_member in zip(members, val):
        assert not set(train_member.episodes) & set(val_member.episodes)
    for r in (0, slow.num_anchors - 1):
        episode, start = slow.anchor(r)
        assert start + clock.span(slow.fps) < slow.source.lengths[episode]


def test_stale_manifest_is_refused(mixture):
    root, emb, *_ = mixture
    embodiment = load_embodiment(emb / "slow_arm.yaml")
    changed = type(embodiment)(**{**embodiment.__dict__, "text": embodiment.text + "# edited"})
    from thesis.datasets.mixture.source import LeRobotV3Source

    with pytest.raises(ValueError, match="stale"):
        load_manifest(root / "data", changed, LeRobotV3Source(root / "data" / "slow"))


def test_weights_follow_hours_and_temperature(mixture):
    members = mixture[-1]
    hours = np.array([m.manifest["hours"] for m in members])
    np.testing.assert_allclose(mixture_weights(members, 1.0), hours / hours.sum())
    np.testing.assert_allclose(mixture_weights(members, 0.0), [0.5, 0.5])
    boosted = mixture_weights(members, 0.0, {"slow_arm": 3.0})
    np.testing.assert_allclose(boosted, [0.75, 0.25])


def test_stream_is_seeded_and_collates_mixed_views(mixture):
    members = mixture[-1]
    policy = CameraPolicy()

    def draw(seed):
        stream = iter(MixtureStream(members, [0.5, 0.5], policy, seed))
        return collate([next(stream) for _ in range(8)])

    a, b, c = draw(0), draw(0), draw(1)
    assert torch.equal(a.views, b.views) and torch.equal(a.emb_id, b.emb_id)
    assert a.views.shape != c.views.shape or not torch.equal(a.views, c.views)
    assert set(a.emb_id.tolist()) == {0, 1}
    assert a.views.shape[0] == int((a.emb_id + 1).sum())
    assert a.view_slot.tolist() == [s for e in a.emb_id.tolist() for s in range(e + 1)]
    assert a.groups[0]["action"].shape[1:] == (3, 2, 3)
    assert a.groups[1]["action"].shape[1:] == (3, 4, 2)
    assert sorted(torch.cat([g["idx"] for g in a.groups.values()]).tolist()) == list(range(8))


def test_camera_policy(mixture):
    fast = mixture[-1][1].embodiment
    assert CameraPolicy(roles=["wrist"]).select(fast) == [1]
    assert CameraPolicy(max_views=1).select(fast) == [0]
    picks = {tuple(CameraPolicy(max_views=1).select(fast, np.random.default_rng(s))) for s in range(20)}
    assert picks == {(0,), (1,)}
    with pytest.raises(ValueError):
        CameraPolicy(roles=["head"]).select(fast)


MODEL = OmegaConf.create({
    "latent_dim": 16, "proj_hidden": 32, "max_steps": 8, "camera_embedding": "per_embodiment_camera",
    "encoder": {"arch": "vit_tiny", "checkpoint": None, "img_size": SIZE, "repeat": 1,
                "train_blocks": None, "grad_checkpoint": False},
    "predictor": {"dim": 24, "depth": 2, "heads": 2},
    "sigreg": {"weight": 0.09, "num_proj": 64},
})


def test_world_model_trains_and_padding_does_not_leak(mixture):
    from thesis.wm.model import WorldModel

    root, emb, names, clock, members = mixture
    torch.manual_seed(0)
    model = WorldModel(mixture_spec(members), MODEL)
    windows = FixedWindows(members, CameraPolicy(), per_member=3, seed=0)
    samples = [windows[i] for i in range(len(windows))]
    batch = collate(samples)

    assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in model.pred_proj.modules())
    out = model(batch)
    assert out["z"].shape == (9, 4, 16) and out["pred"].shape == (9, 3, 16)
    out["loss"].backward()
    assert all(p.grad is not None for p in model.encoder.parameters() if p.requires_grad)

    model.eval()
    with torch.no_grad():
        together = model(batch)["pred"]
        alone = model(collate(samples[:1]))["pred"]
        rolled = model.rollout(model.encode(batch.views), batch, context=2)
    torch.testing.assert_close(together[:1], alone, atol=1e-5, rtol=1e-4)
    assert rolled.shape == (9, 2, 16)


@pytest.mark.parametrize("strategy,tables", [
    ("per_embodiment_camera", [3]), ("per_role", [2]), ("per_role_plus_offset", [2, 3]),
    ("per_index", [2]), ("none", [])])
def test_camera_embedding_strategies(mixture, strategy, tables):
    from thesis.wm.camera_embedding import CameraEmbedding

    module = CameraEmbedding(mixture_spec(mixture[-1]), 8, strategy)
    assert [t.num_embeddings for t in module.tables] == tables
    emb, cam = torch.tensor([0, 1, 1]), torch.tensor([0, 0, 1])
    out = module(emb, cam)
    assert out.shape == (3, 8)
    if strategy in ("per_role", "per_role_plus_offset", "per_index"):
        torch.testing.assert_close(out[0], out[1])
    if strategy == "per_embodiment_camera":
        assert not torch.allclose(out[0], out[1])
