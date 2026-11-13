"""RoboJEPA's world model in the generic trunk (configs/algorithm/robojepa_levjepa_cls.yaml)
and the mixture backend that feeds it: the local attention window, the view axis, padded
camera slots, identity conditioning, the per-embodiment projection, the rollout loss, and
the clip sampler on synthetic LeRobot v3.0 datasets."""

import json
import math
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf

from thesis.algorithms.encoders import build_encoder
from thesis.algorithms.generic_vit_predictor import GenericViTPredictor, GenericViTTrunk
from thesis.algorithms.masking import block_causal_mask, block_intervals
from thesis.algorithms.rope import make_pos_ids

FPS = 5.0
T_IN = 6


def test_window_keeps_the_last_w_steps():
    idx = torch.arange(10)
    start, end = block_intervals(idx, FPS)
    mask = block_causal_mask(start, end, window=3 / FPS)
    for q in range(10):
        assert mask[q].nonzero().squeeze(1).tolist() == list(range(max(0, q - 2), q + 1))


def test_pos_ids_carry_the_view_slot():
    pos = make_pos_ids(torch.tensor([0.0, 0.5]), (1, 1), views=2)
    assert pos["time"].tolist() == [0.0, 0.0, 0.5, 0.5]
    assert pos["view"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert "view" not in make_pos_ids(torch.tensor([0.0]), (2, 2))


def spec(window=None, views=2):
    dim = 8
    conditioning = {
        "frames": {"dim": dim, "views": views, "index": f"-{T_IN - 1}..0", "fps": FPS,
                   "token_in": {"type": "linear"}},
        "action": {"dim": 16, "index": f"-{T_IN - 1}..0", "fps": FPS,
                   "token_in": {"type": "identity"}},
    }
    predict = {
        "frames_next": {"type": "direct", "input": "frames", "dim": dim, "views": views,
                        "index": f"-{T_IN - 2}..1", "fps": FPS,
                        "token_out": {"type": "linear", "norm": "layernorm"}},
    }
    model = dict(model_dim=16, depth=2, num_heads=2, block="prenorm", qk_norm="rmsnorm",
                 final_norm="rmsnorm", init="robojepa", causal=True, attn_backend="sdpa",
                 attention_window=window,
                 embodiment={"field": "embodiment_id", "num": 4, "streams": ["frames"]},
                 camera={"field": "view_id", "num": 7, "streams": ["frames"],
                         "input_embedding": True, "attention_bias": True},
                 rope={"time": {"share": 0.5, "period": "auto"},
                       "view": {"share": 0.5, "min_period": 6.0, "max_period": 60.0}})
    return conditioning, predict, model


def trunk_inputs(B=3, views=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens = {"frames": torch.randn(B, T_IN * views, 8, generator=g),
              "action": torch.randn(B, T_IN, 16, generator=g)}
    meta = {"embodiment_id": torch.tensor([0, 1, 2])[:B],
            "view_id": torch.tensor([[4, 2], [4, 2], [4, 0]])[:B]}
    return tokens, meta


def make_trunk(**kw):
    conditioning, predict, model = spec(**kw)
    trunk = GenericViTTrunk(conditioning, predict, **model)
    # the zero-init identity tables would hide every effect the tests look for
    with torch.no_grad():
        trunk.embodiment_embed.weight.normal_()
        trunk.camera_embed.weight.normal_()
        for blk in trunk.blocks:
            blk.view_bias.weight.normal_()
    return trunk.eval()


def test_prenorm_trunk_is_step_causal():
    trunk = make_trunk()
    tokens, meta = trunk_inputs()
    out = trunk(tokens, meta=meta)["frames_next"]
    assert out.shape == (3, T_IN * 2, 8)
    changed = {**tokens, "frames": tokens["frames"].clone()}
    changed["frames"][:, -2:] += 5.0                   # the last step's two views
    out2 = trunk(changed, meta=meta)["frames_next"]
    torch.testing.assert_close(out[:, :-2], out2[:, :-2])
    assert not torch.allclose(out[:, -2:], out2[:, -2:])


def test_attention_window_forgets_old_steps():
    trunk = make_trunk(window=2 / FPS)
    tokens, meta = trunk_inputs()
    out = trunk(tokens, meta=meta)["frames_next"]
    changed = {**tokens, "frames": tokens["frames"].clone()}
    changed["frames"][:, :2] += 5.0                    # step 0
    out2 = trunk(changed, meta=meta)["frames_next"]
    # depth 2 x window 2: step 0 reaches steps 0..2, never step 3 or later
    torch.testing.assert_close(out[:, 3 * 2:], out2[:, 3 * 2:])
    assert not torch.allclose(out[:, :2], out2[:, :2])


def test_padded_view_slot_is_nobodys_key():
    trunk = make_trunk()
    tokens, meta = trunk_inputs()
    valid = torch.ones(3, T_IN, 2, dtype=torch.bool)
    valid[2, :, 1] = False                             # sample 2 has one camera
    v = {"frames": valid.flatten(1)}
    out = trunk(tokens, meta=meta, valid=v)["frames_next"]
    changed = {**tokens, "frames": tokens["frames"].clone()}
    changed["frames"][2, 1::2] = 100.0                 # sample 2's empty slot
    out2 = trunk(changed, meta=meta, valid=v)["frames_next"]
    torch.testing.assert_close(out[2, 0::2], out2[2, 0::2])
    torch.testing.assert_close(out[:2], out2[:2])


def test_embodiment_and_camera_ids_condition_the_prediction():
    trunk = make_trunk()
    tokens, meta = trunk_inputs()
    out = trunk(tokens, meta=meta)["frames_next"]
    swapped = {**meta, "embodiment_id": torch.tensor([3, 1, 2])}
    assert not torch.allclose(trunk(tokens, meta=swapped)["frames_next"][0], out[0])
    cams = {**meta, "view_id": torch.tensor([[1, 2], [4, 2], [4, 0]])}
    assert not torch.allclose(trunk(tokens, meta=cams)["frames_next"][0], out[0])


def test_robojepa_init_rescales_residual_outputs():
    trunk = GenericViTTrunk(*spec()[:2], **spec()[2])
    std = [blk.mlp[-1].weight.std().item() for blk in trunk.blocks]
    assert std[1] < std[0]
    assert torch.all(trunk.embodiment_embed.weight == 0)


def test_embodiment_linear_picks_per_sample_weights():
    enc = build_encoder({"type": "embodiment_linear", "in_dim": 4, "out_dim": 3, "num": 2})
    x = torch.randn(2, 5, 4)
    out = enc(x, embodiment_id=torch.tensor([0, 1]))
    torch.testing.assert_close(out[0], x[0] @ enc.weight[0] + enc.bias[0])
    torch.testing.assert_close(out[1], x[1] @ enc.weight[1] + enc.bias[1])
    with pytest.raises(ValueError, match="out of range"):
        enc(x, embodiment_id=torch.tensor([0, 2]))


def algorithm_cfg(steps_eval=4):
    conditioning, predict, model = spec()
    cfg = {
        "name": "generic_vit_predictor",
        "encoders": {
            "pix": {"type": "vit", "in_channels": 3, "dim": 8, "patch_size": 8, "depth": 1,
                    "num_heads": 2, "img_size": 16, "pooling": "cls"},
            "vision": {"type": "layernorm", "input": "pix"},
            "act": {"type": "embodiment_linear", "in_dim": 4, "out_dim": 16, "num": 4},
        },
        "conditioning": {
            "frames": {**conditioning["frames"], "from": "observation.images",
                       "encoder": "vision", "valid": {"steps": "step_valid", "views": "view_valid"}},
            "action": {**conditioning["action"], "from": "action", "encoder": "act"},
        },
        "predict": {
            "frames_next": {**predict["frames_next"], "from": "observation.images",
                            "encoder": "vision",
                            "valid": {"steps": "step_valid", "views": "view_valid"}},
        },
        "model": model,
        "optim_groups": {"no_decay": ["encoders.act.*"]},
        "losses": [
            {"type": "prediction", "stream": "frames_next", "weight": 0.5, "norm": "l1",
             "reduce": "step"},
            {"type": "rollout", "stream": "frames_next", "steps": 2, "prefix": "random",
             "weight": 0.5, "label": "rollout"},
            {"type": "rollout", "stream": "frames_next", "steps": steps_eval, "prefix": "all",
             "weight": 0.0, "label": "rollout_eval"},
        ],
    }
    return OmegaConf.create(cfg)


def batch(B=3, short=None):
    T = T_IN + 1
    images = torch.randint(0, 255, (B, T, 2, 3, 16, 16), dtype=torch.uint8)
    step_valid = torch.ones(B, T, dtype=torch.bool)
    if short is not None:
        step_valid[0, short:] = False
    view_valid = torch.tensor([[True, True], [True, False], [True, True]])[:B]
    return {"observation.images": images, "action": torch.randn(B, T_IN, 4),
            "step_valid": step_valid, "view_valid": view_valid,
            "view_id": torch.tensor([[4, 2], [4, 0], [3, 2]])[:B],
            "embodiment_id": torch.tensor([0, 1, 3])[:B]}


def test_model_trains_and_reports_rollout_steps():
    pytest.importorskip("rotary_embedding_torch")
    model = GenericViTPredictor(algorithm_cfg())
    groups = model.optim_params()
    names = {id(p): n for n, p in model.named_parameters()}
    assert all(not names[id(p)].startswith("encoders.act") for p in groups[0]["params"])
    model.train()
    out = model.loss(batch())
    assert set(out) >= {"loss", "loss/frames_next", "loss/rollout", "loss/rollout/step2",
                        "var/frames/temporal"}
    assert "loss/rollout_eval" not in out             # observed terms skip training
    out["loss"].backward()
    assert model.encoders["act"].weight.grad is not None

    model.eval()
    with torch.no_grad():
        out = model.loss(batch(short=4))
    assert {f"loss/rollout_eval/step{k}" for k in range(1, 5)} <= set(out)
    assert all(math.isfinite(v.item()) for v in out.values())


def test_rollout_feeds_back_its_own_predictions():
    """The eval rollout's first autoregressive pass reads the teacher-forced prediction of
    step 1 in input position 1, where the true latent was."""
    model = GenericViTPredictor(algorithm_cfg())
    model.eval()
    b = batch()
    calls = []
    real = model.predictor.forward

    def spy(tokens, *a, **k):
        calls.append(tokens["frames"].clone())
        return real(tokens, *a, **k)

    model.predictor.forward = spy
    with torch.no_grad():
        ctx = model._build_context(b)
    tf = ctx["pred"]["frames_next"].view(3, T_IN, 2, -1)
    # the 4-step eval rollout over prefix 0 writes tf[:, 0] into input step 1
    fed = calls[2].view(3, T_IN, 2, -1)
    torch.testing.assert_close(fed[:, 1], tf[:, 0])


SIGREG = {"type": "sigreg", "streams": ["frames", "frames_next"], "concat": "temporal",
          "statistic": "per_timestep", "head": "mlp", "proj_hidden": 32, "weight": 0.09}


def test_each_real_frame_is_encoded_once():
    """The context window and its shift-by-one target share one encoder pass, which skips
    padded steps and empty camera slots."""
    model = GenericViTPredictor(algorithm_cfg())
    model.train()
    seen = []
    model.encoders["pix"].register_forward_hook(lambda m, i, o: seen.append(i[0].shape))
    model.loss(batch(short=4))
    # 4 steps x 2 cams + 7 steps x 1 cam + 7 steps x 2 cams, one frame per clip
    assert seen == [torch.Size([4 * 2 + 7 + 7 * 2, 1, 3, 16, 16])]


def test_shared_encoding_matches_encoding_each_stream():
    model = GenericViTPredictor(algorithm_cfg())
    model.eval()
    b = batch(short=4)
    with torch.no_grad():
        shared = model._build_context(b)
        model._frame_local = lambda enc: False
        apart = model._build_context(b)
    for name in ("frames", "frames_next"):
        keep = shared["latent_valid"][name]
        torch.testing.assert_close(shared["latent"][name][keep], apart["latent"][name][keep])
        assert not shared["latent"][name][~keep].any()


def test_sigreg_fork_trains_without_touching_the_targets():
    cfg = algorithm_cfg()
    cfg.losses.append(SIGREG)
    model = GenericViTPredictor(cfg)
    term = model.loss_terms[-1]
    assert isinstance(term.head[1], torch.nn.BatchNorm1d)
    model.train()
    out = model.loss(batch(short=4))
    assert "loss/sigreg_frames_frames_next_per_timestep_temporal" in out
    out["loss"].backward()
    assert term.head[0].weight.grad is not None
    assert model.encoders["pix"].cls_token.grad is not None


def test_sigreg_never_sees_padded_tokens():
    from thesis.algorithms.losses import SIGRegLoss
    term = SIGRegLoss(["a"], 1.0, stream_dims={"a": 8}, stream_frames={"a": 5},
                      statistic="per_timestep", head="mlp", proj_hidden=16)
    term.train()
    z = torch.randn(6, 5 * 2, 8)
    valid = torch.ones(6, 5, 2, dtype=torch.bool)
    valid[0, 3:] = False
    valid[2, :, 1] = False
    valid = valid.flatten(1)
    junk = z.clone()
    junk[~valid] = 1e3

    def score(latent):
        torch.manual_seed(0)
        return term({"latent": {"a": latent}, "latent_valid": {"a": valid}})

    torch.testing.assert_close(score(z), score(junk))


# ---- the mixture backend ----------------------------------------------------------------

SIZE = 24


def write_dataset(root, fps, lengths, state, cameras):
    """A LeRobot v3.0 dataset with a given (total, S) state array and constant-color video
    whose brightness encodes the global frame index."""
    total = sum(lengths)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    rows = np.arange(total)
    pq.write_table(pa.table({
        "action": pa.array(np.zeros((total, 2), np.float32).tolist(), pa.list_(pa.float32())),
        "observation.state": pa.array(state.astype(np.float32).tolist(), pa.list_(pa.float32())),
        "index": rows,
    }), root / "data" / "chunk-000" / "file-000.parquet")
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    episodes = {"episode_index": np.arange(len(lengths)), "length": np.array(lengths),
                "dataset_from_index": starts, "dataset_to_index": starts + np.array(lengths)}
    features = {"action": {"dtype": "float32", "shape": [2]},
                "observation.state": {"dtype": "float32", "shape": [state.shape[1]]}}
    for cam in cameras:
        key = f"observation.images.{cam}"
        features[key] = {"dtype": "video", "shape": [SIZE, SIZE, 3]}
        episodes[f"videos/{key}/chunk_index"] = np.zeros(len(lengths), np.int64)
        episodes[f"videos/{key}/file_index"] = np.zeros(len(lengths), np.int64)
        episodes[f"videos/{key}/from_timestamp"] = starts / fps
        episodes[f"videos/{key}/to_timestamp"] = (starts + np.array(lengths)) / fps
        path = root / "videos" / key / "chunk-000" / "file-000.mp4"
        path.parent.mkdir(parents=True)
        frames = np.stack([np.full((SIZE, SIZE, 3), 20 + (7 * i) % 210, np.uint8)
                           for i in range(total)])
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
    pytest.importorskip("torchcodec")
    from thesis.datasets.mixture.embodiment import load_embodiment
    from thesis.datasets.mixture.manifest import build_manifest

    root = tmp_path_factory.mktemp("mixture")
    lengths = [60, 80, 12, 70]
    t = np.arange(sum(lengths)) / 10.0
    pose = np.stack([t, 2 * t, 0 * t, 0.3 * t, 0 * t, 0 * t, np.sin(t)], 1)  # xyz, euler, grip
    write_dataset(root / "data" / "arm", 10, lengths, pose, ["front", "wrist"])
    joints = np.stack([t * (d + 1) for d in range(4)], 1)
    write_dataset(root / "data" / "slow", 5, [40, 50], joints[:90], ["front"])
    emb = root / "embodiments"
    emb.mkdir()
    (emb / "arm.yaml").write_text(
        "name: arm\nsource: arm\naction:\n  - {from: action}\n"
        "cameras:\n  - {name: front, from: observation.images.front, role: exterior}\n"
        "  - {name: wrist, from: observation.images.wrist, role: wrist, view: wrist}\n"
        "proprio:\n  from: observation.state\n  position: [0, 1, 2]\n  rotation: [3, 4, 5]\n"
        "  gripper: [6]\n")
    (emb / "slow.yaml").write_text(
        "name: slow\nsource: slow\naction:\n  - {from: action}\n"
        "cameras:\n  - {name: front, from: observation.images.front, role: exterior, view: front}\n"
        "proprio:\n  from: observation.state\n  other: [0, 1, 2, 3]\n")
    for name in ("arm", "slow"):
        build_manifest(root / "data", load_embodiment(emb / f"{name}.yaml"), val_fraction=0.25)
    return root, emb


def source_cfg(root, emb, **kw):
    return OmegaConf.create({
        "backend": "mixture", "data_root": str(root / "data"), "embodiments_dir": str(emb),
        "members": ["arm", "slow"], "fps": 5, "frames_per_clip": 8, "pad_size": 16,
        "view_groups": {1: 0.5, 2: 0.5}, "image_size": [12, 16], "seed": 0,
        "val_windows_per_member": 4, "epoch_size": 50,
        "conditioning": {"frames": {"from": "observation.images", "index": "-6..0"},
                         "action": {"from": "action", "index": "-6..0"}},
        "predict": {"frames_next": {"from": "observation.images", "index": "-5..1"}},
        **kw})


def test_clips_have_the_contract_shapes(mixture):
    from thesis.datasets import build_dataset

    src = build_dataset(source_cfg(*mixture), split="train")
    for i in range(20):
        item = src[i]
        assert item["observation.images"].shape == (8, 2, 3, 12, 16)
        assert item["action"].shape == (7, 16) and item["observation.state"].shape == (8, 16)
        n = int(item["step_valid"].sum())
        assert n >= 4
        assert torch.all(item["observation.images"][n:] == 0)
        if not item["view_valid"][1]:
            assert torch.all(item["observation.images"][:, 1] == 0)
        if int(item["embodiment_id"]) == 1:          # one camera: never two views
            assert not item["view_valid"][1]


def test_state_delta_composes_rotations(mixture):
    from thesis.datasets.mixture.clips import state_delta
    from thesis.datasets.mixture.embodiment import load_embodiment

    proprio = load_embodiment(mixture[1] / "arm.yaml").proprio
    s = np.array([[0, 0, 0, 3.1, 0, 0, 0.0], [1, 2, 3, -3.1, 0, 0, 0.5]], np.float32)
    a = state_delta(s, proprio)
    np.testing.assert_allclose(a[0, :3], [1, 2, 3])
    # 3.1 -> -3.1 about x is a 0.083 rad turn through pi, not -6.2
    np.testing.assert_allclose(a[0, 3], 2 * math.pi - 6.2, atol=1e-5)
    np.testing.assert_allclose(a[0, 6], 0.5)


def test_actions_and_frames_follow_the_stride(mixture):
    from thesis.datasets.mixture.clips import MixtureSource

    src = MixtureSource(source_cfg(*mixture), split="train")
    arm = src.members[0]
    assert arm.stride == 2
    frame_idx, states = arm.window(1, 3, 8)
    np.testing.assert_array_equal(frame_idx, 3 + 2 * np.arange(8))
    item = src._clip(arm, 1, 3, [0, 1], None)
    row = int(arm.source.first_row[1])
    shown = item["observation.images"][:, 0].float().mean((1, 2, 3)).numpy()
    np.testing.assert_allclose(shown, 20 + (7 * (row + frame_idx)) % 210, atol=3)
    # x moves 0.1 per native frame, so 0.2 per step; the delta of step t ends at state t + 1
    np.testing.assert_allclose(item["action"][:, 0].numpy(), 0.2, atol=1e-5)
    np.testing.assert_allclose(item["observation.state"][1, 0] - item["observation.state"][0, 0],
                               item["action"][0, 0], atol=1e-5)
    assert item["view_id"].tolist() == [4, 2]


def test_held_out_clips_are_fixed_and_disjoint(mixture):
    from thesis.datasets.mixture.clips import MixtureSource

    val = MixtureSource(source_cfg(*mixture), split="val")
    train = MixtureSource(source_cfg(*mixture), split="train")
    assert len(val) == 8
    a, b = val[3], val[3]
    torch.testing.assert_close(a["observation.images"], b["observation.images"])
    for tm, vm in zip(train.members, val.members):
        assert not set(tm.episodes) & set(vm.episodes)


def test_spec_windows_must_share_the_clip_origin(mixture):
    from thesis.datasets.mixture.clips import MixtureSource

    cfg = source_cfg(*mixture)
    cfg.conditioning.action.index = "-5..0"
    with pytest.raises(ValueError, match="must start there"):
        MixtureSource(cfg, split="train")


def test_unfrozen_levjepa_checkpoints_its_blocks():
    enc = build_encoder({"type": "levjepa", "arch": "vit_tiny", "crop_size": [32, 48],
                         "repeat": 1, "attn_mode": "full", "frozen": False})
    frames = torch.randint(0, 255, (2, 3, 3, 32, 48), dtype=torch.uint8)
    enc.train()
    ref = enc(frames)
    ref.sum().backward()
    grads = [p.grad.clone() for p in enc.parameters() if p.grad is not None]
    enc.zero_grad()
    enc.gradient_checkpointing = True
    out = enc(frames)
    out.sum().backward()
    assert out.shape == (2, 3, 192)
    torch.testing.assert_close(out, ref)
    for g, p in zip(grads, [p for p in enc.parameters() if p.grad is not None]):
        torch.testing.assert_close(p.grad, g, rtol=1e-4, atol=1e-5)
