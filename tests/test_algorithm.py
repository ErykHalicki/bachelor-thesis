from pathlib import Path

import torch
from omegaconf import OmegaConf

from thesis.algorithms import build_algorithm
from thesis.algorithms.losses import build_loss_term

ROPE = {"time": {"share": 0.4, "period": "auto"},
        "seq": {"share": 0.2, "period": "auto"},
        "height": {"share": 0.2, "min_period": 4.0, "max_period": 64.0},
        "width": {"share": 0.2, "min_period": 4.0, "max_period": 64.0}}
GRID_ROPE = {"time": {"share": 0.5, "min_period": 4.0, "max_period": 256.0},
             "height": {"share": 0.25, "period": "auto"},
             "width": {"share": 0.25, "period": "auto"}}

CONFIGS = Path(__file__).parent.parent / "configs"


def make_cfg():
    # camera_tokens plays the role of an already-encoded video field: two entries
    # share it, union window -2..1, 4 spatial tokens per step
    return OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": ROPE, "model_dim": 64, "depth": 2, "num_heads": 4},
        "num_flow_steps": 3,
        "conditioning": {
            "video_context": {
                "from": "camera_tokens", "dim": 16,
                "index": "-2..-1", "fps": 1.0, "grid": [2, 2],
            },
            "state": {"via": "cross", "from": "observation.state", "dim": 4, "index": [0]},
        },
        "predict": {
            "future_video": {
                "from": "camera_tokens", "dim": 16,
                "index": "0..1", "fps": 1.0, "grid": [2, 2],
            },
            "action": {"from": "action", "dim": 4, "index": "0..2", "fps": 3.0},
        },
    })


def make_batch(b=2):
    torch.manual_seed(0)
    return {
        "camera_tokens": torch.randn(b, 4 * 4, 16),
        "observation.state": torch.randn(b, 1, 4),
        "action": torch.randn(b, 3, 4),
    }


def test_loss_trains():
    algo = build_algorithm(make_cfg())
    out = algo.loss(make_batch())
    assert out["loss"].isfinite()
    assert {"loss/future_video", "loss/action"} <= set(out)
    out["loss"].backward()
    head_grads = [p.grad for p in algo.predictor.heads.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in head_grads)


def test_flow_term_weight_scales_total():
    algo = build_algorithm(make_cfg())
    batch = make_batch()
    torch.manual_seed(1)
    plain = algo.loss(batch)
    assert torch.allclose(plain["loss"], plain["loss/future_video"] + plain["loss/action"])

    algo.loss_terms = torch.nn.ModuleList([
        build_loss_term({"type": "flow", "stream": "future_video", "weight": 0.0}, []),
        build_loss_term({"type": "flow", "stream": "action", "weight": 1.0}, []),
    ])
    torch.manual_seed(1)
    weighted = algo.loss(batch)
    assert torch.allclose(weighted["loss"], plain["loss/action"])


def test_shared_field_sliced_per_entry():
    algo = build_algorithm(make_cfg())
    batch = make_batch(1)
    field = batch["camera_tokens"]
    ctx = algo._encode("video_context", algo.conditioning["video_context"], batch)
    fut = algo._encode("future_video", algo.predict_spec["future_video"], batch)
    assert torch.equal(ctx, field[:, :8])
    assert torch.equal(fut, field[:, 8:])


def test_predict_from_past_only_obs():
    algo = build_algorithm(make_cfg())
    batch = make_batch()
    obs = {
        "camera_tokens": batch["camera_tokens"][:, :8],
        "observation.state": batch["observation.state"],
    }
    torch.manual_seed(7)
    out = algo.predict(obs)
    assert out["future_video"].shape == (2, 8, 16)
    assert out["action"].shape == (2, 3, 4)
    torch.manual_seed(7)
    out2 = algo.predict(obs)
    assert all(torch.equal(out[k], out2[k]) for k in out)


def make_wm_cfg():
    # patch 8 on 16x16 frames -> 2x2 grid = 4 patch tokens per frame; head_dim
    # 64/4 = 16 still splits across RopeND's four axes
    return OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": ROPE, "model_dim": 64, "depth": 2, "num_heads": 4},
        "num_flow_steps": 2,
        "encoders": {"vit": {"type": "vit", "rope": GRID_ROPE, "patch_size": 8, "img_size": 16, "frames": 2,
                       "dim": 64, "depth": 1, "num_heads": 4}},
        "conditioning": {
            "pixel_context": {"from": "observation.images.pixels", "encoder": "vit", "dim": 64,
                              "index": "-1..0", "fps": 10, "grid": [2, 2]},
            "action": {"from": "action", "dim": 2, "index": "0", "fps": 10},
        },
        "predict": {
            "future_video": {"from": "observation.images.pixels", "encoder": "vit", "type": "flow",
                             "dim": 64, "index": "1", "fps": 10, "grid": [2, 2]},
        },
        "losses": [
            {"type": "flow", "stream": "future_video", "weight": 1.0},
            {"type": "sigreg", "streams": ["pixel_context", "future_video"], "weight": 0.1},
        ],
    })


def test_wm_pixel_trains():
    algo = build_algorithm(make_wm_cfg())
    batch = {
        "observation.images.pixels": torch.randint(0, 256, (2, 3, 3, 16, 16), dtype=torch.uint8),
        "action": torch.randn(2, 1, 2),
    }
    out = algo.loss(batch)
    assert {"loss/future_video", "loss/sigreg_pixel_context_future_video"} <= set(out)
    assert not hasattr(algo, "chunk_len")
    out["loss"].backward()


def test_rollout_latent_shape_and_autoregression():
    torch.manual_seed(0)
    algo = build_algorithm(make_wm_cfg())
    N = 4
    obs = {"observation.images.pixels": torch.randint(0, 256, (N, 2, 3, 16, 16), dtype=torch.uint8)}
    acts = torch.randn(N, 3, 2)
    final = algo.rollout_latent(obs, acts, stream="future_video")
    assert final.shape == (N, 4, 64)
    assert torch.isfinite(final).all()
    other = algo.rollout_latent(obs, acts + 3.0, stream="future_video")
    assert (final - other).abs().max() > 1e-4


def make_goal_cfg():
    # the episode-final ("last") step enters as a cross source; the union window
    # becomes -2..1 plus the appended final step
    cfg = make_cfg()
    cfg.conditioning.goal = {
        "via": "cross", "from": "camera_tokens", "dim": 16, "index": "last", "grid": [2, 2],
    }
    return cfg


def test_goal_last_slices_final_rows_in_training_and_eval_layouts():
    algo = build_algorithm(make_goal_cfg())
    batch = make_batch()
    batch["camera_tokens"] = torch.randn(2, 5 * 4, 16)
    goal_rows = batch["camera_tokens"][:, -4:]

    assert torch.equal(algo._encode("goal", algo.conditioning["goal"], batch), goal_rows)
    assert torch.equal(
        algo._encode("video_context", algo.conditioning["video_context"], batch),
        batch["camera_tokens"][:, :8],
    )
    assert algo.loss(batch)["loss"].isfinite()

    # eval layout: history + the env goal appended, no future rows -- the same
    # end-relative selector still finds the goal step
    obs = {
        "camera_tokens": torch.cat([batch["camera_tokens"][:, :8], goal_rows], dim=1),
        "observation.state": batch["observation.state"],
    }
    assert torch.equal(algo._encode("goal", algo.conditioning["goal"], obs), goal_rows)
    torch.manual_seed(3)
    pred = algo.predict(obs)
    assert pred["action"].shape == (2, 3, 4)

    # the output heads are zero-initialized (velocity 0 at init), so give them
    # weights before asserting the goal changes the prediction
    for p in algo.predictor.heads.parameters():
        torch.nn.init.normal_(p, std=0.02)
    torch.manual_seed(3)
    pred = algo.predict(obs)
    obs2 = {
        **obs,
        "camera_tokens": torch.cat([obs["camera_tokens"][:, :8], goal_rows + 1.0], dim=1),
    }
    torch.manual_seed(3)
    pred2 = algo.predict(obs2)
    assert (pred["action"] - pred2["action"]).abs().max() > 1e-6
