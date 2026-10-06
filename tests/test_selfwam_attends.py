"""Per-stream `attends:` visibility and the SelfWAM clean-action-conditioning pattern:
futures read a clean copy of the executed chunk, the action stream provably cannot --
directly or through any multi-layer path."""

import pytest
import torch
from omegaconf import OmegaConf

from thesis.algorithms import build_algorithm
from thesis.algorithms.masking import (assert_no_attention_path,
                                      build_stream_visibility)

from test_generic_vit_predictor import ENCODER, PROJ, ROPE, randomize


def selfwam_cfg():
    """Minimal SelfWAM shape: video context + clean action copy conditioning; flow
    action chunk + flow future-video latent, asymmetric visibility."""
    return OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": ROPE, "model_dim": 32, "depth": 2, "num_heads": 4,
                  "dim_head": 16, "mlp_ratio": 2.0,
                  "attention": {"no_path": [["action_clean", "action_chunk"]]}},
        "encoders": {"vit": dict(ENCODER), "vit_proj": dict(PROJ)},
        "conditioning": {
            "video_context": {
                "from": "observation.images.pixels", "encoder": "vit_proj",
                "via": "tokens", "dim": 32, "index": "-2..0", "grid": [1, 1],
                "attends": ["video_context"],
            },
            "action_clean": {
                "from": "action", "via": "tokens", "dim": 2, "index": "0..3",
                "fps": 1.0, "attends": ["video_context"],
            },
        },
        "predict": {
            "action_chunk": {"from": "action", "type": "flow", "dim": 2,
                             "index": "0..3", "fps": 1.0,
                             "attends": ["video_context"]},
            "future_video": {
                "from": "observation.images.pixels", "encoder": "vit_proj",
                "type": "flow", "dim": 32, "index": "1..1", "grid": [1, 1],
                "attends": ["video_context", "action_clean"],
            },
        },
        "losses": [
            {"type": "flow", "stream": "action_chunk", "weight": 1.0},
            {"type": "flow", "stream": "future_video", "weight": 1.0},
        ],
    })


def make_batch(b=2):
    torch.manual_seed(0)
    return {
        "observation.images.pixels": (torch.rand(b, 4, 3, 96, 96) * 255).to(torch.uint8),
        "action": torch.randn(b, 4, 2),
    }


def test_mask_matches_declared_visibility():
    trunk = build_algorithm(selfwam_cfg()).predictor
    mask, sl = trunk._mask(), trunk.stream_slices
    sees = lambda q, k: bool(mask[sl[q], sl[k]].any())
    assert sees("future_video", "action_clean")
    assert sees("future_video", "video_context")
    assert not sees("future_video", "action_chunk")
    assert not sees("action_chunk", "action_clean")
    assert not sees("action_chunk", "future_video")
    assert sees("action_chunk", "video_context")
    assert not sees("video_context", "action_clean")
    assert not sees("video_context", "future_video")
    # a stream always attends itself
    assert sees("action_chunk", "action_chunk") and sees("action_clean", "action_clean")


def test_no_path_rejects_transitive_leak():
    """Context reading the clean copy reopens the two-hop route clean -> context ->
    policy; the build must refuse, not silently train a leaky policy."""
    cfg = selfwam_cfg()
    cfg.conditioning.video_context.attends = ["video_context", "action_clean"]
    with pytest.raises(ValueError, match="no_path violated"):
        build_algorithm(cfg)


def test_action_stream_is_blind_to_the_clean_copy():
    """Empirical leak test, independent of how the mask was built: change the clean
    chunk's tokens, the action stream's velocity must not move; the future stream must."""
    algo = build_algorithm(selfwam_cfg())
    trunk = algo.predictor
    randomize(trunk)  # heads are zero-init, which would make both checks pass vacuously
    torch.manual_seed(1)
    tokens = {"video_context": torch.randn(2, 3, 32),
              "action_clean": torch.randn(2, 4, 2),
              "action_chunk": torch.randn(2, 4, 2),
              "future_video": torch.randn(2, 1, 32)}
    t = torch.rand(2)
    with torch.no_grad():
        a = trunk(dict(tokens), t)
        tokens["action_clean"] = torch.randn(2, 4, 2)
        b = trunk(dict(tokens), t)
    assert torch.equal(a["action_chunk"], b["action_chunk"])
    assert not torch.equal(a["future_video"], b["future_video"])


def test_selfwam_trains_and_integrates():
    algo = build_algorithm(selfwam_cfg())
    out = algo.loss(make_batch())
    assert {"loss/action_chunk", "loss/future_video"} <= set(out)
    out["loss"].backward()
    pred = algo.predict(make_batch())
    assert pred["action_chunk"].shape == (2, 4, 2)
    assert pred["future_video"].shape == (2, 1, 32)


def test_attends_rejects_unknown_stream_and_non_token_streams():
    cfg = selfwam_cfg()
    cfg.predict.future_video.attends = ["video_context", "nonexistent"]
    with pytest.raises(ValueError, match="not a token stream"):
        build_algorithm(cfg)
    cfg = selfwam_cfg()
    cfg.conditioning.action_cond = {"from": "action", "via": "cond", "dim": 2,
                                    "index": "-2..0", "attends": ["video_context"]}
    with pytest.raises(AssertionError, match="not a token stream"):
        build_algorithm(cfg)


def test_visibility_helpers_standalone():
    order = ["ctx", "clean", "pol", "fut"]
    vis = build_stream_visibility(
        order, ["ctx", "clean"], ["pol", "fut"],
        {"ctx": ["ctx"], "clean": ["ctx"], "pol": ["ctx"], "fut": ["ctx", "clean"]})
    assert_no_attention_path(order, vis, [["clean", "pol"]])
    vis_bad = build_stream_visibility(
        order, ["ctx", "clean"], ["pol", "fut"],
        {"ctx": ["ctx", "clean"], "clean": ["ctx"], "pol": ["ctx"], "fut": ["ctx", "clean"]})
    with pytest.raises(ValueError, match="clean -> ctx -> pol"):
        assert_no_attention_path(order, vis_bad, [["clean", "pol"]])


def test_no_path_walks_arbitrarily_deep_chains():
    order = ["s0", "s1", "s2", "s3", "s4"]
    chain = {"s0": ["s0"], "s1": ["s0"], "s2": ["s1"], "s3": ["s2"], "s4": ["s3"]}
    vis = build_stream_visibility(order, order[:2], order[2:], chain)
    with pytest.raises(ValueError, match="s0 -> s1 -> s2 -> s3 -> s4"):
        assert_no_attention_path(order, vis, [["s0", "s4"]])
    cut = dict(chain, s2=["s2"])  # break the middle link -> clean
    assert_no_attention_path(order, build_stream_visibility(order, order[:2], order[2:], cut),
                             [["s0", "s4"]])


def test_no_path_rejects_unknown_stream_names():
    cfg = selfwam_cfg()
    cfg.model.attention.no_path = [["action_clean", "typo_stream"]]
    with pytest.raises(ValueError, match="unknown stream 'typo_stream'"):
        build_algorithm(cfg)


def test_policy_only_predict_needs_no_clean_action_rows():
    """streams=['action_chunk'] must not touch the future action rows at all: feed NaN
    there and demand a finite, bit-identical action vs the full pass (same x0)."""
    algo = build_algorithm(selfwam_cfg())
    randomize(algo.predictor)
    obs = make_batch()
    x0 = {"action_chunk": torch.randn(2, 4, 2), "future_video": torch.randn(2, 1, 32)}

    clean, sources, cond, _ = algo._condition(obs)
    full = algo.predictor.rollout(clean, sources, cond=cond, num_steps=2, x0=dict(x0))

    poisoned = {k: v.clone() for k, v in obs.items()}
    poisoned["action"] = poisoned["action"].clone()
    poisoned["action"][:] = float("nan")  # rows 0..3 are exactly the clean-copy window
    # predict() draws x0 internally; go through the trunk with pinned x0 for bit-equality
    algo.predict(poisoned, streams=["action_chunk"])  # the public path must also run on NaN rows
    needed = algo.predictor.required_streams(["action_chunk"])
    clean_p, sources_p, cond_p, _ = algo._condition(poisoned, token_streams=set(needed))
    sub = algo.predictor.rollout(clean_p, sources_p, cond=cond_p, num_steps=2,
                                 x0={"action_chunk": x0["action_chunk"]},
                                 streams=["action_chunk"])
    assert set(sub) == {"action_chunk"}
    assert torch.isfinite(sub["action_chunk"]).all()
    assert torch.equal(sub["action_chunk"], full["action_chunk"])


def test_futures_only_integration_matches_full_pass():
    algo = build_algorithm(selfwam_cfg())
    randomize(algo.predictor)
    obs = make_batch()
    x0 = {"action_chunk": torch.randn(2, 4, 2), "future_video": torch.randn(2, 1, 32)}
    clean, sources, cond, _ = algo._condition(obs)
    full = algo.predictor.rollout(clean, sources, cond=cond, num_steps=2, x0=dict(x0))
    needed = algo.predictor.required_streams(["future_video"])
    assert "action_clean" in needed  # candidate actions ride in through the clean slot
    clean_s = {k: v for k, v in clean.items() if k in needed}
    sub = algo.predictor.rollout(clean_s, sources, cond=cond, num_steps=2,
                                 x0={"future_video": x0["future_video"]},
                                 streams=["future_video"])
    assert set(sub) == {"future_video"}
    assert torch.equal(sub["future_video"], full["future_video"])


def test_subset_refused_when_outputs_would_change():
    cfg = selfwam_cfg()
    cfg.predict.future_video.attends = ["video_context", "action_clean", "action_chunk"]
    algo = build_algorithm(cfg)
    clean, sources, cond, _ = algo._condition(make_batch())
    with pytest.raises(ValueError, match="cannot integrate"):
        algo.predictor.rollout(clean, sources, cond=cond, num_steps=2,
                               streams=["future_video"])


def test_inference_streams_config_default():
    cfg = selfwam_cfg()
    cfg.inference_streams = ["action_chunk"]
    algo = build_algorithm(cfg)
    out = algo.predict(make_batch())
    assert set(out) == {"action_chunk"}
    cfg.inference_streams = ["typo"]
    with pytest.raises(AssertionError, match="not predict streams"):
        build_algorithm(cfg)


def test_no_float64_buffers_anywhere():
    """MPS cannot hold float64: serve.py moves the model with .to(device), which
    converts every buffer -- a float64 buffer breaks Mac serving (regression:
    time_start/time_end, 2026-08-29). The flex-only block_start/block_end buffers are
    exempt because the flex backend never resolves on MPS."""
    algo = build_algorithm(selfwam_cfg())
    bad = [n for n, b in algo.named_buffers()
           if b.dtype == torch.float64 and not n.endswith(("block_start", "block_end"))]
    assert not bad, bad


def test_short_clean_window_fails_loudly_not_with_shape_error():
    """A serve-time batch has no future action rows; a full-stream predict must then
    fail with the named stream and remedy, not a downstream SDPA shape mismatch
    (regression: 83-vs-119 mask error on the Mac server, 2026-08-29)."""
    algo = build_algorithm(selfwam_cfg())
    obs = make_batch()
    obs["action"] = obs["action"][:, :2]  # history-ish only; clean window rows missing
    with pytest.raises(Exception, match="action_clean|missing rows|tokens"):
        algo.predict(obs)


def test_serve_model_overrides_merge_and_scope():
    from omegaconf import OmegaConf

    from thesis.scripts.serve import merge_model_overrides

    stored = OmegaConf.create({"algorithm": {"name": "x", "num_flow_steps": 4},
                               "dataset": {"root": "/d"}})
    out = merge_model_overrides(stored, ["+algorithm.inference_streams=[action]",
                                         "algorithm.num_flow_steps=10"])
    assert out.algorithm.inference_streams == ["action"]
    assert out.algorithm.num_flow_steps == 10
    assert merge_model_overrides(stored, []) is stored
    with pytest.raises(ValueError, match="algorithm"):
        merge_model_overrides(stored, ["dataset.root=/evil"])


def mse_cfg():
    cfg = selfwam_cfg()
    cfg.predict.future_video.type = "mse"
    cfg.losses = [
        {"type": "flow", "stream": "action_chunk", "weight": 1.0},
        {"type": "prediction", "stream": "future_video", "weight": 1.0},
    ]
    return cfg


def test_mse_stream_trains_and_predicts():
    """`type: mse`: learned query tokens in, absolute regression out, no noise."""
    algo = build_algorithm(mse_cfg())
    out = algo.loss(make_batch())
    assert "loss/future_video" in out
    out["loss"].backward()
    assert algo.predictor.query_embed["future_video"].grad is not None
    pred = algo.predict(make_batch())
    assert pred["future_video"].shape == (2, 1, 32)
    assert pred["action_chunk"].shape == (2, 4, 2)
    # deterministic: no noise anywhere in the mse path
    a = algo.predict(make_batch())["future_video"]
    b = algo.predict(make_batch())["future_video"]
    assert torch.equal(a, b)


def test_mse_stream_respects_policy_only_subset():
    algo = build_algorithm(mse_cfg())
    out = algo.predict(make_batch(), streams=["action_chunk"])
    assert set(out) == {"action_chunk"}
