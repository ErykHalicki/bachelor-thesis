"""The `integration` (inference-consistency) loss: the endpoint of a differentiable N-step
Euler solve, scored against the flow target instead of the velocity at a random t.

The term only means anything if the trajectory it scores is the one eval runs, so the tests
pin down (1) the training solve reproduces `predict()` exactly, (2) it starts from the A2A
seed like eval does, (3) latent- and raw-space scoring reach every part they claim -- the
whole Euler chain into the predictor, the codec decoder, the encoder through both the seed
and the target, with nothing stop-gradded anywhere -- and (4) the config validation.
"""

import pytest
import torch

from thesis.algorithms import build_algorithm

from test_a2a import ae_cfg, make_batch, wam_cfg

INT = {"type": "integration", "stream": "action_chunk", "weight": 1.0, "num_steps": 3}


def int_cfg(**term):
    """The codec-latent action recipe (ae_cfg) plus one integration term."""
    cfg = ae_cfg()
    cfg.losses = [t for t in cfg.losses if t["type"] == "flow"]
    cfg.losses.append({**INT, **term})
    return cfg


def grad_norm(module):
    return sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)


def wake_heads(algo):
    """DiT zero-inits each head's output layer, so at init d(velocity)/d(trunk) is exactly
    zero and NO loss -- flow or integration -- reaches the blocks. Perturb the heads first
    when the test is about where gradients go."""
    for p in algo.predictor.heads.parameters():
        torch.nn.init.normal_(p, std=0.05)


def test_training_solve_reproduces_the_inference_rollout():
    """The whole point of the term: `ctx["integrated"]` is the endpoint `predict()` would
    produce from the same batch, not an approximation of it. The two paths differ in
    no_grad, KV caching and where the decoder is applied -- none of which may move the
    numbers."""
    cfg = int_cfg(num_steps=4)
    cfg.num_flow_steps = 4
    algo = build_algorithm(cfg)
    algo.eval()
    # a training batch: predict() reads the same conditioning windows out of it, and
    # the seed is deterministic, so no randn is drawn on either path
    batch = make_batch()
    ctx = algo._build_context(batch)
    endpoint = algo.encoders["act_dec"](ctx["integrated"]["action_chunk"])
    assert torch.allclose(endpoint, algo.predict(batch)["action_chunk"], atol=1e-5)


def test_integration_starts_from_the_a2a_seed():
    """With num_steps small and the field frozen at v = 0 the endpoint IS the seed -- the
    property that makes an informed x0 worth having. Zeroing the heads' output weights is
    the cheapest way to say `v = 0`."""
    algo = build_algorithm(int_cfg())
    algo.eval()
    for head in algo.predictor.heads.values():
        for p in head.parameters():
            torch.nn.init.zeros_(p)
    batch = make_batch()
    clean, sources, cond, hidden = algo._condition(batch)
    seeds = algo.build_flow_seeds(batch, {**clean, **sources, **cond, **hidden}, training=True)
    out = algo._forward_integrate(clean, sources, cond, seeds, 3)
    assert torch.allclose(out["action_chunk"], seeds["action_chunk"], atol=1e-6)


def test_raw_space_term_trains_predictor_and_decoder():
    """`decoder:` scores the endpoint in executed-action units, so gradients must reach both
    the trunk (through the solve) and the codec decoder (through the decode)."""
    algo = build_algorithm(int_cfg(decoder="act_dec", norm="l1"))
    wake_heads(algo)
    out = algo.loss(make_batch())
    assert "loss/int_action_chunk" in out and torch.isfinite(out["loss/int_action_chunk"])
    out["loss"].backward()
    assert grad_norm(algo.predictor.blocks) > 0
    assert grad_norm(algo.encoders["act_dec"]) > 0


def test_latent_term_scores_against_the_encoded_target():
    """Without a decoder the comparison is flow-space against x1: no gradient reaches the
    decoder (nothing decodes), and the encoder is pulled from both ends -- the seed it
    produces and the target it produces."""
    torch.manual_seed(0)
    cfg = int_cfg()
    cfg.losses = [t for t in cfg.losses if t["type"] == "integration"]
    algo = build_algorithm(cfg)
    wake_heads(algo)
    algo.loss(make_batch())["loss"].backward()
    assert grad_norm(algo.predictor.blocks) > 0
    assert grad_norm(algo.encoders["act_enc"]) > 0
    assert grad_norm(algo.encoders["act_dec"]) == 0


def test_backprop_runs_the_whole_euler_chain():
    """Every step's velocity is on the gradient path, not just the last one: a longer solve
    puts strictly more of the trunk's compute under the loss."""
    algo = build_algorithm(int_cfg(decoder="act_dec"))
    wake_heads(algo)
    batch = make_batch()
    grads = {}
    for steps in (1, 4):
        algo.zero_grad(set_to_none=True)
        clean, sources, cond, hidden = algo._condition(batch)
        seeds = algo.build_flow_seeds(batch, {**clean, **sources, **cond, **hidden}, training=True)
        out = algo._forward_integrate(clean, sources, cond, seeds, steps)
        out["action_chunk"].square().mean().backward()
        grads[steps] = grad_norm(algo.predictor.blocks)
    assert grads[1] > 0
    assert grads[4] != grads[1]


def test_a_stream_with_no_source_still_integrates_from_noise():
    """`source:` and `integration` are independent knobs: a plain Gaussian-x0 stream may be
    scored too, drawing its own x0 inside the solve."""
    cfg = wam_cfg()
    cfg.predict.action_chunk.source = None
    cfg.losses.append({**INT, "weight": 0.5})
    algo = build_algorithm(cfg)
    wake_heads(algo)
    out = algo.loss(make_batch())
    assert torch.isfinite(out["loss/int_action_chunk"])
    out["loss"].backward()
    assert grad_norm(algo.predictor.blocks) > 0


def test_num_steps_defaults_to_the_eval_solve():
    cfg = int_cfg()
    del cfg.losses[-1]["num_steps"]
    algo = build_algorithm(cfg)
    assert algo._integration["num_steps"] == 2


def test_integration_validation():
    with pytest.raises(ValueError, match="not predict streams"):
        build_algorithm(int_cfg(stream="nope"))
    with pytest.raises(ValueError, match="not an `encoders:` entry"):
        build_algorithm(int_cfg(decoder="nope"))
    cfg = int_cfg()
    cfg.losses.append({**INT, "num_steps": 5})
    with pytest.raises(ValueError, match="disagree on the solve"):
        build_algorithm(cfg)
    cfg = wam_cfg()
    cfg.predict.action_chunk.source = None
    cfg.predict.action_chunk.type = "direct"
    cfg.predict.action_chunk.input = "prev_action"
    cfg.predict.action_chunk.index = "0"
    cfg.losses = [{"type": "flow", "stream": "future_video", "weight": 1.0}, dict(INT)]
    with pytest.raises(ValueError, match="not flow streams"):
        build_algorithm(cfg)


def test_no_integration_term_means_no_extra_solve(monkeypatch):
    """The solve is num_steps extra trunk passes; a config that did not ask for it must not
    pay for it."""
    algo = build_algorithm(ae_cfg())
    assert algo._integration is None
    monkeypatch.setattr(type(algo), "_forward_integrate",
                        lambda *a, **k: pytest.fail("integrated without an integration term"))
    ctx = algo._build_context(make_batch())
    assert "integrated" not in ctx


def test_raw_units_rescale_by_the_normalizer_span():
    """`units: raw` reports the endpoint error in the field's own units, so the number
    survives a change of normalization stats. Normalized metrics do not: recomputing stats
    on more episodes rescales them without the policy changing."""
    import torch
    from thesis.algorithms.losses import IntegrationLoss

    pred, target = torch.zeros(2, 4, 3), torch.zeros(2, 4, 3)
    pred[..., 0] = 0.1                       # 0.1 of normalized range on dim 0 only
    ctx = {"integrated": {"a": pred}, "latent": {"a": target}}

    norm = IntegrationLoss("a", weight=1.0, norm="mse")
    assert norm.label == "int_a"
    base = float(norm(ctx))

    # dim0 spans 200 raw units, so half-span 100: a 0.1 normalized error is 10 raw units
    ctx["unit_scale"] = {"a": torch.tensor([100.0, 1.0, 1.0])}
    raw = IntegrationLoss("a", weight=1.0, norm="mse", units="raw")
    assert raw.label == "int_a_raw"
    assert float(raw(ctx)) == pytest.approx(base * 100.0 ** 2)


def test_raw_units_without_stats_is_a_clear_error():
    import torch
    from thesis.algorithms.losses import IntegrationLoss

    ctx = {"integrated": {"a": torch.zeros(2, 2, 2)}, "latent": {"a": torch.zeros(2, 2, 2)}}
    with pytest.raises(ValueError, match="no scale to convert by|carries no norm_stats"):
        IntegrationLoss("a", weight=1.0, units="raw")(ctx)


def test_unknown_units_rejected():
    from thesis.algorithms.losses import IntegrationLoss
    with pytest.raises(AssertionError, match="unknown integration units"):
        IntegrationLoss("a", weight=1.0, units="physical")
