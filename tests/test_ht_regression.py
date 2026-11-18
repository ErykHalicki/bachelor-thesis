"""Heteroscedastic Student-t (HT) regression: the `ht` loss term, the three `scale:`
heads on an mse stream, and the paper's `query: zeros` readout of a flow network."""

import math

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from thesis.algorithms import build_algorithm
from thesis.algorithms.generic_vit_predictor import scale_token_name
from thesis.algorithms.losses import HTLoss

from test_generic_vit_predictor import ENCODER, PROJ, ROPE


def policy_cfg(scale="token", query="learned", action_type="mse", losses=None):
    """Video context in, a 4-step 2-d action chunk out."""
    action = {"from": "action", "type": action_type, "dim": 2, "index": "0..3",
              "fps": 1.0, "token_in": {"type": "mlp", "mlp_dim": 32},
              "token_out": {"type": "mlp", "mlp_dim": 16}}
    if action_type == "mse":
        action["query"] = query
        if scale is not None:
            action["scale"] = scale
    return OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": ROPE, "model_dim": 32, "depth": 2, "num_heads": 4,
                  "dim_head": 16, "mlp_ratio": 2.0},
        "encoders": {"vit": dict(ENCODER), "vit_proj": dict(PROJ)},
        "conditioning": {
            "video_context": {
                "from": "observation.images.pixels", "encoder": "vit_proj",
                "dim": 32, "index": "-2..0", "grid": [1, 1],
            },
        },
        "predict": {"action": action},
        "losses": losses or ([{"type": "ht", "stream": "action"}] if action_type == "mse"
                             else [{"type": "flow", "stream": "action"}]),
    })


def make_batch(b=3):
    torch.manual_seed(0)
    return {
        "observation.images.pixels": (torch.rand(b, 3, 3, 28, 28) * 255).to(torch.uint8),
        "action": torch.randn(b, 4, 2),
    }


def reference_ht(pred, target, raw, nu, bias, min_scale=1e-3, mask=None):
    """The authors' `ht_loss` (RegressionPolicy/ht_loss/functional.py), reduction mean."""
    B = pred.shape[0]
    sigma = F.softplus(raw.reshape(B) + bias) + min_scale
    sq = (pred - target).square()
    w = torch.ones_like(sq) if mask is None else torch.broadcast_to(mask, sq.shape).to(sq)
    count = w.reshape(B, -1).sum(1)
    S = (sq * w).reshape(B, -1).sum(1)
    dim = torch.where(count > 0, count, torch.ones_like(count))
    nu = 4 * dim if nu is None else nu
    L = 0.5 * (nu + count) * torch.log1p(S / (nu * sigma.square())) + count * sigma.log()
    return L.sum() / count.sum()


def ctx_for(pred, target, raw, valid=None):
    ctx = {"pred": {"a": pred}, "target": {"a": target}, "scale": {"a": raw}}
    if valid is not None:
        ctx["valid"] = {"a": valid}
    return ctx


@pytest.mark.parametrize("nu", [None, 200.0])
def test_loss_matches_authors_reference(nu):
    torch.manual_seed(1)
    pred, target = torch.randn(5, 4, 3), torch.randn(5, 4, 3)
    raw = torch.randn(5, 1)
    valid = torch.ones(5, 4, dtype=torch.bool)
    valid[0, 2:] = False
    valid[3] = False
    term = HTLoss("a", 1.0, nu=nu, scale_bias=0.3)
    ours = term(ctx_for(pred, target, raw, valid))
    ref = reference_ht(pred, target, raw, nu, 0.3, mask=valid.unsqueeze(-1))
    assert torch.allclose(ours, ref, atol=1e-6)


def test_gradient_is_eq2():
    """d L / d f = r / sigma^2 * (nu + d) / (nu + |r|^2 / sigma^2), per event."""
    torch.manual_seed(2)
    pred = torch.randn(2, 4, 3, requires_grad=True)
    target = torch.randn(2, 4, 3)
    raw = torch.randn(2, 1)
    term = HTLoss("a", 1.0, scale_bias=0.0)
    term(ctx_for(pred, target, raw)).backward()
    d = 12.0
    nu = 4 * d
    sigma = F.softplus(raw[:, 0]) + 1e-3
    r = (pred - target).detach()
    s = r.square().sum((1, 2)) / sigma.square()
    w = (nu + d) / (nu + s)
    expected = r / sigma.square().view(-1, 1, 1) * w.view(-1, 1, 1) / (2 * d)
    assert torch.allclose(pred.grad, expected, atol=1e-6)


def test_large_nu_is_heteroscedastic_gaussian():
    torch.manual_seed(3)
    pred, target, raw = torch.randn(4, 4, 2), torch.randn(4, 4, 2), torch.randn(4, 1)
    term = HTLoss("a", 1.0, nu_ratio=1e7, scale_bias=0.0)
    sigma = F.softplus(raw[:, 0]) + 1e-3
    S = (pred - target).square().sum((1, 2))
    gauss = (0.5 * S / sigma.square() + 8 * sigma.log()).sum() / 32
    assert torch.allclose(term(ctx_for(pred, target, raw)), gauss, rtol=1e-4)


def test_calibration_starts_sigma_at_residual_rms():
    torch.manual_seed(4)
    pred, target = torch.randn(6, 4, 2), 3 * torch.randn(6, 4, 2)
    term = HTLoss("a", 1.0)
    assert not bool(term.calibrated)
    term(ctx_for(pred, target, torch.zeros(6, 1)))
    rms = (pred - target).square().mean().sqrt()
    assert bool(term.calibrated)
    assert torch.allclose(F.softplus(term.scale_bias) + 1e-3, rms, atol=1e-5)
    before = term.scale_bias.clone()
    term(ctx_for(pred, 2 * target, torch.zeros(6, 1)))
    assert torch.equal(term.scale_bias, before)
    assert "scale_bias" in term.state_dict() and "calibrated" in term.state_dict()


def test_step_events_ignore_padded_steps():
    torch.manual_seed(5)
    pred, target, raw = torch.randn(2, 4, 3), torch.randn(2, 4, 3), torch.randn(2, 4)
    valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    term = HTLoss("a", 1.0, scale_bias=0.0)
    ours = term(ctx_for(pred, target, raw, valid))
    target2 = target.clone()
    target2[0, 2:] += 100.0
    assert torch.allclose(term(ctx_for(pred, target2, raw, valid)), ours)
    keep = valid.flatten()
    ref = reference_ht(pred.reshape(8, 1, 3)[keep], target.reshape(8, 1, 3)[keep],
                       raw.reshape(8, 1)[keep], None, 0.0)
    assert torch.allclose(ours, ref, atol=1e-6)


@pytest.mark.parametrize("scale", ["token", "pool", "step"])
@pytest.mark.parametrize("query", ["learned", "zeros"])
def test_scale_heads_train_and_predict(scale, query):
    algo = build_algorithm(policy_cfg(scale=scale, query=query))
    out = algo.loss(make_batch())
    for key in ("loss/action", "loss/action/mse", "loss/action/sigma", "loss/action/weight"):
        assert key in out
    out["loss"].backward()
    head = algo.predictor.scale_heads["action"]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    if scale == "token":
        assert algo.predictor.scale_embed["action"].grad is not None
    pred = algo.predict(make_batch())
    assert set(pred) == {"action"} and pred["action"].shape == (3, 4, 2)
    assert torch.equal(pred["action"], algo.predict(make_batch())["action"])


def test_scale_token_is_a_pure_reader():
    """No token attends the readout token, so the mean is bit-identical without it --
    which is what lets inference drop it."""
    algo = build_algorithm(policy_cfg(scale="token"))
    algo.eval()
    trunk = algo.predictor
    for p in trunk.parameters():
        torch.nn.init.normal_(p, std=0.1)
    clean, sources, cond, _ = algo._condition(make_batch())
    ref = next(iter(clean.values()))
    with_tok, scale = trunk({**clean, **trunk.query_tokens(ref, scale=True)}, None, sources,
                            cond=cond, return_scale=True)
    without = trunk({**clean, **trunk.query_tokens(ref)}, None, sources, cond=cond)
    assert scale["action"].shape == (3, 1)
    assert torch.allclose(with_tok["action"], without["action"], atol=1e-6)
    sl, j = trunk.stream_slices, trunk.stream_slices[scale_token_name("action")]
    mask = trunk._mask()
    if mask is not None:
        assert not mask[sl["action"], j].any() and not mask[sl["video_context"], j].any()
        assert mask[j, sl["action"]].all() and mask[j, sl["video_context"]].all()


def test_zeros_readout_loads_a_flow_checkpoint():
    """`query: zeros` is the flow network read at t = 0 with zero actions, so every flow
    weight has a home and the only new weights are the scale head's."""
    flow = build_algorithm(policy_cfg(action_type="flow"))
    ht = build_algorithm(policy_cfg(scale="token", query="zeros"))
    missing, unexpected = ht.load_state_dict(flow.state_dict(), strict=False)
    assert not unexpected
    assert all("scale" in k or k.startswith("loss_terms.") for k in missing), missing
    trunk = flow.predictor
    clean, sources, cond, _ = flow._condition(make_batch())
    v = trunk({**clean, "action": torch.zeros(3, 4, 2)}, torch.zeros(3), sources, cond=cond)
    ht.eval()
    assert torch.allclose(ht.predict(make_batch())["action"], v["action"], atol=1e-6)


def test_ht_without_scale_head_is_refused():
    with pytest.raises(ValueError, match="scale"):
        build_algorithm(policy_cfg(scale=None))


def test_scale_only_on_mse_streams():
    cfg = policy_cfg(action_type="flow")
    cfg.predict.action.scale = "token"
    with pytest.raises(AssertionError, match="mse"):
        build_algorithm(cfg)


def test_zeros_readout_refuses_flow_neighbours():
    cfg = policy_cfg(scale="token", query="zeros")
    cfg.predict.other = {"from": "action", "role": "aux", "type": "flow", "dim": 2,
                         "index": "0..3", "fps": 1.0}
    with pytest.raises(Exception):
        build_algorithm(cfg)


def test_inverse_softplus_is_exact():
    term = HTLoss("a", 1.0)
    S, d = torch.tensor([50.0]), torch.tensor([2.0])
    term._calibrate(S, d)
    assert math.isclose(F.softplus(term.scale_bias).item() + 1e-3, 5.0, rel_tol=1e-6)


def test_dim_mask_drops_padded_coordinates():
    """A zero-padded action width counts in no loss: masked MSE is the mean over the real
    coordinates, and HT's d is their count, as the authors' `mask`."""
    from thesis.algorithms.losses import PredictionLoss

    torch.manual_seed(6)
    pred, target = torch.randn(3, 4, 5), torch.randn(3, 4, 5)
    dims = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    valid = torch.ones(3, 4, dtype=torch.bool)
    valid[2, 3] = False
    ctx = {"pred": {"a": pred}, "target": {"a": target}, "valid": {"a": valid},
           "dim_valid": {"a": dims}}
    w = dims[:, None, :] & valid[..., None]
    mse = PredictionLoss(["a"], 1.0)(ctx)
    assert torch.allclose(mse, (pred - target).square()[w].mean())
    padded = target.clone()
    padded[~dims[:, None, :].expand_as(target)] += 50.0
    assert torch.allclose(PredictionLoss(["a"], 1.0)({**ctx, "target": {"a": padded}}), mse)

    raw = torch.randn(3, 1)
    ht = HTLoss("a", 1.0, scale_bias=0.2)(
        {**ctx, "scale": {"a": raw}})
    ref = reference_ht(pred, target, raw, None, 0.2, mask=w)
    assert torch.allclose(ht, ref, atol=1e-6)
