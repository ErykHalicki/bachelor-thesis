"""SADL's Supply Regularizer: the scores, and the `sadl:` option of a sigreg term in both
the head (fork) and the chain (LeWM BN projector) layouts."""

import pytest
import torch

from thesis.algorithms import build_algorithm
from thesis.algorithms.sadl import SupplyRegularizer, excitation, transmission
from test_generic_vit_predictor import lewm_cfg, make_batch


def test_excitation_scores_isotropic_high_and_shared_mean_low():
    torch.manual_seed(0)
    iso = torch.randn(4096, 16)
    k_iso, f_iso = excitation(iso)
    assert k_iso > 0.95 and f_iso > 0.99
    # the failure SADL targets: one big shared vector plus a sliver of variation
    shared = torch.full((1, 16), 100.0) + 0.1 * torch.randn(4096, 16)
    k_sh, f_sh = excitation(shared)
    assert k_sh < 0.01 and f_sh < 1e-4
    # rank-deficient but zero-mean: f stays high, excitation still drops
    low = torch.randn(4096, 2) @ torch.randn(2, 16)
    assert excitation(low)[0] < 0.3 and excitation(low)[1] > 0.99


def test_scores_are_scale_invariant():
    torch.manual_seed(0)
    h, z = torch.randn(512, 8), torch.randn(512, 8)
    z = z + h
    assert torch.allclose(excitation(h)[0], excitation(7.0 * h)[0], atol=1e-5)
    assert torch.allclose(transmission(h, z), transmission(7.0 * h, z), atol=1e-5)


def test_transmission_sees_whether_variation_reaches_z():
    torch.manual_seed(0)
    h = torch.randn(2048, 8)
    assert transmission(h, h @ torch.randn(8, 8)) > 0.9
    assert transmission(h, torch.randn(2048, 8)) < 0.1


def test_penalty_is_one_sided_and_pushes_a_degraded_h():
    torch.manual_seed(0)
    reg = SupplyRegularizer()
    z = torch.randn(1024, 8)
    healthy = z.clone().requires_grad_()
    loss, logs = reg(healthy, z, torch.tensor(1.0), 100.0)
    assert loss.item() == 0.0 and logs["active"] == 0
    # same z, but H is a shared vector plus a faint copy of it: supply falls short
    degraded = (torch.full((1, 8), 50.0) + 0.01 * z).requires_grad_()
    loss, logs = reg(degraded, z, torch.tensor(1.0), 100.0)
    assert loss.item() > 0 and logs["active"] == 1 and logs["k_exc"] < logs["q"]
    loss.backward()
    assert torch.isfinite(degraded.grad).all() and degraded.grad.abs().sum() > 0
    # no demand at complete endpoint collapse: D(Z) = D(0) closes the gate
    loss, _ = reg(degraded, z, torch.tensor(100.0), 100.0)
    assert loss.item() == 0.0


def _with_sadl(sigreg_overrides):
    cfg = lewm_cfg()
    cfg.losses[1] = {**dict(cfg.losses[1]), **sigreg_overrides}
    return build_algorithm(cfg)


def _assert_sadl_logs_and_grads(algo):
    algo.train()
    out = algo.loss(make_batch(action_rows=3, b=8))
    label = "loss/sigreg_video_context_future_video"
    for key in ("k_exc", "k_tr", "q", "q_reg", "q_exc", "f_h", "f_z", "active", "penalty"):
        assert f"{label}/sadl/{key}" in out
        assert torch.isfinite(out[f"{label}/sadl/{key}"]).all()
    term = algo.loss_terms[1]
    assert torch.is_tensor(term.aux_loss)
    algo.zero_grad()
    term.aux_loss.backward()
    grads = [p.grad for p in algo.encoders["vit"].parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads) or term.aux_loss.item() == 0
    return out, term


def test_sigreg_term_without_sadl_is_unchanged():
    algo = build_algorithm(lewm_cfg())
    out = algo.loss(make_batch(action_rows=3))
    assert not any("/sadl/" in k for k in out)
    assert algo.loss_terms[1].aux_loss == 0.0


def test_sadl_head_layout_supplies_from_the_term_input():
    algo = _with_sadl({"head": "mlp", "proj_hidden": 64, "sadl": 0.01})
    out, term = _assert_sadl_logs_and_grads(algo)
    assert term.sadl_supply is None


def test_sadl_chain_layout_taps_the_bn_projector():
    algo = _with_sadl({"sadl": {"weight": 0.01, "supply": "vit_proj"}})
    out, term = _assert_sadl_logs_and_grads(algo)
    assert term.sadl_supply == "vit_proj" and algo._stage_io is None


def test_sadl_supply_must_name_an_encoder():
    with pytest.raises(ValueError, match="not `encoders:` entries"):
        _with_sadl({"sadl": {"weight": 0.01, "supply": "nope"}})


def test_sadl_options_need_a_weight():
    with pytest.raises(ValueError, match="positive `weight`"):
        _with_sadl({"sadl": {"supply": "vit_proj"}})
