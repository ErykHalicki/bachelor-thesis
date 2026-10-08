"""UWM-style independent flow time: one t per flow stream in training, and clamped
streams selecting the sampling mode (policy, forward/inverse dynamics, video, joint)."""

import pytest
import torch
from omegaconf import OmegaConf

from thesis.algorithms import build_algorithm

ROPE = {"time": {"share": 0.6, "period": "auto"}, "seq": {"share": 0.2, "period": "auto"}}


def uwm_cfg(**flow_time):
    """State context, an action chunk and two future 'frame latents' as flow streams."""
    return OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": ROPE, "model_dim": 32, "depth": 2, "num_heads": 4, "mlp_ratio": 2.0},
        "num_flow_steps": 3,
        "flow_time": {"independent": True, **flow_time},
        "encoders": {},
        "conditioning": {
            "state": {"from": "state", "dim": 5, "index": "-1..0"},
        },
        "predict": {
            "action": {"from": "action", "type": "flow", "dim": 3, "index": "0..3",
                       "block_size": 4},
            "future": {"from": "future", "type": "flow", "dim": 6, "index": "1"},
            "wrist": {"from": "wrist", "type": "flow", "dim": 6, "index": "1"},
        },
        "losses": [
            {"type": "flow", "stream": "action", "weight": 1.0},
            {"type": "flow", "stream": "future", "weight": 1.0},
            {"type": "flow", "stream": "wrist", "weight": 1.0},
        ],
    })


def wake(model):
    """Randomize the zero-initialized output projections: at init every velocity is
    exactly zero, so no input changes an output and no gradient reaches the embedders."""
    with torch.no_grad():
        for p in model.parameters():
            if not p.abs().sum():
                p.normal_(0.0, 0.1)
    return model


def batch(n=8):
    g = torch.Generator().manual_seed(0)
    return {
        "state": torch.randn(n, 2, 5, generator=g),
        "action": torch.randn(n, 4, 3, generator=g),
        "future": torch.randn(n, 1, 6, generator=g),
        "wrist": torch.randn(n, 1, 6, generator=g),
    }


def test_each_stream_gets_its_own_embedder():
    model = build_algorithm(uwm_cfg())
    trunk = model.predictor
    assert trunk.independent_flow_time and trunk.t_embedder is None
    assert sorted(trunk.t_embedders) == ["action", "future", "wrist"]


def test_loss_trains_both_streams():
    model = wake(build_algorithm(uwm_cfg(noise_prob=0.1, clean_prob=0.1)))
    out = model.loss(batch())
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    for name in ("action", "future", "wrist"):
        grads = [p.grad for p in model.predictor.t_embedders[name].parameters()]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_endpoint_probabilities_pin_the_draw():
    model = build_algorithm(uwm_cfg(noise_prob={"future": 1.0}, clean_prob={"action": 1.0}))
    t = model._sample_independent_time({"action": torch.zeros(64, 4, 3),
                                        "future": torch.zeros(64, 1, 6),
                                        "wrist": torch.zeros(64, 1, 6)})
    assert torch.all(t["future"] == 0.0)
    assert torch.all(t["action"] == 1.0)


def test_a_stream_handed_in_clean_scores_zero():
    model = build_algorithm(uwm_cfg(clean_prob={"action": 1.0}))
    ctx = model._build_context(batch())
    assert torch.equal(ctx["pred"]["action"], ctx["target"]["action"])
    assert not torch.equal(ctx["pred"]["future"], ctx["target"]["future"])


@pytest.mark.parametrize(
    ("clamp", "returned"),
    [
        (None, {"action", "future", "wrist"}),                     # joint
        ({"future": 0.0, "wrist": 0.0}, {"action"}),               # policy
        ({"action": 1.0}, {"future", "wrist"}),                    # forward dynamics
        ({"future": 1.0, "wrist": 1.0}, {"action"}),               # inverse dynamics
        ({"action": 0.0}, {"future", "wrist"}),                    # video prediction
    ],
)
def test_every_mode_samples(clamp, returned):
    model = build_algorithm(uwm_cfg())
    out = model.predict(batch(), clamp=clamp)
    assert set(out) == returned
    for v in out.values():
        assert torch.isfinite(v).all()


def test_a_clean_clamp_conditions_on_the_given_value():
    model = wake(build_algorithm(uwm_cfg()))
    torch.manual_seed(0)
    a = model.predict(batch(), clamp={"action": 1.0})["future"]
    other = batch()
    other["action"] = other["action"] + 5.0
    torch.manual_seed(0)
    b = model.predict(other, clamp={"action": 1.0})["future"]
    assert not torch.allclose(a, b)


def test_inference_clamp_config_is_the_default_mode():
    cfg = uwm_cfg()
    cfg.inference_clamp = {"future": 0.0, "wrist": 0.0}
    model = build_algorithm(cfg)
    assert set(model.predict(batch())) == {"action"}


def test_shared_time_models_are_unchanged():
    cfg = uwm_cfg()
    del cfg["flow_time"]
    model = build_algorithm(cfg)
    assert model.predictor.t_embedder is not None and not model.predictor.t_embedders
    with pytest.raises(AssertionError, match="independent_flow_time"):
        model.predict(batch(), clamp={"future": 0.0})


def grouped_cfg(**flow_time):
    return uwm_cfg(groups=[["future", "wrist"]], **flow_time)


def test_a_time_group_shares_every_draw():
    model = build_algorithm(grouped_cfg(noise_prob=0.3, clean_prob=0.3))
    flow = {"action": torch.zeros(512, 4, 3), "future": torch.zeros(512, 1, 6),
            "wrist": torch.zeros(512, 1, 6)}
    t = model._sample_independent_time(flow)
    assert torch.equal(t["future"], t["wrist"])
    assert not torch.equal(t["future"], t["action"])
    # policy mode -- the whole group at noise -- now comes up at the pin rate itself
    assert 0.2 < (t["future"] == 0).float().mean() < 0.4


def test_ungrouped_streams_draw_apart():
    model = build_algorithm(uwm_cfg())
    flow = {"action": torch.zeros(64, 4, 3), "future": torch.zeros(64, 1, 6),
            "wrist": torch.zeros(64, 1, 6)}
    t = model._sample_independent_time(flow)
    assert not torch.equal(t["future"], t["wrist"])


@pytest.mark.parametrize(
    ("groups", "extra", "message"),
    [
        ([["future", "nope"]], {}, "not a flow stream"),
        ([["future", "wrist"], ["wrist", "action"]], {}, "twice"),
        ([["future"]], {}, "groups nothing"),
        ([["future", "wrist"]], {"noise_prob": {"future": 0.1, "wrist": 0.2}}, "different"),
    ],
)
def test_bad_groups_fail_at_build(groups, extra, message):
    with pytest.raises(ValueError, match=message):
        build_algorithm(uwm_cfg(groups=groups, **extra))


def test_pinning_part_of_a_group_is_refused():
    model = build_algorithm(grouped_cfg())
    with pytest.raises(ValueError, match="only in part"):
        model.predict(batch(), clamp={"future": 0.0})
    with pytest.raises(ValueError, match="only in part"):
        model.predict(batch(), clamp={"future": 0.0, "wrist": 1.0})
    assert set(model.predict(batch(), clamp={"future": 0.0, "wrist": 0.0})) == {"action"}
