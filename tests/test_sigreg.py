import pytest
import torch

from thesis.algorithms.losses import SIGReg


@pytest.fixture
def sigreg():
    return SIGReg(knots=17, num_proj=1024, dtype=torch.float32)


def test_gaussian_loss_is_low(sigreg):
    proj = torch.randn(256, 4, 768)
    loss = sigreg(proj)
    assert loss.item() < 5.0, f"Expected low loss for N(0,1) input, got {loss.item():.4f}"


def test_uniform_loss_is_high(sigreg):
    proj = torch.rand(256, 4, 768) * 6 - 3
    loss = sigreg(proj)
    assert loss.item() > 20.0, f"Expected high loss for uniform input, got {loss.item():.4f}"


def test_wide_gaussian_loss_is_high(sigreg):
    proj = torch.randn(256, 4, 768) * 5.0
    loss = sigreg(proj)
    assert loss.item() > 100.0, f"Expected high loss for N(0,25) input, got {loss.item():.4f}"



from thesis.algorithms.losses import SIGRegLoss, build_loss_term


def _ctx(z):
    return {"latent": {"s": z}}


def test_per_timestep_equals_pooled_for_single_frame():
    """With T=1 the per-frame bag IS the pooled bag, so the two statistics coincide
    exactly (same seed -> same random projections)."""
    z = torch.randn(64, 8, 32)
    pooled = SIGRegLoss(["s"], 1.0, num_proj=256, stream_dims={"s": 32})
    per_t = SIGRegLoss(["s"], 1.0, num_proj=256, statistic="per_timestep",
                       stream_frames={"s": 1}, stream_dims={"s": 32})
    torch.manual_seed(0)
    a = pooled(_ctx(z))
    torch.manual_seed(0)
    b = per_t(_ctx(z))
    assert torch.allclose(a, b)


def test_per_timestep_penalizes_frame_position_shift():
    """Each frame's mean is offset along one direction, with the within-frame spread
    shrunk so the mixture over frames is still N(0, I). Pooling therefore sees the target
    exactly and is blind; only conditioning on the frame reveals the drift.

    Confining the offset to a single direction of a 64-d space is what makes pooled blind:
    spread over every dimension, the mixture's higher moments drift far enough for the
    pooled statistic to catch it too.
    """
    b, t, d, share = 256, 4, 64, 0.5
    torch.manual_seed(0)
    u = torch.randn(d)
    u /= u.norm()
    base = torch.randn(b, t, d)
    along = (base @ u).unsqueeze(-1)
    offsets = torch.tensor([-1.0, -1 / 3, 1 / 3, 1.0])
    offsets = offsets * (share / offsets.var(unbiased=False)).sqrt()
    z = (base - along * u) + offsets.view(1, t, 1) * u + (1 - share) ** 0.5 * along * u

    pooled = SIGRegLoss(["s"], 1.0, num_proj=512, stream_dims={"s": d})
    per_t = SIGRegLoss(["s"], 1.0, num_proj=512, statistic="per_timestep",
                       stream_frames={"s": t}, stream_dims={"s": d})
    torch.manual_seed(1)
    lp = pooled(_ctx(z)).item()
    torch.manual_seed(1)
    lt = per_t(_ctx(z)).item()
    assert lp < 1.4, f"pooled {lp:.2f} should sit at the null: the mixture IS N(0, I)"
    assert lt > 1.7, f"per-timestep {lt:.2f} should price the per-frame drift"


def test_per_timestep_requires_frame_count():
    with pytest.raises(AssertionError):
        SIGRegLoss(["s"], 1.0, statistic="per_timestep", stream_frames={})


def test_build_loss_term_passes_statistic():
    term = build_loss_term(
        {"type": "sigreg", "streams": ["s"], "weight": 0.09, "statistic": "per_timestep"},
        [], stream_dims={"s": 32}, stream_frames={"s": 4},
    )
    assert term.statistic == "per_timestep"
    loss = term(_ctx(torch.randn(32, 4, 32)))
    assert torch.isfinite(loss)



from omegaconf import OmegaConf

from thesis.algorithms import build_algorithm

GRID_ROPE = {"time": {"share": 0.5, "min_period": 4.0, "max_period": 256.0},
             "height": {"share": 0.25, "period": "auto"},
             "width": {"share": 0.25, "period": "auto"}}

ROPE = {"time": {"share": 0.4, "period": "auto"},
        "seq": {"share": 0.2, "period": "auto"},
        "height": {"share": 0.2, "min_period": 4.0, "max_period": 64.0},
        "width": {"share": 0.2, "min_period": 4.0, "max_period": 64.0}}


def _var_algo():
    return build_algorithm(OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": ROPE, "model_dim": 64, "depth": 1, "num_heads": 4, "mlp_ratio": 2.0},
        "num_flow_steps": 2,
        "encoders": {"enc": {"type": "vit", "rope": GRID_ROPE, "pooling": "grid", "dim": 32, "depth": 1,
                             "num_heads": 2, "patch_size": 8, "img_size": 24, "frames": 4}},
        "conditioning": {"vid": {"from": "observation.images.pixels", "encoder": "enc",
                                 "dim": 32, "index": "-3..0", "fps": 10, "grid": [3, 3]}},
        "predict": {"action": {"from": "action", "type": "flow", "dim": 2,
                               "index": "0..3", "fps": 10}},
    }))


def _logs(algo, z, steps, span):
    algo._stream_axes = {"x": (steps, span)}
    b, d = z.shape[0], z.shape[-1]
    return {k: v.item() for k, v in
            algo._variance_logs({"x": z.reshape(b, steps * span, d)}).items()}


def test_variance_logs_reported_for_token_streams():
    algo = _var_algo()
    out = algo.loss({
        "observation.images.pixels": (torch.rand(8, 4, 3, 24, 24) * 255).to(torch.uint8),
        "action": torch.randn(8, 4, 2),
    })
    assert {"var/vid/temporal", "var/vid/spatial", "var/action/temporal"} <= set(out)


def test_variance_measures_context_and_future_as_one_family():
    """Streams sharing a `from:` field concatenate along time and are measured jointly
    under the members' common prefix -- the estimate covers the full context+future
    trajectory, not each half individually."""
    algo, (B, S, D) = _var_algo(), (64, 3, 16)
    algo._stream_axes = {"cam_context": (2, S), "cam_future": (4, S)}
    algo.conditioning = {"cam_context": {"from": "observation.images.cam"}}
    algo.predict_spec = {"cam_future": {"from": "observation.images.cam"}}
    torch.manual_seed(0)
    ctx, fut = torch.randn(B, 2, S, D), torch.randn(B, 4, S, D)
    logs = algo._variance_logs({
        "cam_context": ctx.reshape(B, 2 * S, D), "cam_future": fut.reshape(B, 4 * S, D),
    })
    assert set(logs) == {"var/cam/temporal", "var/cam/spatial"}
    joint = torch.cat([ctx, fut], dim=1)
    total = joint.reshape(-1, D).var(dim=0).mean()
    assert logs["var/cam/temporal"].item() == pytest.approx(
        (joint.var(dim=1).mean() / total).item(), rel=1e-5)


def test_variance_is_one_for_independent_tokens_and_zero_when_collapsed():
    """The diagnostic exists to separate correlated-but-varying from constant, which a
    SIGReg term cannot: both read as a reduced effective sample count there."""
    algo, (B, T, S, D) = _var_algo(), (256, 4, 9, 16)
    torch.manual_seed(0)
    g = torch.randn(B, T, S, D)

    iid = _logs(algo, g, T, S)
    assert iid["var/x/temporal"] == pytest.approx(1.0, abs=0.1)
    assert iid["var/x/spatial"] == pytest.approx(1.0, abs=0.1)

    flat_t = _logs(algo, g[:, :1].repeat(1, T, 1, 1), T, S)
    assert flat_t["var/x/temporal"] == pytest.approx(0.0, abs=1e-6)
    assert flat_t["var/x/spatial"] == pytest.approx(1.0, abs=0.1)

    flat_s = _logs(algo, g[:, :, :1].repeat(1, 1, S, 1), T, S)
    assert flat_s["var/x/spatial"] == pytest.approx(0.0, abs=1e-6)
    assert flat_s["var/x/temporal"] == pytest.approx(1.0, abs=0.1)

    z = g.clone()
    for t in range(1, T):
        z[:, t] = 0.9 * z[:, t - 1] + (1 - 0.81) ** 0.5 * g[:, t]
    smooth = _logs(algo, z, T, S)["var/x/temporal"]
    assert 0.01 < smooth < 0.5, smooth


def test_variance_logs_skipped_for_scalar_grid_streams():
    """A [1, 1] grid has no spatial axis and a single step has no temporal one."""
    algo = _var_algo()
    assert _logs(algo, torch.randn(32, 4, 1, 8), 4, 1).keys() == {"var/x/temporal"}
    assert _logs(algo, torch.randn(32, 1, 9, 8), 1, 9).keys() == {"var/x/spatial"}




def _grid(b=384, t=4, s=4, d=48, seed=0):
    torch.manual_seed(seed)
    return torch.randn(b, t, s, d)


def _stat(statistic, z, t):
    b, d = z.shape[0], z.shape[-1]
    term = SIGRegLoss(["s"], 1.0, num_proj=512, statistic=statistic,
                      stream_frames={"s": t}, stream_dims={"s": d})
    torch.manual_seed(1)
    return term(_ctx(z.reshape(b, -1, d))).item()


def test_per_spatial_catches_temporal_collapse_that_per_timestep_misses():
    """The two statistics are transposes: separating an axis prices drift along it,
    pooling an axis prices dependence along it. Only per_spatial pools time, so only it
    sees a stream that is constant over time."""
    t, s = 4, 4
    g = _grid(t=t, s=s)
    frozen_in_time = g[:, :1].repeat(1, t, 1, 1)
    assert _stat("per_spatial", frozen_in_time, t) > 2.5 * _stat("per_spatial", g, t)
    assert _stat("per_timestep", frozen_in_time, t) < 1.5 * _stat("per_timestep", g, t)


def test_per_timestep_catches_grid_collapse_that_per_spatial_misses():
    t, s = 4, 4
    g = _grid(t=t, s=s)
    frozen_in_space = g[:, :, :1].repeat(1, 1, s, 1)
    assert _stat("per_timestep", frozen_in_space, t) > 2.5 * _stat("per_timestep", g, t)
    assert _stat("per_spatial", frozen_in_space, t) < 1.5 * _stat("per_spatial", g, t)


def test_per_spatial_equals_pooled_for_a_single_grid_position():
    """With one token per step there is no grid axis to separate, so the per-position bag
    IS the pooled bag -- which is why a [1, 1] stream can use either name."""
    z = _grid(t=4, s=1)
    assert _stat("per_spatial", z, 4) == pytest.approx(_stat("pooled", z, 4), rel=1e-9)


def test_statistic_qualifies_the_label_so_terms_do_not_collide():
    """Two terms regularizing one set of streams along different axes must log separately."""
    streams = ["a", "b"]
    kw = dict(stream_frames={"a": 2, "b": 2}, stream_dims={"a": 8, "b": 8})
    pooled = SIGRegLoss(streams, 1.0, stream_dims={"a": 8, "b": 8})
    per_t = SIGRegLoss(streams, 1.0, statistic="per_timestep", **kw)
    per_s = SIGRegLoss(streams, 1.0, statistic="per_spatial", **kw)
    assert len({pooled.label, per_t.label, per_s.label}) == 3
    assert pooled.label == "sigreg_a_b", "the default statistic keeps the original name"




def test_concat_temporal_matches_scoring_one_joined_stream():
    """le-wm encodes the whole window at once and calls SIGReg once on it. Joining a
    context and future stream on the time axis must reproduce that exactly, rather than
    summing two separately-normalized terms."""
    b, d = 256, 32
    torch.manual_seed(0)
    ctx_z, fut_z = torch.randn(b, 2, d), torch.randn(b, 3, d)
    joined = SIGRegLoss(["c", "f"], 1.0, num_proj=256, statistic="per_timestep",
                        stream_frames={"c": 2, "f": 3}, concat="temporal",
                        stream_dims={"c": d, "f": d})
    single = SIGRegLoss(["w"], 1.0, num_proj=256, statistic="per_timestep",
                        stream_frames={"w": 5}, stream_dims={"w": d})
    torch.manual_seed(1)
    a = joined({"latent": {"c": ctx_z, "f": fut_z}}).item()
    torch.manual_seed(1)
    expected = single({"latent": {"w": torch.cat([ctx_z, fut_z], dim=1)}}).item()
    assert a == pytest.approx(expected, rel=1e-9)


def test_concat_none_is_the_sum_of_separate_terms():
    """Default stays additive, so grouping streams into one term is purely cosmetic."""
    b, d, t = 256, 32, 3
    torch.manual_seed(0)
    za, zb = torch.randn(b, t, d), torch.randn(b, t, d)
    sf = {"a": t, "b": t}
    kw = dict(num_proj=256, statistic="per_timestep", stream_frames=sf,
              stream_dims={"a": d, "b": d})
    # build every term before seeding: each head draws from the RNG, which would
    # otherwise leave the forwards on different projections
    grouped = SIGRegLoss(["a", "b"], 1.0, **kw)
    alone_a = SIGRegLoss(["a"], 1.0, **kw)
    alone_b = SIGRegLoss(["b"], 1.0, **kw)
    torch.manual_seed(1)
    both = grouped({"latent": {"a": za, "b": zb}}).item()
    torch.manual_seed(1)
    one_a = alone_a({"latent": {"a": za}}).item()
    torch.manual_seed(1)
    one_b = alone_b({"latent": {"b": zb}}).item()
    # the grouped term draws a fresh projection per stream while the singles reuse
    # the first, so the two sides carry independent Monte-Carlo noise
    assert both == pytest.approx(one_a + one_b, rel=0.05)


def test_concat_prices_streams_collapsing_onto_each_other():
    """Summed terms never compare streams, so two identical cameras cost nothing. Joining
    them puts both in one bag, where the duplication reads as a halved sample count."""
    b, d, t = 256, 32, 3
    torch.manual_seed(0)
    za, zb = torch.randn(b, t, d), torch.randn(b, t, d)
    sf = {"a": t, "b": t}

    def score(latents, concat):
        term = SIGRegLoss(["a", "b"], 1.0, num_proj=256, statistic="per_timestep",
                          stream_frames=sf, concat=concat, stream_dims={"a": d, "b": d})
        torch.manual_seed(1)
        return term({"latent": latents}).item()

    distinct, identical = {"a": za, "b": zb}, {"a": za, "b": za.clone()}
    assert score(identical, None) == pytest.approx(score(distinct, None), rel=0.2)
    for concat in ("spatial", "batch"):
        assert score(identical, concat) > 1.7 * score(distinct, concat)


def test_concat_requires_the_other_axes_to_match():
    b, d = 64, 16
    term = SIGRegLoss(["a", "b"], 1.0, num_proj=64, statistic="per_timestep",
                      stream_frames={"a": 2, "b": 3}, concat="spatial",
                      stream_dims={"a": d, "b": d})
    with pytest.raises(ValueError, match="other axes must match"):
        term({"latent": {"a": torch.randn(b, 2, d), "b": torch.randn(b, 3, d)}})


def test_concat_is_rejected_when_unknown_and_needs_frame_counts():
    with pytest.raises(AssertionError, match="unknown concat"):
        SIGRegLoss(["a"], 1.0, concat="diagonal")
    with pytest.raises(AssertionError, match="frame count"):
        SIGRegLoss(["a"], 1.0, concat="temporal", stream_frames={})


def test_build_loss_term_passes_concat():
    term = build_loss_term(
        {"type": "sigreg", "streams": ["a", "b"], "weight": 0.09,
         "statistic": "per_timestep", "concat": "temporal"},
        [], stream_dims={"a": 8, "b": 8}, stream_frames={"a": 2, "b": 3},
    )
    assert term.concat == "temporal"
    assert term.label.endswith("_per_timestep_temporal")


def test_concat_temporal_deduplicates_overlapping_windows():
    """A shift-by-one target overlaps its own context (le-wm: frames -1, 0 of -2..0 and
    -1..1). Those are one moment of one timestream, so joining must score the union of
    distinct times, not the concatenation."""
    b, d = 256, 32
    torch.manual_seed(0)
    ctx_z, fut_z = torch.randn(b, 3, d), torch.randn(b, 3, d)
    fut_z[:, 0], fut_z[:, 1] = ctx_z[:, 1], ctx_z[:, 2]
    times = {"c": (-0.2, -0.1, 0.0), "f": (-0.1, 0.0, 0.1)}
    kw = dict(num_proj=256, statistic="per_timestep", stream_frames={"c": 3, "f": 3},
              stream_dims={"c": d, "f": d})

    joined = SIGRegLoss(["c", "f"], 1.0, concat="temporal", stream_times=times, **kw)
    torch.manual_seed(1)
    got = joined({"latent": {"c": ctx_z, "f": fut_z}}).item()

    union = torch.stack([ctx_z[:, 0], ctx_z[:, 1], ctx_z[:, 2], fut_z[:, 2]], dim=1)
    ref = SIGRegLoss(["w"], 1.0, num_proj=256, statistic="per_timestep",
                     stream_frames={"w": 4}, stream_dims={"w": d})
    torch.manual_seed(1)
    assert got == pytest.approx(ref({"latent": {"w": union}}).item(), rel=1e-9)


def test_concat_temporal_without_times_keeps_every_slice():
    """Times are what identify a shared frame; with none supplied nothing is dropped."""
    b, d = 128, 16
    torch.manual_seed(0)
    za, zb = torch.randn(b, 3, d), torch.randn(b, 3, d)
    kw = dict(num_proj=128, statistic="per_timestep", stream_frames={"a": 3, "b": 3},
              concat="temporal", stream_dims={"a": d, "b": d})
    joined = SIGRegLoss(["a", "b"], 1.0, **kw)
    ref = SIGRegLoss(["w"], 1.0, num_proj=128, statistic="per_timestep",
                     stream_frames={"w": 6}, stream_dims={"w": d})
    torch.manual_seed(1)
    got = joined({"latent": {"a": za, "b": zb}}).item()
    torch.manual_seed(1)
    expected = ref({"latent": {"w": torch.cat([za, zb], 1)}}).item()
    assert got == pytest.approx(expected, rel=1e-9)


def test_stream_times_survive_the_last_sentinel():
    """`index: "last"` parses to a string, not an offset; it must stay usable as a frame
    identity so two streams sharing the goal frame still dedupe."""
    from thesis.utils.spec import LAST
    b, d = 64, 16
    torch.manual_seed(0)
    goal = torch.randn(b, 1, d)
    times = {"a": (0.0, LAST), "b": (LAST,)}
    term = SIGRegLoss(["a", "b"], 1.0, num_proj=64, statistic="per_timestep",
                      stream_frames={"a": 2, "b": 1}, concat="temporal", stream_times=times,
                      stream_dims={"a": d, "b": d})
    out = term({"latent": {"a": torch.cat([torch.randn(b, 1, d), goal], 1), "b": goal}})
    assert torch.isfinite(out)




def test_head_defaults_to_a_square_identity_projection():
    """Omitting proj_dim now gives a square head rather than none, so the backbone is
    always freed from making its raw features isotropic-Gaussian. Identity init keeps that
    default numerically neutral until the head learns."""
    d = 24
    term = SIGRegLoss(["s"], 1.0, num_proj=64, stream_dims={"s": d})
    assert term.head.weight.shape == (d, d)
    assert torch.equal(term.head.weight, torch.eye(d))
    z = torch.randn(32, 3, d)
    assert torch.allclose(term.head(z), z, atol=1e-6)


def test_proj_dim_still_narrows_and_is_randomly_initialized():
    d, proj = 24, 8
    term = SIGRegLoss(["s"], 1.0, num_proj=64, stream_dims={"s": d}, proj_dim=proj)
    assert term.head.weight.shape == (proj, d)
    assert not torch.equal(term.head.weight, torch.zeros(proj, d))


def test_head_needs_every_stream_dim_and_rejects_mismatched_ones():
    with pytest.raises(ValueError, match="latent dim of every stream"):
        SIGRegLoss(["a", "b"], 1.0, stream_dims={"a": 8})
    with pytest.raises(ValueError, match="equal stream dims"):
        SIGRegLoss(["a", "b"], 1.0, stream_dims={"a": 8, "b": 16})


def test_head_trains_and_is_the_only_thing_the_term_owns():
    d = 16
    term = SIGRegLoss(["s"], 1.0, num_proj=64, stream_dims={"s": d})
    assert [n for n, _ in term.named_parameters()] == ["head.weight"]
    z = torch.randn(64, 2, d, requires_grad=True)
    term({"latent": {"s": z}}).backward()
    assert term.head.weight.grad is not None and term.head.weight.grad.abs().sum() > 0
