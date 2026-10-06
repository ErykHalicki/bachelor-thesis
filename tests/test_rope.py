import math

import pytest
import torch

from thesis.algorithms.rope import (
    RopeND,
    apply_rotary_emb,
    axes_from_positions,
    fit_ladder,
    make_pos_ids,
    resolve_axes,
    split_dims,
)

# a b601-shaped window: 2.5 fps context, then a 36-step action chunk at 30 fps
TIMES = torch.cat([torch.tensor([-0.4, 0.0]), torch.arange(36).float() / 30.0])
TRUNK_POS = {"time": TIMES, "seq": torch.arange(len(TIMES)).float()}
TRUNK_SPEC = {"time": {"share": 0.75, "period": "auto"}, "seq": {"share": 0.25, "period": "auto"}}
TRUNK_AXES = axes_from_positions(TRUNK_POS, TRUNK_SPEC)

GRID_POS = make_pos_ids(torch.arange(4), (8, 8))
GRID_AXES = axes_from_positions(
    GRID_POS,
    {a: {"share": s, "period": "auto"}
     for a, s in [("time", 0.5), ("height", 0.25), ("width", 0.25)]},
)


def test_make_pos_ids_video_grid():
    pos = make_pos_ids(torch.tensor([0.0, 0.5]), grid=(2, 3))
    assert set(pos) == {"time", "height", "width"}
    assert all(v.shape == (12,) for v in pos.values())
    assert torch.equal(pos["time"], torch.tensor([0.0] * 6 + [0.5] * 6))
    assert torch.equal(pos["height"][:6], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
    assert torch.equal(pos["width"][:6], torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]))


def test_make_pos_ids_default_grid_is_nonspatial():
    times = torch.tensor([-0.2, 0.0, 0.2, 0.4])
    pos = make_pos_ids(times)
    assert torch.equal(pos["time"], times)
    assert torch.equal(pos["height"], torch.zeros(4))
    assert torch.equal(pos["width"], torch.zeros(4))


def test_shares_set_the_split():
    assert split_dims({"time": 0.75, "seq": 0.25}, 64) == {"time": 48, "seq": 16}


def test_shares_are_normalized():
    assert split_dims({"time": 3, "seq": 1}, 64) == split_dims({"time": 0.75, "seq": 0.25}, 64)


def test_split_covers_the_head_dim_exactly_and_stays_even():
    for shares, head_dim in [
        ({"time": 0.75, "seq": 0.25}, 64),
        ({"time": 1, "height": 1, "width": 1}, 64),      # 64/3 does not divide
        ({"time": 0.5, "height": 0.25, "width": 0.25}, 32),
        ({"time": 0.9, "seq": 0.1}, 16),
    ]:
        dims = split_dims(shares, head_dim)
        assert sum(dims.values()) == head_dim
        assert all(d % 2 == 0 and d >= 2 for d in dims.values())


def test_a_tiny_share_still_rotates():
    dims = split_dims({"time": 0.999, "seq": 0.001}, 64)
    assert dims["seq"] == 2
    assert sum(dims.values()) == 64


def test_head_dim_too_small_for_the_axes_is_an_error():
    with pytest.raises(AssertionError, match="rotation pair"):
        split_dims({"time": 1, "height": 1, "width": 1, "seq": 1}, 6)


def test_auto_sizes_both_ends_to_two_times_the_scale():
    # rel_tol, not exact: fit_ladder rounds ids to 6 decimals, so a 1/30 s gap comes back
    # as 0.033333
    ids = torch.arange(30).float() / 30.0
    fitted = fit_ladder(ids)
    assert math.isclose(fitted["min_period"], 2 * (1 / 30.0), rel_tol=1e-4)
    assert math.isclose(fitted["max_period"], 2 * (29 / 30.0), rel_tol=1e-4)

    fastest = 2 * math.pi / fitted["min_period"]
    assert math.isclose(fastest * (1 / 30.0), math.pi, rel_tol=1e-4)


def test_the_two_fit_rules_can_never_cross():
    for n in range(2, 40):
        for scale in (1.0, 1 / 30.0, 7.5):
            fitted = fit_ladder(torch.arange(n).float() * scale)
            assert fitted["min_period"] <= fitted["max_period"]


def test_two_positions_give_a_single_period():
    fitted = fit_ladder(torch.tensor([0.0, 0.4]))
    assert fitted["min_period"] == fitted["max_period"] == pytest.approx(0.8)
    rope = RopeND(16, {"time": {"share": 1.0, **fitted}})
    assert torch.allclose(rope.freqs_time, rope.freqs_time[0])


def test_auto_slowest_dial_sweeps_exactly_half_a_turn():
    ids = torch.arange(30).float() / 30.0
    span = (ids[-1] - ids[0]).item()
    omega_slowest = 2 * math.pi / fit_ladder(ids)["max_period"]
    assert math.isclose(omega_slowest * span, math.pi, rel_tol=1e-4)
    assert math.isclose(math.cos(omega_slowest * span), -1.0, abs_tol=1e-6)


def test_auto_tracks_the_sampling_rate():
    fast = fit_ladder(torch.arange(30).float() / 30.0)
    slow = fit_ladder(torch.arange(30).float() / 2.5)
    assert math.isclose(slow["min_period"] / fast["min_period"], 12.0, rel_tol=1e-4)


def test_explicit_periods_win_over_the_fit():
    axes = axes_from_positions(
        TRUNK_POS, {"time": {"share": 1.0, "min_period": 0.05, "max_period": 8.0}}
    )
    assert axes["time"]["min_period"] == 0.05
    assert axes["time"]["max_period"] == 8.0


def test_either_end_can_be_auto_alone():
    axes = axes_from_positions(
        TRUNK_POS, {"seq": {"share": 1.0, "min_period": "auto", "max_period": 512.0}}
    )
    assert axes["seq"]["max_period"] == 512.0
    assert math.isclose(axes["seq"]["min_period"], 2.0, rel_tol=1e-6)  # 1-token gap


def test_periods_must_be_declared():
    with pytest.raises(AssertionError, match="period: auto"):
        axes_from_positions(TRUNK_POS, {"time": {"share": 1.0}})


def test_share_must_be_declared():
    with pytest.raises(AssertionError, match="share of the head dim"):
        axes_from_positions(TRUNK_POS, {"time": {"period": "auto"}})


def test_spec_is_required():
    with pytest.raises(AssertionError, match="no default axis set"):
        axes_from_positions(TRUNK_POS, None)


def test_period_only_takes_auto():
    with pytest.raises(AssertionError, match="only takes 'auto'"):
        axes_from_positions(TRUNK_POS, {"time": {"share": 1.0, "period": 4.0}})


def test_auto_and_explicit_together_is_an_error():
    with pytest.raises(AssertionError, match="pick one"):
        axes_from_positions(
            TRUNK_POS, {"time": {"share": 1.0, "period": "auto", "min_period": 0.05}}
        )


def test_auto_on_a_constant_axis_is_an_error():
    pos = make_pos_ids(torch.tensor([0.0, 0.4]))
    with pytest.raises(AssertionError, match="nothing to fit"):
        axes_from_positions(pos, {"height": {"share": 1.0, "period": "auto"}})
    forced = axes_from_positions(
        pos, {"height": {"share": 1.0, "min_period": 4.0, "max_period": 64.0}}
    )
    assert forced["height"]["max_period"] == 64.0


def test_declared_axes_partition_the_head_dim():
    for axes, head_dim in [(TRUNK_AXES, 64), (GRID_AXES, 64), (GRID_AXES, 32), (TRUNK_AXES, 16)]:
        dims = [d for d, _, _ in resolve_axes(axes, head_dim).values()]
        assert sum(dims) == head_dim
        assert all(d % 2 == 0 and d > 0 for d in dims)


def test_every_declared_axis_gets_live_dims():
    rope = RopeND(64, TRUNK_AXES)
    n = 8
    freqs = rope({"time": torch.linspace(0.0, 1.2, n), "seq": torch.arange(n).float()})
    assert freqs.shape == (n, 64)
    assert int((freqs.std(0) > 1e-9).sum()) == 64


def test_zero_positions_are_identity():
    rope = RopeND(32, TRUNK_AXES)
    x = torch.randn(1, 2, 5, 32)
    assert torch.allclose(
        apply_rotary_emb(rope({"time": torch.zeros(5), "seq": torch.zeros(5)}), x), x
    )


def test_missing_axis_rotates_by_nothing():
    rope = RopeND(32, TRUNK_AXES)
    split = rope.axes["time"][0]
    n, x = 5, torch.randn(1, 2, 5, 32)
    both = rope({"time": torch.linspace(0.1, 1.0, n), "seq": torch.zeros(n)})
    only_time = rope({"time": torch.linspace(0.1, 1.0, n)})
    assert torch.equal(both, only_time)
    out = apply_rotary_emb(only_time, x)
    assert not torch.allclose(out[..., :split], x[..., :split])
    assert torch.allclose(out[..., split:], x[..., split:])


def test_relative_positions_are_translation_invariant():
    """Shifting every position id must leave q.k unchanged."""
    rope = RopeND(32, TRUNK_AXES)
    q, k = torch.randn(1, 1, 4, 32), torch.randn(1, 1, 4, 32)
    t, s = torch.linspace(0.0, 3.0, 4), torch.arange(4).float()

    def logits(dt, ds):
        freqs = rope({"time": t + dt, "seq": s + ds})
        return apply_rotary_emb(freqs, q) @ apply_rotary_emb(freqs, k).transpose(-2, -1)

    # atol is float32's limit, not the property's: a 5 s shift puts freqs near 450 rad.
    # A genuine break would be O(1), not O(1e-4).
    assert torch.allclose(logits(0.0, 0.0), logits(5.0, 7.0), atol=1e-3)


def test_time_axis_resolves_one_action_step():
    """A 1/30 s step must produce a real rotation, not a rounding error."""
    rope = RopeND(64, TRUNK_AXES)
    n, step = 36, 1.0 / 30.0

    assert math.isclose(rope.freqs_time.max().item() * step, math.pi, rel_tol=0.05)

    x = torch.randn(64)
    x = (x / x.norm()).expand(1, 1, n, 64).contiguous()
    rotated = apply_rotary_emb(rope({"time": torch.arange(n).float() * step}), x)[0, 0]
    cos = rotated @ rotated[0]
    assert cos[1] < 0.99
    assert cos[n // 2] < cos[1]
