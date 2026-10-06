import pytest
import torch
from omegaconf import OmegaConf

from thesis.algorithms import build_algorithm
from thesis.algorithms.predictive_model import stitch_views
from thesis.utils.spec import spec_fields

ROPE = {"time": {"share": 0.4, "period": "auto"},
        "seq": {"share": 0.2, "period": "auto"},
        "height": {"share": 0.2, "min_period": 4.0, "max_period": 64.0},
        "width": {"share": 0.2, "min_period": 4.0, "max_period": 64.0}}
GRID_ROPE = {"time": {"share": 0.5, "min_period": 4.0, "max_period": 256.0},
             "height": {"share": 0.25, "period": "auto"},
             "width": {"share": 0.25, "period": "auto"}}


def _views():
    return (torch.randint(0, 256, (2, 3, 3, 376, 672), dtype=torch.uint8),
            torch.randint(0, 256, (2, 3, 3, 360, 640), dtype=torch.uint8))


def test_horizontal_stitch_widens_and_keeps_the_first_view_untouched():
    left, wrist = _views()
    out = stitch_views([left, wrist])
    assert out.shape == (2, 3, 3, 376, 672 * 2)
    assert out.dtype == torch.uint8
    assert torch.equal(out[..., :672], left)


def test_vertical_stitch_stacks():
    left, wrist = _views()
    assert stitch_views([left, wrist], "vertical").shape == (2, 3, 3, 376 * 2, 672)


def test_unknown_stitch_mode_is_an_error():
    left, wrist = _views()
    with pytest.raises(ValueError, match="unknown stitch"):
        stitch_views([left, wrist], "diagonal")


def test_flat_streams_stitch_on_the_feature_axis():
    """Two state columns become one wider vector per step, so a single encoder can read
    quantities the dataset stores separately (the A2A configs' pos + torq history)."""
    pos, torq = torch.zeros(2, 36, 7), torch.ones(2, 36, 7)
    out = stitch_views([pos, torq])
    assert out.shape == (2, 36, 14)
    assert torch.equal(out[..., :7], pos) and torch.equal(out[..., 7:], torq)


def test_flat_streams_must_share_a_window():
    with pytest.raises(ValueError, match="must share a window"):
        stitch_views([torch.zeros(2, 36, 7), torch.zeros(2, 12, 7)])


def test_a_flat_stream_cannot_stitch_vertically():
    """There is no second spatial axis to stack along; silently joining on features
    instead would make `vertical` a confusing alias for `horizontal`."""
    with pytest.raises(ValueError, match="needs a spatial axis"):
        stitch_views([torch.zeros(2, 36, 7), torch.zeros(2, 36, 7)], "vertical")


def test_a_stitched_state_history_is_one_conditioning_token():
    """The A2A state stream: 36 steps of two 7-dim columns -> 14 dims per step -> one
    token, the same summary shape the action codec makes of the action past. No decoder
    and no reconstruction term, so the encoder is trained only by the losses reading it.
    """
    cfg = OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": ROPE, "model_dim": 64, "depth": 2, "num_heads": 4, "mlp_ratio": 2.0,
                  "dropout": 0.0},
        "num_flow_steps": 2,
        "encoders": {"state_enc": {"type": "chunk_mlp", "in_dim": 14, "in_steps": 36,
                                   "out_dim": 64, "out_steps": 1, "hidden_dim": 32,
                                   "depth": 1}},
        "conditioning": {
            "state_context": {
                "from": ["observation.state.pos", "observation.state.torq"],
                "encoder": "state_enc", "dim": 64, "index": "0", "raw_index": "-35..0",
                "fps": 30, "grid": [1, 1], "block_size": 1,
            },
        },
        "predict": {"action": {"from": "action", "type": "flow", "dim": 7,
                               "index": "0..3", "fps": 30}},
    })
    algo = build_algorithm(cfg)
    assert sorted(algo.field_offsets) == ["observation.state.pos", "observation.state.torq"]
    assert algo.field_offsets["observation.state.pos"] == list(range(-35, 1))
    assert algo.predictor.num_tokens == 1 + 4

    batch = {
        "observation.state.pos": torch.randn(2, 36, 7),
        "observation.state.torq": torch.randn(2, 36, 7),
        "action": torch.randn(2, 4, 7),
    }
    assert algo.loss(batch)["loss"].isfinite()

    # the velocity head is zero-initialized (DiT), so a fresh model predicts exactly
    # zero and every input looks identical; step 0 measures the init, not the wiring
    opt = torch.optim.AdamW(algo.parameters(), lr=1e-2)
    for step in range(3):
        opt.zero_grad()
        torch.manual_seed(step)
        algo.loss(batch)["loss"].backward()
        opt.step()

    grads = [p.grad for p in algo.encoders["state_enc"].parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), (
        "the state encoder has no reconstruction term, so the flow loss is the only "
        "thing that can train it -- if no gradient reaches it, nothing does"
    )

    algo.eval()
    seen = []
    handle = algo.predictor.register_forward_hook(
        lambda module, args, out: seen.append(out["action"].detach().clone())
    )
    for seed in (1, 2):
        torch.manual_seed(seed)
        state = {
            "observation.state.pos": torch.randn(2, 36, 7),
            "observation.state.torq": torch.randn(2, 36, 7),
        }
        torch.manual_seed(0)
        algo.loss({**batch, **state})
    handle.remove()
    assert not torch.allclose(seen[0], seen[1]), (
        "the state history is conditioning nothing -- the action velocity is the same "
        "whatever the arm was doing"
    )


def test_spec_fields_reads_one_or_many():
    assert spec_fields("cam", {}) == ["cam"]
    assert spec_fields("cam", {"from": "a"}) == ["a"]
    assert spec_fields("cam", OmegaConf.create({"from": ["a", "b"]})) == ["a", "b"]


def _algo(**camera):
    cfg = OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": ROPE, "model_dim": 64, "depth": 2, "num_heads": 4, "mlp_ratio": 2.0, "dropout": 0.0},
        "num_flow_steps": 2,
        "encoders": {"vid": {"type": "vit", "rope": GRID_ROPE, "patch_size": 16, "dim": 64, "depth": 1,
                             "num_heads": 4, "img_size": [32, 64], "frames": 2}},
        "conditioning": {"cams": {"encoder": "vid", "dim": 64, "index": "-1..0", "fps": 5,
                                  "grid": [2, 4], **camera}},
        "predict": {"action": {"from": "action", "type": "flow", "dim": 7, "index": "0..3",
                               "fps": 30}},
    })
    return build_algorithm(cfg)


def test_a_stitched_stream_encodes_both_views_as_one():
    algo = _algo(**{"from": ["cam.left", "cam.right"], "stitch": "horizontal"})
    assert sorted(algo.field_offsets) == ["cam.left", "cam.right"]
    assert algo.predictor.num_tokens == 2 * 8 + 4

    batch = {
        "cam.left": torch.randint(0, 256, (2, 2, 3, 32, 32), dtype=torch.uint8),
        "cam.right": torch.randint(0, 256, (2, 2, 3, 32, 32), dtype=torch.uint8),
        "action": torch.randn(2, 4, 7),
    }
    assert algo.loss(batch)["loss"].isfinite()


def test_stitched_views_must_share_a_window():
    cfg = OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": ROPE, "model_dim": 64, "depth": 2, "num_heads": 4, "mlp_ratio": 2.0, "dropout": 0.0},
        "num_flow_steps": 2,
        "encoders": {"vid": {"type": "vit", "rope": GRID_ROPE, "patch_size": 16, "dim": 64, "depth": 1,
                             "num_heads": 4, "img_size": [32, 64], "frames": 2}},
        "conditioning": {
            "cams": {"from": ["cam.left", "cam.right"], "encoder": "vid", "dim": 64,
                     "index": "-1..0", "fps": 5, "grid": [2, 4]},
            # a second entry pulls cam.left over a wider window, so the two views no longer
            # line up row for row and one selector cannot serve both
            "extra": {"from": "cam.left", "via": "cross", "dim": 64, "index": "-3..0",
                      "fps": 5, "encoder": "vid"},
        },
        "predict": {"action": {"from": "action", "type": "flow", "dim": 7, "index": "0..3",
                               "fps": 30}},
    })
    with pytest.raises(ValueError, match="different row layouts"):
        build_algorithm(cfg)
