import pytest
import torch
from omegaconf import OmegaConf

from thesis.algorithms import build_algorithm
from thesis.algorithms.encoders import ViTEncoder

GRID_ROPE = {"time": {"share": 0.5, "min_period": 4.0, "max_period": 256.0},
             "height": {"share": 0.25, "period": "auto"},
             "width": {"share": 0.25, "period": "auto"}}

ROPE = {"time": {"share": 0.4, "period": "auto"},
        "seq": {"share": 0.2, "period": "auto"},
        "height": {"share": 0.2, "min_period": 4.0, "max_period": 64.0},
        "width": {"share": 0.2, "min_period": 4.0, "max_period": 64.0}}

RAW_INDEX = "-54, -48, -42, -36, -30, -24, -18, -12, -6, 0"


@pytest.fixture
def tubelet_encoder(monkeypatch):
    """A from-scratch ViT standing in for V-JEPA's contract -- two raw frames per index
    step, collapsed to one latent step -- so these tests exercise the tubelet path without
    needing the vjepa2 submodule or its weights.
    """
    forward = ViTEncoder.forward
    monkeypatch.setattr(ViTEncoder, "raw_steps_per_index", 2)
    monkeypatch.setattr(ViTEncoder, "forward", lambda self, frames: forward(self, frames[:, 0::2]))


def _algo(**overrides):
    cfg = OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": ROPE, "model_dim": 64, "depth": 2, "num_heads": 4, "mlp_ratio": 2.0, "dropout": 0.0},
        "num_flow_steps": 2,
        "encoders": {"vid": {"type": "vit", "rope": GRID_ROPE, "img_size": 32, "frames": 10, "patch_size": 16,
                             "dim": 64, "depth": 1, "num_heads": 4}},
        "conditioning": {"video": {"from": "observation.images.cam", "encoder": "vid",
                                   "dim": 64, "index": "-4..0", "fps": 2.5, "grid": [2, 2],
                                   **overrides}},
        "predict": {"action": {"from": "action", "type": "flow", "dim": 7, "index": "0..39",
                               "fps": 30}},
    })
    return build_algorithm(cfg)


def test_multi_row_index_steps_need_a_raw_index(tubelet_encoder):
    with pytest.raises(ValueError, match="consumes 2 raw rows per index step"):
        _algo()


def test_raw_index_row_count_must_match_the_index_steps(tubelet_encoder):
    with pytest.raises(ValueError, match="naming the 10 rows"):
        _algo(raw_index="-6, 0")


def test_raw_index_drives_the_selector_and_the_rollout_geometry(tubelet_encoder):
    algo = _algo(raw_index=RAW_INDEX)
    assert algo._selectors["video"] == slice(0, 10)
    offsets = [-54, -48, -42, -36, -30, -24, -18, -12, -6, 0]
    assert algo.field_offsets["observation.images.cam"] == offsets
    assert algo.obs_len == 55
    assert algo.chunk_len == 40


def test_a_raw_index_equal_to_index_changes_nothing():
    plain = _algo()
    explicit = _algo(raw_index="-4..0")
    assert plain._selectors == explicit._selectors
    assert plain.field_offsets == explicit.field_offsets
    assert plain.obs_len == explicit.obs_len


def test_grid_fanout_still_expands_rows_without_a_raw_index():
    # a passthrough token field carries grid-area rows per step: a SPATIAL fan-out,
    # not extra time steps, so it needs no raw_index
    cfg = OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": ROPE, "model_dim": 64, "depth": 1, "num_heads": 2, "mlp_ratio": 2.0, "dropout": 0.0},
        "num_flow_steps": 2,
        "conditioning": {"latents": {"from": "observation.latents", "dim": 8,
                                     "index": "-2..-1", "fps": 10, "grid": [2, 2]}},
        "predict": {"action": {"from": "action", "type": "flow", "dim": 4, "index": "0..3",
                               "fps": 10}},
    })
    algo = build_algorithm(cfg)
    assert algo._selectors["latents"] == slice(0, 8)


def test_the_model_trains_on_the_rows_raw_index_asked_for(tubelet_encoder):
    algo = _algo(raw_index=RAW_INDEX)
    batch = {
        "observation.images.cam": torch.randint(0, 256, (2, 10, 3, 32, 32), dtype=torch.uint8),
        "action": torch.randn(2, 40, 7),
    }
    assert algo.loss(batch)["loss"].isfinite()
    assert algo.predictor.num_tokens == 5 * 4 + 40
