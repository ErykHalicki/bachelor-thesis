import pytest
import torch
from omegaconf import OmegaConf

from thesis.algorithms import build_algorithm
from thesis.algorithms.encoders import build_encoder, resolve_chains

ROPE = {"time": {"share": 0.4, "period": "auto"},
        "seq": {"share": 0.2, "period": "auto"},
        "height": {"share": 0.2, "min_period": 4.0, "max_period": 64.0},
        "width": {"share": 0.2, "min_period": 4.0, "max_period": 64.0}}


def test_resolve_chains_orders_root_first():
    chains = resolve_chains({"a": {}, "b": {"input": "a"}, "c": {"input": "b"}})
    assert chains == {"a": ["a"], "b": ["a", "b"], "c": ["a", "b", "c"]}


def test_resolve_chains_rejects_cycles():
    with pytest.raises(ValueError, match="no raw input"):
        resolve_chains({"a": {"input": "b"}, "b": {"input": "a"}})
    with pytest.raises(ValueError, match="no raw input"):
        resolve_chains({"a": {"input": "a"}})


def test_resolve_chains_rejects_unknown_input():
    with pytest.raises(ValueError, match="unknown encoder"):
        resolve_chains({"a": {"input": "ghost"}})


def test_frozen_encoder_has_no_grads_and_stays_eval():
    enc = build_encoder({"type": "linear", "in_dim": 8, "out_dim": 4, "frozen": True})
    assert all(not p.requires_grad for p in enc.parameters())
    enc.train(True)
    assert not enc.training


def test_heads_default_trainable():
    enc = build_encoder({"type": "linear", "in_dim": 8, "out_dim": 4})
    assert all(p.requires_grad for p in enc.parameters())
    enc.train(True)
    assert enc.training


def make_pool(**overrides):
    return build_encoder({"type": "attentive_pool", "in_dim": 8, "out_dim": 16,
                          "tokens_per_step": 9, "num_heads": 4, **overrides})


def test_pool_one_token_per_step():
    enc = make_pool()
    assert enc(torch.randn(2, 4 * 9, 8)).shape == (2, 4, 16)


def test_pool_num_queries_widens_the_step():
    enc = make_pool(num_queries=4)
    assert enc(torch.randn(2, 4 * 9, 8)).shape == (2, 16, 16)


def test_pool_rejects_token_count_that_is_not_whole_steps():
    with pytest.raises(ValueError, match="tokens_per_step"):
        make_pool()(torch.randn(2, 20, 8))


def test_pool_keeps_steps_independent():
    """A query pools one step, so scrambling one step must leave the others bit-identical:
    this is what lets a per_timestep sigreg term treat each step as its own sample bag.
    """
    enc = make_pool().eval()
    x = torch.randn(2, 4 * 9, 8)
    with torch.no_grad():
        base = enc(x)
        x[:, 9:18] = torch.randn(2, 9, 8)
        after = enc(x)
    assert not torch.allclose(base[:, 1], after[:, 1])
    assert torch.equal(base[:, [0, 2, 3]], after[:, [0, 2, 3]])


def test_pool_projects_into_the_trunk_width_without_a_linear_head():
    enc = build_encoder({"type": "attentive_pool", "in_dim": 1024, "out_dim": 256,
                         "tokens_per_step": 4})
    assert enc(torch.randn(2, 8, 1024)).shape == (2, 2, 256)


def test_pool_defaults_trainable_and_consumes_one_raw_row_per_step():
    enc = make_pool()
    assert not enc.frozen and all(p.requires_grad for p in enc.parameters())
    assert enc.raw_steps_per_index == 1


def make_resnet(**overrides):
    spec = {"type": "resnet", "variant": "resnet18", **overrides}
    try:
        return build_encoder(spec)
    except ImportError as err:
        pytest.skip(str(err))


def test_resnet_grid_tokens_per_frame():
    enc = make_resnet()
    frames = (torch.rand(2, 3, 3, 96, 96) * 255).to(torch.uint8)
    assert enc(frames).shape == (2, 27, 512)


def test_resnet_avg_pooling_one_vector_per_frame():
    enc = make_resnet(pooling="avg", img_size=64)
    frames = (torch.rand(2, 3, 3, 96, 96) * 255).to(torch.uint8)
    assert enc(frames).shape == (2, 3, 512)


def test_resnet_defaults_trainable_and_chains():
    enc = make_resnet()
    assert not enc.frozen and all(p.requires_grad for p in enc.parameters())
    chains = resolve_chains({"cnn": {"input": None}, "head": {"input": "cnn"}})
    assert chains["head"] == ["cnn", "head"]


def make_vjepa(**overrides):
    spec = {"type": "vjepa2", "arch": "vit_tiny", "crop_size": 32, **overrides}
    try:
        return build_encoder(spec)
    except ImportError as err:
        pytest.skip(str(err))


def test_vjepa_rejects_unknown_arch():
    make_vjepa()
    with pytest.raises(ValueError, match="unknown vjepa2 arch"):
        build_encoder({"type": "vjepa2", "arch": "vit_bogus", "crop_size": 32})


def test_vjepa_image_mode_one_latent_step_per_frame():
    enc = make_vjepa()
    assert enc.frozen and enc.raw_steps_per_index == 1
    frames = (torch.rand(2, 2, 3, 48, 64) * 255).to(torch.uint8)
    out = enc(frames)
    assert out.shape == (2, 8, 192)


def test_vjepa_tubelet_mode_pairs_raw_frames():
    enc = make_vjepa(frames_per_step=2)
    assert enc.raw_steps_per_index == 2
    frames = (torch.rand(2, 4, 3, 32, 32) * 255).to(torch.uint8)
    out = enc(frames)
    assert out.shape == (2, 8, 192)


def test_vjepa_accepts_single_image():
    enc = make_vjepa()
    out = enc((torch.rand(2, 3, 32, 32) * 255).to(torch.uint8))
    assert out.shape == (2, 4, 192)


def test_vjepa_pretrained_alias_needs_published_weights():
    enc = make_vjepa()
    # tiny has no published 2.1 release; the error must name the archs that do
    with pytest.raises(ValueError, match="vit_large"):
        enc._resolve_checkpoint("pretrained", "vit_tiny")
    # a local path passes through untouched (no network)
    assert enc._resolve_checkpoint("/some/local.pt", "vit_tiny") == "/some/local.pt"


def test_vjepa_checkpoint_roundtrip(tmp_path):
    src = make_vjepa()
    path = tmp_path / "vjepa21.pt"
    state = {f"module.{k}": v for k, v in src.backbone.state_dict().items()}
    torch.save({"ema_encoder": state}, path)
    dst = make_vjepa(checkpoint=str(path))
    for (ka, va), (kb, vb) in zip(
        src.backbone.state_dict().items(), dst.backbone.state_dict().items(), strict=True
    ):
        assert ka == kb and torch.equal(va, vb)


def chain_cfg():
    """Frozen vjepa2.1 -> trainable linear head -> sigreg on the head's output."""
    return OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": ROPE, "model_dim": 64, "depth": 1, "num_heads": 4, "mlp_ratio": 2.0},
        "num_flow_steps": 2,
        "encoders": {
            "vjepa": {"type": "vjepa2", "arch": "vit_tiny", "crop_size": 32, "frozen": True},
            "head": {"type": "linear", "input": "vjepa", "in_dim": 192, "out_dim": 32},
        },
        "conditioning": {
            "pixel_context": {
                "from": "observation.images.pixels", "encoder": "head",
                "dim": 32, "index": "-1..0", "fps": 10, "grid": [2, 2],
            },
        },
        "predict": {
            "action": {"from": "action", "type": "flow", "dim": 2, "index": "0..3", "fps": 10},
        },
        "losses": [
            {"type": "flow", "stream": "action", "weight": 1.0},
            {"type": "sigreg", "streams": ["pixel_context"], "weight": 0.1, "proj_dim": 8},
        ],
    })


def chain_algo(cfg=None):
    try:
        return build_algorithm(cfg if cfg is not None else chain_cfg())
    except ImportError as err:
        pytest.skip(str(err))


def make_chain_batch(b=4):
    torch.manual_seed(0)
    return {
        "observation.images.pixels": (torch.rand(b, 2, 3, 32, 32) * 255).to(torch.uint8),
        "action": torch.randn(b, 4, 2),
    }


def test_chain_trains_head_but_not_backbone():
    algo = chain_algo()
    out = algo.loss(make_chain_batch())
    assert any(k.startswith("loss/sigreg") for k in out)
    out["loss"].backward()

    head = algo.encoders["head"]
    assert head.proj.weight.grad is not None and head.proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in algo.encoders["vjepa"].parameters())

    summary = algo.summary()
    assert summary["params/trainable"] < summary["params/total"]


def test_chain_predict_returns_action_chunk():
    algo = chain_algo()
    obs = {"observation.images.pixels": (torch.rand(3, 2, 3, 32, 32) * 255).to(torch.uint8)}
    out = algo.predict(obs)
    assert out["action"].shape == (3, 4, 2)


def test_chain_cycle_in_config_raises():
    cfg = chain_cfg()
    cfg.encoders.vjepa["input"] = "head"
    with pytest.raises(ValueError, match="no raw input"):
        chain_algo(cfg)


def test_batchnorm_encoder_pins_unit_scale_over_every_axis():
    from omegaconf import OmegaConf
    from thesis.algorithms.encoders import build_encoder

    enc = build_encoder(OmegaConf.create({"type": "batchnorm", "dim": 6}))
    assert not any(p.requires_grad for p in enc.parameters())   # affine off by default
    x = 0.2 * torch.randn(16, 3, 6) + 5.0
    y = enc(x)
    assert y.shape == x.shape
    flat = y.reshape(-1, 6)
    assert torch.allclose(flat.mean(0), torch.zeros(6), atol=1e-4)
    assert torch.allclose(flat.std(0, unbiased=False), torch.ones(6), atol=1e-3)
    enc.eval()
    assert not torch.allclose(enc(x), y)   # eval reads the running statistics instead


def test_batchnorm_encoder_affine_is_opt_in():
    from omegaconf import OmegaConf
    from thesis.algorithms.encoders import build_encoder

    enc = build_encoder(OmegaConf.create({"type": "batchnorm", "dim": 4, "affine": True}))
    assert sum(p.numel() for p in enc.parameters() if p.requires_grad) == 8
