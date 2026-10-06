"""A2A flow sources (per-stream `source:` blocks), the chunk_mlp codec, and the ae_recon
term. The seed replaces the Gaussian x0 of a flow stream -- training interpolates from it,
rollout integrates from it -- so the tests pin down (1) the seed is exactly the configured
history window / context latent, (2) rollout consumes it instead of drawing noise, (3) a
codec-latent action stream still hands eval a raw-space chunk, and (4) the config
validation that keeps a seed from leaking the future.
"""

import pytest
import torch
from omegaconf import OmegaConf

from thesis.algorithms import build_algorithm
from thesis.algorithms.encoders import build_encoder
from thesis.utils.spec import flow_source_entries

ROPE = {"time": {"share": 0.75, "period": "auto"},
        "seq": {"share": 0.25, "period": "auto"}}

ENCODER = {
    "type": "vit", "pooling": "cls", "dim": 32, "patch_size": 14, "depth": 2,
    "num_heads": 4, "mlp_ratio": 2.0, "img_size": 28,
}
PROJ = {"type": "mlp", "input": "vit", "in_dim": 32, "hidden_dim": 64, "out_dim": 32,
        "norm": "batch"}
MODEL = {"rope": ROPE, "model_dim": 32, "depth": 2, "num_heads": 4, "dim_head": 16, "mlp_ratio": 2.0}


def wam_cfg():
    """Two seeded flow streams: the action chunk from its own previous chunk (field mode),
    the future latents from the last context latent (stream mode, persistence init)."""
    return OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": dict(MODEL),
        "num_flow_steps": 2,
        "encoders": {"vit": dict(ENCODER), "vit_proj": dict(PROJ)},
        "conditioning": {
            "video_context": {
                "from": "observation.images.pixels", "encoder": "vit_proj", "via": "tokens",
                "dim": 32, "index": "-2..0", "grid": [1, 1],
            },
            "prev_action": {"from": "action", "via": "tokens", "dim": 2, "index": "-1"},
        },
        "predict": {
            "future_video": {
                "from": "observation.images.pixels", "encoder": "vit_proj", "type": "flow",
                "dim": 32, "index": "1..2", "grid": [1, 1],
                "source": {"stream": "video_context", "rows": "-1", "fill": "repeat_last",
                           "noise_std": 0.0},
            },
            "action_chunk": {
                "from": "action", "type": "flow", "dim": 2, "index": "0..3", "fps": 1.0,
                "source": {"from": "action", "index": "-4..-1", "noise_std": 0.0},
            },
        },
        "losses": [
            {"type": "flow", "stream": "action_chunk", "weight": 1.0},
            {"type": "flow", "stream": "future_video", "weight": 1.0},
        ],
    })


def ae_cfg():
    """The action chunk flows in a chunk_mlp codec latent: encoder in, decoder out,
    the previous chunk (through the same encoder) as the source, L1 recon grounding."""
    return OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": dict(MODEL),
        "num_flow_steps": 2,
        "encoders": {
            "vit": dict(ENCODER), "vit_proj": dict(PROJ),
            "act_enc": {"type": "chunk_mlp", "in_dim": 2, "in_steps": 4, "out_dim": 32,
                        "out_steps": 1, "hidden_dim": 32, "depth": 1},
            "act_dec": {"type": "chunk_mlp", "in_dim": 32, "in_steps": 1, "out_dim": 2,
                        "out_steps": 4, "hidden_dim": 32, "depth": 1},
        },
        "conditioning": {
            "video_context": {
                "from": "observation.images.pixels", "encoder": "vit_proj", "via": "tokens",
                "dim": 32, "index": "-2..0", "grid": [1, 1],
            },
            "prev_action": {"from": "action", "via": "tokens", "dim": 2, "index": "-1"},
        },
        "predict": {
            "action_chunk": {
                "from": "action", "type": "flow", "encoder": "act_enc", "decoder": "act_dec",
                "dim": 32, "index": "1", "raw_index": "0..3", "chunk_len": 4, "fps": 1.0,
                "source": {"from": "action", "index": "-4..-1", "encoder": "act_enc",
                           "noise_std": 0.0},
            },
        },
        "losses": [
            {"type": "flow", "stream": "action_chunk", "weight": 1.0},
            {"type": "ae_recon", "stream": "action_chunk", "decoder": "act_dec",
             "weight": 1.0, "norm": "l1"},
        ],
    })


def make_batch(b=4, pixel_rows=5, action_rows=8):
    torch.manual_seed(0)
    return {
        "observation.images.pixels": (torch.rand(b, pixel_rows, 3, 96, 96) * 255).to(torch.uint8),
        "action": torch.randn(b, action_rows, 2),
    }


def make_obs(b=2):
    """History only: pixels at -2..0, actions at -4..-1 (source window plus prev_action)."""
    torch.manual_seed(1)
    return {
        "observation.images.pixels": (torch.rand(b, 3, 3, 96, 96) * 255).to(torch.uint8),
        "action": torch.randn(b, 4, 2),
    }


def test_chunk_mlp_shapes_and_chunk_independence():
    enc = build_encoder({"type": "chunk_mlp", "in_dim": 2, "in_steps": 4, "out_dim": 8,
                         "out_steps": 3, "hidden_dim": 16, "depth": 1})
    x = torch.randn(5, 8, 2)
    z = enc(x)
    assert z.shape == (5, 6, 8)
    x2 = x.clone()
    x2[:, 4:] += 1.0
    assert torch.allclose(enc(x2)[:, :3], z[:, :3])
    assert enc.raw_steps_per_index == 4
    with pytest.raises(ValueError, match="in_steps"):
        enc(torch.randn(5, 7, 2))


def test_reshape_is_an_exact_bijection_between_rows_and_features():
    flat = build_encoder({"type": "reshape", "in_dim": 2, "in_steps": 4, "out_dim": 8,
                          "out_steps": 1})
    unflat = build_encoder({"type": "reshape", "in_dim": 8, "in_steps": 1, "out_dim": 2,
                            "out_steps": 4})
    x = torch.randn(5, 8, 2)
    z = flat(x)
    assert z.shape == (5, 2, 8)
    assert torch.equal(unflat(z), x)
    assert not list(flat.parameters())
    assert flat.raw_steps_per_index == 4
    with pytest.raises(ValueError, match="in_steps"):
        flat(torch.randn(5, 7, 2))
    with pytest.raises(ValueError, match="in_dim"):
        flat(torch.randn(5, 8, 3))
    with pytest.raises(ValueError, match="value count"):
        build_encoder({"type": "reshape", "in_dim": 2, "in_steps": 4, "out_dim": 7,
                       "out_steps": 1})


def test_field_source_seed_is_the_previous_chunk():
    algo = build_algorithm(wam_cfg())
    batch = make_batch()
    clean, sources, cond, hidden = algo._condition(batch)
    seeds = algo.build_flow_seeds(batch, {**clean, **sources, **cond, **hidden}, training=True)
    assert torch.equal(seeds["action_chunk"], batch["action"][:, :4].float())
    ctx = clean["video_context"]
    assert torch.equal(seeds["future_video"], ctx[:, -1:].expand(-1, 2, -1))


def test_source_noise():
    cfg = wam_cfg()
    cfg.predict.action_chunk.source.noise_std = 0.5
    algo = build_algorithm(cfg)
    batch = make_batch()
    enc = algo._condition(batch)
    encoded = {**enc[0], **enc[1], **enc[2], **enc[3]}
    a = algo.build_flow_seeds(batch, encoded, training=True)["action_chunk"]
    b = algo.build_flow_seeds(batch, encoded, training=True)["action_chunk"]
    assert not torch.equal(a, b)
    # eval seeds carry the same noise the training seeds did
    a = algo.build_flow_seeds(batch, encoded, training=False)["action_chunk"]
    b = algo.build_flow_seeds(batch, encoded, training=False)["action_chunk"]
    assert not torch.equal(a, b)
    # the removed `noise_at_eval` off-switch is a hard error, not a silent no-op
    cfg.predict.action_chunk.source.noise_at_eval = False
    algo = build_algorithm(cfg)
    with pytest.raises(ValueError, match="noise_at_eval"):
        algo.build_flow_seeds(batch, encoded, training=False)


def test_rollout_integrates_from_the_seed_not_noise():
    algo = build_algorithm(wam_cfg())
    obs = make_obs()
    out1 = algo.predict(obs)
    out2 = algo.predict(obs)
    assert torch.equal(out1["action_chunk"], out2["action_chunk"])
    assert torch.equal(out1["future_video"], out2["future_video"])


def test_loss_runs_and_seed_gradients_reach_the_encoder():
    algo = build_algorithm(wam_cfg())
    out = algo.loss(make_batch())
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in algo.encoders["vit_proj"].parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_codec_latent_action_stream_decodes_to_raw_chunks():
    algo = build_algorithm(ae_cfg())
    assert algo.chunk_len == 4 and algo.action_dim == 2
    out = algo.loss(make_batch())
    assert "loss/ae_action_chunk" in out
    out["loss"].backward()
    for name in ("act_enc", "act_dec"):
        grads = [p.grad for p in algo.encoders[name].parameters()]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), name
    pred = algo.predict(make_obs())["action_chunk"]
    assert pred.shape == (2, 4, 2)


def test_flow_loss_reaches_a_jointly_trained_target_encoder():
    """Nothing on this path is stop-gradded: the flow loss shapes the encoder that produces
    its own target x1, and collapse is held off by ae_recon/sigreg rather than by cutting
    the gradient."""
    cfg = wam_cfg()
    del cfg.predict.action_chunk
    cfg.losses = [{"type": "flow", "stream": "future_video", "weight": 1.0}]
    cfg.predict.future_video.source = None
    cfg.encoders.future_proj = dict(PROJ)
    cfg.predict.future_video.encoder = "future_proj"
    torch.manual_seed(0)
    algo = build_algorithm(cfg)
    algo.loss(make_batch())["loss"].backward()
    assert sum(
        p.grad.abs().sum().item()
        for p in algo.encoders["future_proj"].parameters() if p.grad is not None
    ) > 0


def test_source_validation():
    with pytest.raises(ValueError, match="last"):
        flow_source_entries({"a": {"type": "flow", "from": "action",
                                   "source": {"from": "action", "index": "last"}}})
    cfg = wam_cfg()
    cfg.predict.action_chunk.source.index = "-3..0"
    with pytest.raises(ValueError, match="does not know"):
        build_algorithm(cfg)
    cfg = wam_cfg()
    cfg.predict.future_video.source = {"stream": "action_chunk"}
    with pytest.raises(ValueError, match="conditioning"):
        build_algorithm(cfg)


def test_frozen_codec_stream_is_not_cache_eligible():
    """freeze_world_model freezes every encoder, which must not turn the action codec
    into an encoding-cache candidate: the cache holds visual windows, and the dataset
    rejects a non-visual cache field."""
    cfg = ae_cfg()
    cfg.encoders.act_enc.frozen = True
    cfg.encoders.vit.frozen = True
    algo = build_algorithm(cfg)
    cacheable = algo.cacheable_streams()
    assert "action_chunk" not in cacheable
    assert "video_context" in cacheable


def test_dataset_union_covers_source_windows():
    entries = flow_source_entries(wam_cfg().predict)
    assert entries == {"action_chunk.source": {"from": "action", "index": "-4..-1"}}



def hidden_ae_cfg():
    """ae_cfg with the seed moved to STREAM MODE through a `via: null` hidden conditioning
    stream -- the b601 A2A layout in miniature. The hidden stream encodes the same
    -4..-1 window through the same act_enc as ae_cfg's field-mode source, so the two
    configs must build the identical seed."""
    cfg = ae_cfg()
    cfg.conditioning.prev_action.role = "action"
    cfg.conditioning.action_history = {
        "from": "action", "encoder": "act_enc", "via": None, "dim": 32,
        "index": "-1", "raw_index": "-4..-1", "fps": 1.0,
    }
    cfg.predict.action_chunk.source = {"stream": "action_history", "noise_std": 0.0}
    return cfg


def test_via_null_routes_to_hidden_and_lands_in_latent_but_not_the_trunk():
    algo = build_algorithm(hidden_ae_cfg())
    batch = make_batch()
    clean, sources, cond, hidden = algo._condition(batch)
    assert set(hidden) == {"action_history"}
    assert all("action_history" not in d for d in (clean, sources, cond))
    ctx = algo._build_context(batch)
    assert "action_history" in ctx["latent"]
    assert flow_source_entries(hidden_ae_cfg().predict) == {}


def test_hidden_stream_changes_nothing_but_the_seed_provenance():
    """The b601 requirement in miniature: the history latent enters ONLY as the x0 seed.
    Same init seed, same batch -> the field-mode and hidden-stream configs must produce
    the same seed, the same predictions, and trunks with identical parameter sets (the
    hidden stream allocates no token slot)."""
    torch.manual_seed(0)
    field = build_algorithm(ae_cfg())
    torch.manual_seed(0)
    hid = build_algorithm(hidden_ae_cfg())
    f_names = [n for n, _ in field.named_parameters()]
    h_names = [n for n, _ in hid.named_parameters()]
    assert f_names == h_names
    batch = make_batch()
    torch.manual_seed(7)
    ctx_f = field._build_context(batch)
    torch.manual_seed(7)
    ctx_h = hid._build_context(batch)
    for name in ctx_f["pred"]:
        assert torch.equal(ctx_f["pred"][name], ctx_h["pred"][name])


def test_sigreg_bags_the_hidden_history_with_the_flow_latent():
    """The sigact term: one pooled bag over [history latent, future latent], reading the
    hidden stream from ctx["latent"] -- the wiring no field-mode source could offer."""
    cfg = hidden_ae_cfg()
    cfg.losses.append({"type": "sigreg", "weight": 0.1, "concat": "temporal",
                       "streams": ["action_history", "action_chunk"]})
    algo = build_algorithm(cfg)
    out = algo.loss(make_batch())
    assert any(k.startswith("loss/sigreg_action_history") for k in out)
    out["loss"].backward()
    grads = [p.grad.abs().sum() for p in algo.encoders["act_enc"].parameters()
             if p.grad is not None]
    assert sum(grads) > 0
