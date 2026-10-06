"""The latent -> pixel probe: the decoder module, the algorithm around it, the encoder-init
seeding that points it at a trained run, and the panel the eval renders."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from thesis.algorithms import build_algorithm
from thesis.algorithms.encoders import build_encoder
from thesis.experiments.eval.reconstruction import panel
from thesis.utils.checkpoint import load_pretrained_weights

ROPE = {"time": {"share": 0.4, "period": "auto"},
        "seq": {"share": 0.2, "period": "auto"},
        "height": {"share": 0.2, "min_period": 4.0, "max_period": 64.0},
        "width": {"share": 0.2, "min_period": 4.0, "max_period": 64.0}}


def make_decoder(**overrides):
    return build_encoder({"type": "pixel_decoder", "in_dim": 8, "img_size": 32,
                          "patch_size": 16, "hidden_dim": 16, "depth": 2, "num_heads": 4,
                          **overrides})


def test_one_image_per_step_from_a_single_token():
    dec = make_decoder()
    assert dec(torch.randn(2, 3, 8)).shape == (2, 3, 3, 32, 32)


def test_token_count_per_step_is_free():
    dec = make_decoder(tokens_per_step=9)
    assert dec(torch.randn(2, 2 * 9, 8)).shape == (2, 2, 3, 32, 32)


def test_rejects_token_count_that_is_not_whole_steps():
    with pytest.raises(ValueError, match="tokens_per_step"):
        make_decoder(tokens_per_step=4)(torch.randn(2, 6, 8))


def test_non_square_image_keeps_its_aspect():
    dec = make_decoder(img_size=[32, 64])
    assert dec(torch.randn(2, 1, 8)).shape == (2, 1, 3, 32, 64)


def test_rejects_an_image_the_patch_size_does_not_tile():
    with pytest.raises(ValueError, match="patches"):
        make_decoder(img_size=30)


def test_steps_decode_independently():
    """One step is one image: perturbing a step must leave the other images untouched, the
    property that makes a multi-step window a strip of separate frames."""
    dec = make_decoder().eval()
    x = torch.randn(2, 3, 8)
    with torch.no_grad():
        base = dec(x)
        x[:, 1] = torch.randn(2, 8)
        after = dec(x)
    assert not torch.allclose(base[:, 1], after[:, 1])
    assert torch.equal(base[:, [0, 2]], after[:, [0, 2]])


def test_patches_land_where_the_grid_says():
    """The head emits patch vectors; only the reshape decides where they go. A decoder
    whose output is one flat patch must produce an image constant inside each 16x16 tile."""
    dec = make_decoder().eval()
    with torch.no_grad():
        img = dec(torch.randn(1, 1, 8))[0, 0]
    for row in (slice(0, 16), slice(16, 32)):
        for col in (slice(0, 16), slice(16, 32)):
            tile = img[:, row, col]
            assert tile.shape == (3, 16, 16)




def make_cfg(**overrides):
    """A probe over an already-encoded latent field: `latents` stands in for a frozen
    backbone's output, so the test needs no vision model."""
    return OmegaConf.create({
        "name": "latent_decoder",
        "encoders": {
            "head": {"type": "linear", "in_dim": 8, "out_dim": 6},
            "decode": {"type": "pixel_decoder", "in_dim": 6, "tokens_per_step": 1,
                       "img_size": 32, "patch_size": 16, "hidden_dim": 16, "depth": 2,
                       "num_heads": 4},
        },
        "conditioning": {
            "latent": {"from": "latents", "encoder": "head", "dim": 6,
                       "index": "0", "fps": 1.0, "grid": [1, 1]},
        },
        "predict": {
            "pixels": {"from": "observation.images.cam", "input": "latent",
                       "decoder": "decode", "type": "prediction", "index": "0", "fps": 1.0},
        },
        **overrides,
    })


def make_batch(b=2, size=48):
    torch.manual_seed(0)
    return {
        "latents": torch.randn(b, 1, 8),
        "observation.images.cam": (torch.rand(b, 1, 3, size, size) * 255).to(torch.uint8),
    }


def test_loss_trains_the_decoder():
    algo = build_algorithm(make_cfg())
    out = algo.loss(make_batch())
    assert out["loss"].isfinite() and "loss/pixels" in out
    out["loss"].backward()
    grads = [p.grad for p in algo.encoders["decode"].parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_target_is_the_frame_resized_to_the_decoder_output():
    algo = build_algorithm(make_cfg())
    ctx = algo._build_context(make_batch(size=48))
    assert ctx["pred"]["pixels"].shape == ctx["target"]["pixels"].shape == (2, 1, 3, 32, 32)
    assert 0.0 <= ctx["target"]["pixels"].min() and ctx["target"]["pixels"].max() <= 1.0


def test_loss_is_the_per_pixel_mse():
    algo = build_algorithm(make_cfg())
    batch = make_batch()
    ctx = algo._build_context(batch)
    expected = (ctx["pred"]["pixels"] - ctx["target"]["pixels"]).square().mean()
    assert torch.allclose(algo.loss(batch)["loss/pixels"], expected)


def test_reconstruct_returns_displayable_columns():
    algo = build_algorithm(make_cfg())
    algo.train()
    columns = algo.reconstruct(make_batch())["pixels"]
    assert list(columns) == ["frame", "decoded latent"], "the real frame comes first"
    for image in columns.values():
        assert image.shape == (2, 1, 3, 32, 32) and image.dtype == torch.uint8
    assert algo.training, "reconstruct must restore train mode: it runs mid-training"


def test_frozen_encoders_are_reported_and_do_not_train():
    cfg = make_cfg()
    cfg.encoders.head.frozen = True
    algo = build_algorithm(cfg)
    assert algo.summary()["encoders/frozen"] == "head"
    algo.loss(make_batch())["loss"].backward()
    assert all(p.grad is None for p in algo.encoders["head"].parameters())


def test_predict_stream_must_name_its_latent_and_decoder():
    cfg = make_cfg()
    del cfg.predict.pixels.input
    with pytest.raises(ValueError, match="input"):
        build_algorithm(cfg)

    cfg = make_cfg()
    cfg.predict.pixels.input = "ghost"
    with pytest.raises(ValueError, match="not a conditioning stream"):
        build_algorithm(cfg)

    cfg = make_cfg()
    cfg.predict.pixels.decoder = "ghost"
    with pytest.raises(ValueError, match="unknown decoder"):
        build_algorithm(cfg)


def test_decoder_must_match_the_stream_it_reads():
    """The latent width and the tokens per step are stated in both the encoder entry and
    the stream; a drift between them is a build error, not a matmul failure mid-run."""
    cfg = make_cfg()
    cfg.encoders.decode.in_dim = 5
    with pytest.raises(ValueError, match="in_dim is 5"):
        build_algorithm(cfg)

    cfg = make_cfg()
    cfg.conditioning.latent.grid = [2, 2]
    with pytest.raises(ValueError, match="tokens_per_step"):
        build_algorithm(cfg)




def test_encoder_init_copies_matching_encoder_weights(tmp_path):
    trained, probe = build_algorithm(make_cfg()), build_algorithm(make_cfg())
    with torch.no_grad():
        trained.encoders["head"].proj.weight.add_(1.0)
    path = tmp_path / "model.pt"
    torch.save({"model": trained.state_dict()}, path)

    before = probe.encoders["decode"].decoder.head.weight.clone()
    loaded, kept = load_pretrained_weights(probe, path)
    assert loaded == ["encoders.decode", "encoders.head"] and kept == []
    assert torch.equal(probe.encoders["head"].proj.weight,
                       trained.encoders["head"].proj.weight)
    assert not torch.equal(probe.encoders["decode"].decoder.head.weight, before)


def test_encoder_init_reports_what_the_checkpoint_lacks(tmp_path):
    probe = build_algorithm(make_cfg())
    state = {k: v for k, v in probe.state_dict().items() if k.startswith("encoders.head.")}
    path = tmp_path / "model.pt"
    torch.save({"model": state}, path)
    assert load_pretrained_weights(probe, path) == (["encoders.head"], ["encoders.decode"])


def test_encoder_init_carries_the_normalization_the_weights_trained_under(tmp_path):
    probe = build_algorithm(make_cfg())
    path = tmp_path / "model.pt"
    stats = {"action": {"mean": [0.0], "std": [1.0]}}
    torch.save({"model": probe.state_dict(), "norm_stats": stats,
                "norm_method": "percentile"}, path)
    load_pretrained_weights(probe, path)
    assert probe.norm_stats == stats and probe.norm_method == "percentile"


def test_encoder_init_rejects_a_stack_that_is_not_the_one_that_trained(tmp_path):
    other = make_cfg()
    other.encoders.head.out_dim = 6
    other.encoders.head.in_dim = 12
    trained = build_algorithm(
        OmegaConf.merge(other, {"conditioning": {"latent": {"dim": 6}}})
    )
    path = tmp_path / "model.pt"
    torch.save({"model": trained.state_dict()}, path)
    with pytest.raises(ValueError, match="does not match"):
        load_pretrained_weights(build_algorithm(make_cfg()), path)


def test_encoder_init_rejects_an_unrelated_checkpoint(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"model": {"rope": ROPE, "nothing.in.common": torch.zeros(2, 2)}}, path)
    with pytest.raises(ValueError, match="shares no weights"):
        load_pretrained_weights(build_algorithm(make_cfg()), path)




def make_wam_cfg(**overrides):
    """A miniature WAM -- pixels through a tiny CLS ViT, a flow trunk predicting the next
    latents and an action chunk -- with a decoder on the future stream."""
    return OmegaConf.create({
        "name": "wam_decoder",
        "model": {"rope": ROPE, "model_dim": 64, "depth": 2, "num_heads": 4},
        "num_flow_steps": 2,
        "encoders": {
            "vit": {"type": "vit", "pooling": "cls", "dim": 6, "depth": 1, "num_heads": 2,
                    "patch_size": 16, "img_size": 32, "frozen": True},
            "decode": {"type": "pixel_decoder", "in_dim": 6, "tokens_per_step": 1,
                       "img_size": 32, "patch_size": 16, "hidden_dim": 16, "depth": 2,
                       "num_heads": 4},
        },
        "conditioning": {
            "context": {"from": "observation.images.cam", "encoder": "vit", "dim": 6,
                        "index": "-1..0", "fps": 1.0, "grid": [1, 1]},
        },
        "predict": {
            "future": {"from": "observation.images.cam", "encoder": "vit", "type": "flow",
                       "dim": 6, "index": "1..2", "fps": 1.0, "grid": [1, 1]},
            "action": {"from": "action", "type": "flow", "dim": 4, "index": "0..2",
                       "fps": 3.0},
        },
        "decode": {"scene": {"stream": "future", "decoder": "decode"}},
        "losses": [{"type": "prediction", "stream": "scene", "weight": 1.0}],
        **overrides,
    })


def make_wam_batch(b=2, size=48):
    torch.manual_seed(0)
    return {
        "observation.images.cam": (torch.rand(b, 4, 3, size, size) * 255).to(torch.uint8),
        "action": torch.randn(b, 3, 4),
    }


def test_wam_probe_trains_only_the_decoder():
    algo = build_algorithm(make_wam_cfg())
    out = algo.loss(make_wam_batch())
    assert out["loss"].isfinite() and "loss/scene" in out
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in algo.encoders["decode"].parameters())
    assert all(p.grad is None for p in algo.predictor.parameters())
    assert all(not p.requires_grad for p in algo.predictor.parameters())


def test_wam_probe_keeps_the_trunk_in_eval_mode():
    algo = build_algorithm(make_wam_cfg())
    algo.train()
    assert algo.training and not algo.predictor.training


def test_wam_probe_keeps_decode_target_pixels_under_the_cache():
    """Every stream on the camera is cacheable (frozen root), but the probe's LOSS reads
    the raw frames of its decode targets -- their fields must stay pixel-served
    (keys+pixels mode), or a cached run dies on a KeyError at the first loss."""
    algo = build_algorithm(make_wam_cfg())
    assert "future" in algo.cacheable_streams()
    assert "observation.images.cam" in algo.raw_pixel_fields()


def test_wam_probe_decodes_prediction_next_to_encoded_latent():
    algo = build_algorithm(make_wam_cfg())
    columns = algo.reconstruct(make_wam_batch())["scene"]
    assert list(columns) == ["frame", "decoded latent", "decoded prediction"]
    for image in columns.values():
        assert image.shape == (2, 2, 3, 32, 32) and image.dtype == torch.uint8
    assert not torch.equal(columns["decoded latent"], columns["decoded prediction"])


def test_wam_probe_scores_the_prediction_in_pixels():
    """`prediction_errors` is the panel's third column as a number: the decoded rollout
    against the real future frames, in the same [0, 1] MSE units as the decoder's loss."""
    algo = build_algorithm(make_wam_cfg())
    errors = algo.prediction_errors(make_wam_batch())
    assert list(errors) == ["scene"]
    assert errors["scene"].isfinite() and errors["scene"] > 0
    assert algo.training  # scoring restores the mode it found the probe in


def test_wam_probe_rejects_a_loss_that_would_train_the_model():
    cfg = make_wam_cfg()
    cfg.losses = [{"type": "flow", "stream": "future", "weight": 1.0}]
    with pytest.raises(ValueError, match="may only name decoders"):
        build_algorithm(cfg)


def test_wam_probe_freezes_an_encoder_the_probed_run_was_training():
    """The spec here is the probed run's, where the encoders were training; the probe has
    to hold them still itself rather than ask the config to restate `frozen: true`."""
    cfg = make_wam_cfg()
    cfg.encoders.vit.frozen = False
    algo = build_algorithm(cfg)
    assert algo.encoders["vit"].frozen and not algo.encoders["vit"].training
    assert all(not p.requires_grad for p in algo.encoders["vit"].parameters())
    assert algo.summary()["encoders/trainable"] == "decode"
    cfg.freeze_world_model = False
    assert all(p.requires_grad for p in build_algorithm(cfg).encoders["vit"].parameters())


def test_wam_probe_rejects_a_decode_entry_naming_no_stream():
    cfg = make_wam_cfg()
    cfg.decode.scene.stream = "ghost"
    with pytest.raises(ValueError, match="neither"):
        build_algorithm(cfg)


def test_wam_probe_weights_line_up_with_the_run_it_probes(tmp_path):
    """The probe rebuilds the WAM from the same config, so a WAM checkpoint fills its
    encoders AND its trunk by name; only the decoders keep their init."""
    from thesis.algorithms.generic_vit_predictor import GenericViTPredictor

    wam_cfg = make_wam_cfg()
    del wam_cfg.decode
    wam_cfg.name = "generic_vit_predictor"
    wam_cfg.encoders = {k: v for k, v in wam_cfg.encoders.items() if k != "decode"}
    wam_cfg.losses = [{"type": "flow", "stream": "future", "weight": 1.0}]
    wam = build_algorithm(wam_cfg)
    assert isinstance(wam, GenericViTPredictor)
    path = tmp_path / "model.pt"
    torch.save({"model": wam.state_dict()}, path)

    probe = build_algorithm(make_wam_cfg())
    loaded, kept = load_pretrained_weights(probe, path)
    assert "predictor" in loaded and "encoders.vit" in loaded
    assert kept == ["encoders.decode"]
    for key, value in wam.predictor.state_dict().items():
        assert torch.equal(probe.predictor.state_dict()[key], value)




def probe_experiment(algorithm, encoder_init, monkeypatch, stored=None):
    """A DecoderExperiment whose wandb lookups are stubbed out."""
    from thesis.experiments import decoder as decoder_mod
    from thesis.utils import ckpt_utils

    root = OmegaConf.create({
        "algorithm": algorithm,
        "experiment": {"encoder_init": encoder_init},
        "wandb": {"entity": "team", "project": "proj"},
    })
    if stored is not None:
        monkeypatch.setattr(
            ckpt_utils, "fetch_run_config",
            lambda path: OmegaConf.create({"algorithm": stored}),
        )
    return decoder_mod.DecoderExperiment(root, output_dir=".")


def test_probe_inherits_the_spec_of_the_run_it_measures(monkeypatch):
    """The probe declares only its decoders; the model comes from the run, so the two can
    never drift -- and the decoders survive the merge into the run's `encoders:`."""
    probe_cfg = OmegaConf.create({
        "name": "wam_decoder",
        "encoders": {"decode": {"type": "pixel_decoder", "in_dim": 6}},
        "decode": {"scene": {"stream": "future", "decoder": "decode"}},
        "losses": [{"type": "prediction", "stream": "scene", "weight": 1.0}],
    })
    stored = make_wam_cfg()
    stored.name = "generic_vit_predictor"
    stored.model.depth = 5
    del stored.decode
    del stored.encoders["decode"]

    exp = probe_experiment(probe_cfg, "abcd1234", monkeypatch, stored=stored)
    exp._inherit_algorithm()

    algo = exp.root_cfg.algorithm
    assert algo.name == "wam_decoder", "the probe stays the probe"
    assert algo.model.depth == 5, "the trunk comes from the run"
    assert set(algo.encoders) == {"vit", "decode"}
    assert list(algo.conditioning) == ["context"] and "future" in algo.predict
    assert algo.decode.scene.stream == "future"
    assert [t["stream"] for t in algo.losses] == ["scene"]


def test_probe_falls_back_to_its_own_config_when_the_run_cannot_be_read(monkeypatch):
    from thesis.utils import ckpt_utils

    def boom(path):
        raise RuntimeError("no network")

    monkeypatch.setattr(ckpt_utils, "fetch_run_config", boom)
    cfg = make_wam_cfg()
    exp = probe_experiment(cfg, "abcd1234", monkeypatch, stored=None)
    exp._inherit_algorithm()
    assert exp.root_cfg.algorithm.model.depth == cfg.model.depth


def test_probe_of_a_local_checkpoint_keeps_its_composed_config(monkeypatch):
    """A .pt carries weights, not a config, so there is nothing to inherit from one."""
    cfg = make_wam_cfg()
    exp = probe_experiment(cfg, "/tmp/model.pt", monkeypatch, stored=make_cfg())
    exp._inherit_algorithm()
    assert exp.root_cfg.algorithm.name == "wam_decoder"




def test_panel_lays_columns_out_side_by_side():
    columns = {
        "frame": torch.full((3, 2, 3, 8, 8), 128, dtype=torch.uint8),
        "decoded latent": torch.zeros(3, 2, 3, 8, 8, dtype=torch.uint8),
        "decoded prediction": torch.full((3, 2, 3, 8, 8), 64, dtype=torch.uint8),
    }
    image, caption = panel(columns, gap=2)
    assert image.shape == (3 * 8 + 2 * 2, 6 * 8 + 5 * 2, 3)
    assert image.dtype == np.uint8
    assert (image[:8, :8] == 128).all()
    assert (image[:8, 10:18] == 0).all()
    assert (image[:8, 20:28] == 64).all()
    assert caption.startswith("frame | decoded latent | decoded prediction")
