import pytest
import torch

from thesis.utils.augment import Augmenter, AugmentedSource, _Rng, color_jitter


class _ToySource:
    """Fixed items, so any difference in an output comes from the augmenter."""

    def __init__(self, item):
        self.item = item
        self.provided_modalities = set(item)
        self.stats = {"action": {"mean": [0.0]}}

    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return {k: v.clone() for k, v in self.item.items()}


def _batch(image_dtype=torch.uint8, frames=4, channels=3, size=16):
    if image_dtype == torch.uint8:
        image = torch.randint(0, 256, (frames, channels, size, size), dtype=torch.uint8)
    else:
        image = torch.rand(frames, channels, size, size)
    return {
        "observation.images.front": image,
        "observation.state": torch.zeros(2, 6),
        "action": torch.zeros(8, 6),
    }


def test_gaussian_noise_perturbs_only_configured_streams():
    batch = _batch()
    out = Augmenter({"observation.state": {"gaussian_noise": {"std": 0.1}}}, seed=0)(batch)
    assert out["observation.state"].abs().sum() > 0
    assert torch.equal(out["action"], batch["action"])
    assert torch.equal(out["observation.images.front"], batch["observation.images.front"])


def test_shape_and_dtype_are_preserved():
    for dtype in (torch.uint8, torch.float32):
        batch = _batch(image_dtype=dtype)
        out = Augmenter(
            {
                "observation.images.*": {
                    "random_crop": {"scale": 0.8},
                    "rotation": {"degrees": 10.0},
                    "color_jitter": {"brightness": 0.3, "contrast": 0.3,
                                     "saturation": 0.3, "hue": 0.1},
                    "gaussian_noise": {"std": 0.05},
                }
            },
            seed=0,
        )(batch)
        image = out["observation.images.front"]
        assert image.shape == batch["observation.images.front"].shape
        assert image.dtype == dtype
        limit = 255 if dtype == torch.uint8 else 1.0
        assert image.min() >= 0 and image.max() <= limit


def test_image_augmentation_is_shared_across_the_time_window():
    frame = torch.rand(1, 3, 16, 16)
    batch = {"observation.images.front": frame.repeat(4, 1, 1, 1)}
    out = Augmenter(
        {
            "observation.images.front": {
                "random_crop": {"scale": [0.5, 0.9]},
                "rotation": {"degrees": 15.0},
                "color_jitter": {"brightness": 0.5, "hue": 0.2},
            }
        },
        seed=3,
    )(batch)["observation.images.front"]
    for i in range(1, 4):
        assert torch.equal(out[0], out[i])


def test_fields_draw_independently():
    frame = torch.rand(2, 3, 16, 16)
    batch = {"cam.left": frame.clone(), "cam.right": frame.clone()}
    out = Augmenter({"cam.*": {"random_crop": {"scale": 0.5}}}, seed=1)(batch)
    assert not torch.equal(out["cam.left"], out["cam.right"])


def test_glob_patterns_merge_in_declaration_order():
    aug = Augmenter(
        {
            "observation.images.*": {"gaussian_noise": {"std": 0.1}, "rotation": {"degrees": 5.0}},
            "observation.images.front": {"gaussian_noise": {"std": 0.0}},
        }
    )
    ops = aug._ops_for("observation.images.front")
    assert ops["gaussian_noise"] == {"std": 0.0}
    assert ops["rotation"] == {"degrees": 5.0}
    assert list(ops) == ["rotation", "gaussian_noise"]


def test_p_zero_is_identity():
    batch = _batch(image_dtype=torch.float32)
    out = Augmenter(
        {
            "observation.images.front": {
                "random_crop": {"scale": 0.5, "p": 0.0},
                "rotation": {"degrees": 20.0, "p": 0.0},
                "color_jitter": {"brightness": 0.9, "p": 0.0},
                "gaussian_noise": {"std": 1.0, "p": 0.0},
            }
        },
        seed=0,
    )(batch)
    assert torch.equal(out["observation.images.front"], batch["observation.images.front"])


def test_seed_makes_augmentation_reproducible():
    config = {"observation.images.front": {"random_crop": {"scale": 0.7},
                                           "gaussian_noise": {"std": 0.1}}}
    batch = _batch(image_dtype=torch.float32)
    first = Augmenter(config, seed=7)(batch)["observation.images.front"]
    second = Augmenter(config, seed=7)(batch)["observation.images.front"]
    other = Augmenter(config, seed=8)(batch)["observation.images.front"]
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_consecutive_items_are_augmented_differently():
    source = AugmentedSource(
        _ToySource(_batch(image_dtype=torch.float32)),
        {"seed": 0, "streams": {"observation.state": {"gaussian_noise": {"std": 0.1}}}},
    )
    assert not torch.equal(source[0]["observation.state"], source[1]["observation.state"])


def test_wrapper_forwards_attributes_of_the_wrapped_source():
    inner = _ToySource(_batch())
    source = AugmentedSource(inner, {"streams": {"action": {"gaussian_noise": {"std": 0.1}}}})
    assert len(source) == len(inner)
    assert source.provided_modalities == inner.provided_modalities
    assert source.stats == inner.stats


def test_unknown_op_and_param_are_config_errors():
    with pytest.raises(ValueError, match="unknown augmentation 'blur'"):
        Augmenter({"action": {"blur": {"radius": 2}}})
    with pytest.raises(ValueError, match="unknown parameter"):
        Augmenter({"action": {"gaussian_noise": {"stdev": 0.1}}})


def test_image_op_on_a_vector_stream_is_an_error():
    with pytest.raises(ValueError, match="not \\(\\.\\.\\., C, H, W\\)"):
        Augmenter({"action": {"random_crop": {"scale": 0.9}}})(_batch())


def test_pattern_matching_no_field_is_an_error():
    with pytest.raises(ValueError, match="matches no batch field"):
        Augmenter({"observation.images.wrist": {"gaussian_noise": {"std": 0.1}}})(_batch())


def test_rotation_of_a_uniform_image_is_near_identity():
    batch = {"cam": torch.full((2, 3, 12, 20), 0.4)}
    out = Augmenter({"cam": {"rotation": {"degrees": 20.0}}}, seed=2)(batch)["cam"]
    assert torch.allclose(out, batch["cam"], atol=1e-5)


def test_rotation_is_aspect_corrected_on_a_non_square_image():
    # without the aspect correction the per-axis normalized coordinates shear this
    # radially symmetric blob into an ellipse
    height, width = 24, 48
    ys = torch.arange(height).view(-1, 1) - (height - 1) / 2
    xs = torch.arange(width).view(1, -1) - (width - 1) / 2
    radius = (xs**2 + ys**2).sqrt()
    blob = torch.exp(-(radius**2) / 50.0).expand(1, 3, height, width).clone()
    inside = radius < height / 2 - 1
    out = Augmenter({"cam": {"rotation": {"degrees": 30.0}}}, seed=1)({"cam": blob})["cam"]
    assert (out - blob).abs()[..., inside].max() < 0.02


def test_hue_shift_is_a_no_op_at_zero_and_reversible():
    from thesis.utils.augment import _hsv_to_rgb, _rgb_to_hsv

    image = torch.rand(2, 3, 8, 8)
    assert torch.allclose(_hsv_to_rgb(*_rgb_to_hsv(image)), image, atol=1e-5)


def test_gamma_moves_midtones_and_pins_the_endpoints():
    """A tone curve, not the gain `brightness` applies: black stays black and white stays
    white while everything between moves.
    """
    ramp = torch.linspace(0.0, 1.0, 256).view(1, 1, 16, 16)
    out = color_jitter(ramp, _Rng(0), gamma=0.5, p=1.0)

    assert out[..., 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert out[..., -1, -1] == pytest.approx(1.0, abs=1e-6)
    assert not torch.allclose(out, ramp)
    assert (out.diff(dim=-1)[..., :-1] > 0).all()


def test_gamma_runs_before_contrast():
    """Contrast subtracts the mean, so a negative base would reach the fractional power and
    come back NaN if the two were applied the other way round.
    """
    x = torch.rand(1, 3, 8, 8)
    out = color_jitter(x, _Rng(1), contrast=0.9, gamma=0.5, p=1.0)
    assert torch.isfinite(out).all()


def test_apply_batch_matches_per_item_draws():
    """`experiment.augment_on: device` must reproduce the worker path: sample i of a batch
    gets the parameters the per-item call would have drawn i-th from the same seed."""
    streams = {"observation.images.*": {"random_crop": {"scale": [0.8, 1.0]},
                                        "color_jitter": {"brightness": 0.3, "hue": 0.05},
                                        "gaussian_noise": {"std": 0.01}},
               "observation.state": {"gaussian_noise": {"std": 0.1}}}
    items = [_batch() for _ in range(3)]
    for it in items:
        it["observation.images.front"] = torch.randint(0, 256, it["observation.images.front"].shape,
                                                       dtype=torch.uint8)
    collated = {k: torch.stack([it[k] for it in items]) for k in items[0]}
    collated["task"] = ["a", "b", "c"]
    batched = Augmenter(streams, seed=3).apply_batch(collated)
    per_item = Augmenter(streams, seed=3)
    for i, it in enumerate(items):
        ref = per_item(it)
        for k in it:
            assert torch.equal(batched[k][i], ref[k]), (k, i)
    assert batched["task"] == ["a", "b", "c"]
    assert batched["observation.images.front"].dtype == torch.uint8
    assert not torch.equal(batched["observation.images.front"][0],
                           batched["observation.images.front"][1])


def test_live_augment_streams_drops_patterns_for_cached_fields():
    from thesis.utils.augment import live_augment_streams

    streams = {"observation.images.*": {"random_crop": {}}, "observation.state": {"gaussian_noise": {"std": 0.1}}}
    assert live_augment_streams(streams, {"observation.state", "action"}) == {
        "observation.state": {"gaussian_noise": {"std": 0.1}}}
    assert live_augment_streams(streams, set()) == {}
    # a pattern that survives must still match at call time
    with pytest.raises(ValueError, match="matches no batch field"):
        Augmenter(live_augment_streams(streams, {"observation.state"}), seed=0)(
            {"action": torch.zeros(2, 3)})
