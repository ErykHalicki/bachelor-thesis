import torch

from thesis.algorithms.generic_vit_predictor import GenericViTTrunk

CONDITIONING = {
    "video_context": {"dim": 16, "index": [-3, -2, -1, 0], "fps": 1.0, "grid": (2, 2)},
    "state": {"dim": 4, "via": "cross", "index": [0]},
}
LANGUAGE = {"language": {"dim": 12, "via": "cross"}}
PREDICT = {
    "future_video": {"dim": 16, "index": [1, 2], "fps": 1.0, "grid": (2, 2)},
    "action": {"dim": 4, "index": [1, 2, 3, 4, 5, 6], "fps": 3.0},
}
ROPE = {"time": {"share": 0.5, "period": "auto"},
        "height": {"share": 0.125, "period": "auto"},
        "width": {"share": 0.125, "period": "auto"},
        "seq": {"share": 0.25, "period": "auto"}}
DIMS = dict(rope=ROPE, model_dim=64, depth=2, num_heads=4)


def stream_len(model, name):
    s = model.stream_slices[name]
    return s.stop - s.start


def make_model(language=False, predict=PREDICT):
    torch.manual_seed(0)
    conditioning = {**CONDITIONING, **(LANGUAGE if language else {})}
    return GenericViTTrunk(conditioning, predict, **DIMS)


def make_inputs(model, batch=2):
    torch.manual_seed(1)
    tokens = {
        name: torch.randn(batch, stream_len(model, name), model.stream_dims[name])
        for name in model.stream_order
    }
    sources = {"state": torch.randn(batch, 1, 4)}
    return tokens, sources


def randomize(model):
    for p in model.parameters():
        p.data.normal_(0.0, 0.02)


def test_token_layout_from_spec():
    model = make_model()
    assert model.stream_order == ["video_context", "future_video", "action"]
    assert stream_len(model, "video_context") == 16
    assert stream_len(model, "future_video") == 8
    assert stream_len(model, "action") == 6
    assert model.num_tokens == 30


def test_single_stream_specs():
    action_only = make_model(predict={"action": PREDICT["action"]})
    tokens, sources = make_inputs(action_only)
    v = action_only(tokens, torch.rand(2), sources)
    assert set(v) == {"action"}

    video_only = make_model(predict={"future_video": PREDICT["future_video"]})
    tokens, sources = make_inputs(video_only)
    v = video_only(tokens, torch.rand(2), sources)
    assert set(v) == {"future_video"}


def test_language_conditioning_and_kv_cache_isolation():
    model = make_model(language=True)
    randomize(model)
    tokens, sources = make_inputs(model, batch=1)
    t = torch.rand(1)
    lang = [torch.randn(1, 3, 12) for _ in range(2)]

    model.clear_cache()
    v_cond = model(tokens, t, {**sources, "language": lang}, use_kv_cache=True)
    v_uncond = model(tokens, t, sources, use_kv_cache=False)
    assert not torch.allclose(v_cond["action"], v_uncond["action"], atol=1e-6)

    v_again = model(tokens, t, {**sources, "language": lang}, use_kv_cache=True)
    assert torch.allclose(v_cond["action"], v_again["action"])


def test_ode_solve_shapes_and_determinism():
    model = make_model(language=True)
    randomize(model)
    tokens, sources = make_inputs(model)
    sources["language"] = [torch.randn(2, 3, 12) for _ in range(2)]
    clean = {"video_context": tokens["video_context"]}

    torch.manual_seed(7)
    out = model.ode_solve(clean, sources, num_steps=3)
    assert out["future_video"].shape == (2, 8, 16)
    assert out["action"].shape == (2, 6, 4)

    torch.manual_seed(7)
    out2 = model.ode_solve(clean, sources, num_steps=3)
    assert all(torch.equal(out[k], out2[k]) for k in out)


def test_ode_solve_cfg_differs():
    model = make_model(language=True)
    randomize(model)
    tokens, sources = make_inputs(model, batch=1)
    sources["language"] = [torch.randn(1, 3, 12) for _ in range(2)]
    clean = {"video_context": tokens["video_context"]}

    torch.manual_seed(7)
    plain = model.ode_solve(clean, sources, num_steps=2)
    torch.manual_seed(7)
    guided = model.ode_solve(clean, sources, num_steps=2, cfg_scale=3.0, cfg_drop=("language",))
    assert not torch.allclose(plain["action"], guided["action"], atol=1e-6)
