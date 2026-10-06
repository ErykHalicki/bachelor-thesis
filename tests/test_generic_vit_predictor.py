import pytest
import torch
from omegaconf import OmegaConf

from thesis.algorithms import build_algorithm
from thesis.algorithms.generic_vit_predictor import sample_flow_time
from thesis.algorithms.layers import FinalLayer, InputLayer, MLPFinalLayer, MLPInputLayer

ROPE = {"time": {"share": 0.75, "period": "auto"},
        "seq": {"share": 0.25, "period": "auto"}}

ENCODER = {
    "type": "vit", "pooling": "cls", "dim": 32, "patch_size": 14, "depth": 2,
    "num_heads": 4, "mlp_ratio": 2.0, "img_size": 28,
}
PROJ = {"type": "mlp", "input": "vit", "in_dim": 32, "hidden_dim": 64, "out_dim": 32,
        "norm": "batch"}
# causal: the shared base builds a `direct` stream, which is unsound unmasked, and the
# mask tests below want a mask to look at. Attention is unmasked by default
MODEL = {"rope": ROPE, "model_dim": 32, "depth": 2, "num_heads": 4, "dim_head": 16,
         "mlp_ratio": 2.0, "causal": True}


def lewm_cfg():
    """LeWorldModel: pure `direct` -- CLS latents in, next latent out, action via AdaLN."""
    return OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": dict(MODEL),
        "encoders": {"vit": dict(ENCODER), "vit_proj": dict(PROJ),
                     "out_proj": {"type": "mlp", "in_dim": 32, "hidden_dim": 64,
                                  "out_dim": 32, "norm": "batch"}},
        "conditioning": {
            "video_context": {
                "from": "observation.images.pixels", "encoder": "vit_proj", "via": "tokens",
                "dim": 32, "index": "-2..0", "grid": [1, 1],
            },
            "action": {"from": "action", "via": "cond", "dim": 2, "index": "-2..0"},
        },
        "predict": {
            "future_video": {
                "from": "observation.images.pixels", "encoder": "vit_proj", "type": "direct",
                "input": "video_context", "dim": 32, "index": "-1..1", "grid": [1, 1],
                "decoder": "out_proj",
            },
        },
        "losses": [
            {"type": "prediction", "stream": "future_video", "weight": 1.0},
            {"type": "sigreg", "streams": ["video_context", "future_video"], "weight": 0.09},
        ],
    })


def mixed_cfg():
    """Both paradigms in one predict block: a flow action chunk + a direct video stream.
    The action chunk sits at index >= 1 so the block-causal mask keeps the context tokens
    (index <= 0) from attending to the noisy ones.
    """
    cfg = lewm_cfg()
    cfg.predict.action_chunk = {
        "from": "action", "type": "flow", "dim": 2, "index": "1..4", "fps": 1.0,
    }
    cfg.losses.insert(1, {"type": "flow", "stream": "action_chunk", "weight": 1.0})
    cfg.num_flow_steps = 2
    return cfg


def make_batch(action_rows, b=4):
    # each field carries the union of the windows reading it: pixels span -2..1 (4
    # rows); `action` spans -2..0 for lewm (3), plus the flow chunk's 1..4 for mixed (7)
    torch.manual_seed(0)
    return {
        "observation.images.pixels": (torch.rand(b, 4, 3, 96, 96) * 255).to(torch.uint8),
        "action": torch.randn(b, action_rows, 2),
    }


def randomize(model):
    for p in model.parameters():
        p.data.normal_(0.0, 0.02)


def test_direct_stream_adds_no_tokens_and_is_not_a_policy():
    algo = build_algorithm(lewm_cfg())
    assert algo.predictor.num_tokens == 3
    assert algo.predictor.direct_input == {"future_video": "video_context"}
    assert algo.predictor.flow_names == []
    assert algo.action_stream is None and algo.action_field is None
    assert not hasattr(algo, "chunk_len")
    assert algo.obs_len == 3


def test_a_stream_named_anything_still_carries_the_actions():
    """`action_chunk` reads the `action` column, so it is the chunk eval executes -- the
    stream's own name is a label. Looked up by that label, this model had no chunk_len and
    a policy eval would have taken it for a world model to plan with."""
    algo = build_algorithm(mixed_cfg())
    assert (algo.action_stream, algo.action_field) == ("action_chunk", "action")
    assert algo.chunk_len == 4


def test_role_marks_the_action_stream_when_the_column_is_named_otherwise():
    """An arm whose dataset column is not called `action` still evaluates."""
    cfg = lewm_cfg()
    cfg.predict.arm_cmd = {
        "from": "observation.joint_target", "role": "action", "type": "flow",
        "dim": 2, "index": "1..4", "fps": 1.0,
    }
    cfg.num_flow_steps = 2
    algo = build_algorithm(cfg)
    assert (algo.action_stream, algo.action_field) == ("arm_cmd", "observation.joint_target")
    assert algo.chunk_len == 4


def test_two_action_streams_are_rejected_at_build_time():
    """Building at all would mean picking one and silently executing it."""
    cfg = mixed_cfg()
    cfg.predict.action_chunk.role = "action"
    cfg.predict.arm_cmd = {
        "from": "cmd", "role": "action", "type": "flow", "dim": 2, "index": "1..4", "fps": 1.0,
    }
    with pytest.raises(ValueError, match="exactly one"):
        build_algorithm(cfg)


def test_a_declared_role_outranks_a_stream_named_action():
    """Explicit beats the convention, so adding a role to a config that already has an
    `action` stream does not deadlock on ambiguity."""
    cfg = mixed_cfg()
    cfg.predict.arm_cmd = {
        "from": "cmd", "role": "action", "type": "flow", "dim": 2, "index": "1..2", "fps": 1.0,
    }
    algo = build_algorithm(cfg)
    assert (algo.action_stream, algo.action_field) == ("arm_cmd", "cmd")
    assert algo.chunk_len == 2


def test_pure_direct_has_no_timestep_slot():
    algo = build_algorithm(lewm_cfg())
    trunk = algo.predictor
    assert trunk.has_flow is False and trunk.t_embedder is None
    assert trunk.blocks[0].adaLN_modulation[-1].in_features == MODEL["model_dim"]


def test_loss_logs_prediction_and_sigreg_and_trains():
    algo = build_algorithm(lewm_cfg())
    out = algo.loss(make_batch(action_rows=3))
    assert "loss/future_video" in out
    assert any(k.startswith("loss/sigreg") for k in out)
    out["loss"].backward()
    grads = [p.grad for p in algo.predictor.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_rollout_latent_matches_encoded_goal_space():
    algo = build_algorithm(lewm_cfg())
    n, horizon = 6, 8
    obs = {"observation.images.pixels": (torch.rand(n, 3, 3, 96, 96) * 255).to(torch.uint8)}
    final = algo.rollout_latent(obs, torch.randn(n, horizon, 2), stream="future_video")
    goal = (torch.rand(n, 1, 3, 96, 96) * 255).to(torch.uint8)
    goal_emb = algo.run_encoder("vit_proj", goal)
    assert final.shape == goal_emb.shape
    cost = torch.linalg.vector_norm(final.reshape(n, -1) - goal_emb.reshape(n, -1), dim=-1)
    assert cost.shape == (n,) and torch.isfinite(cost).all()


def lewm_mlp_action_cfg():
    """lewm_cfg with the action stream routed through an `mlp` encoder (le-wm's
    Embedder), so the trunk conditions on a nonlinear action embedding."""
    cfg = lewm_cfg()
    cfg.encoders["act_mlp"] = {"type": "mlp", "in_dim": 2, "hidden_dim": 16, "out_dim": 32}
    cfg.conditioning.action.encoder = "act_mlp"
    cfg.conditioning.action.dim = 32
    return cfg


def test_action_encoder_applied_in_training_and_rollout():
    """rollout_latent hand-assembles trunk inputs instead of going through _condition,
    so it must apply the action stream's spec-level encoder itself. With a 1-step
    horizon and identical windows, the rolled-out latent must equal the last position
    of the training-path predict() output -- which fails (shape error) if either path
    skips the encoder."""
    torch.manual_seed(0)
    algo = build_algorithm(lewm_mlp_action_cfg())
    n = 4
    px = (torch.rand(n, 3, 3, 96, 96) * 255).to(torch.uint8)
    actions = torch.randn(n, 3, 2)
    with torch.no_grad():
        pred = algo.predict({"observation.images.pixels": px, "action": actions})["future_video"]
        final = algo.rollout_latent(
            {"observation.images.pixels": px},
            actions[:, 2:],
            stream="future_video",
            past_actions=actions[:, :2],
        )
    assert final.shape == (n, 1, 32)
    assert torch.allclose(final, pred[:, -1:], atol=1e-5)


def test_mlp_action_encoder_trains():
    algo = build_algorithm(lewm_mlp_action_cfg())
    # AdaLN-zero passes zero gradient to its INPUT until its weights move, so
    # perturb it to test the wiring
    for m in algo.predictor.modules():
        if isinstance(m, torch.nn.Linear) and (m.weight == 0).all():
            torch.nn.init.normal_(m.weight, std=1e-3)
    out = algo.loss(make_batch(action_rows=3))
    out["loss"].backward()
    grads = [p.grad for p in algo.encoders["act_mlp"].parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_mixed_flow_and_direct_build_and_train():
    algo = build_algorithm(mixed_cfg())
    trunk = algo.predictor
    assert trunk.flow_names == ["action_chunk"]
    assert trunk.direct_input == {"future_video": "video_context"}
    assert trunk.has_flow is True
    assert trunk.blocks[0].adaLN_modulation[-1].in_features == 2 * MODEL["model_dim"]

    out = algo.loss(make_batch(action_rows=7))
    assert "loss/future_video" in out and "loss/action_chunk" in out
    out["loss"].backward()


def test_timestep_never_reaches_the_direct_path():
    """The timestep slot is zeroed on the direct stream's input tokens, and the block-causal
    mask stops `t` arriving via attention, so the direct head is EXACTLY t-invariant (a
    structural guarantee, hence bit-exact) while the flow head is not.
    """
    algo = build_algorithm(mixed_cfg())
    randomize(algo)
    trunk = algo.predictor
    torch.manual_seed(1)
    tokens = {
        "video_context": torch.randn(2, 3, 32),
        "action_chunk": torch.randn(2, 4, 2),
    }
    cond = {"action": torch.randn(2, 3, 2)}

    lo = trunk(tokens, torch.zeros(2), cond=cond)
    hi = trunk(tokens, torch.ones(2), cond=cond)
    assert torch.equal(lo["future_video"], hi["future_video"])
    assert not torch.equal(lo["action_chunk"], hi["action_chunk"])


def test_timestep_slot_is_zeroed_only_on_direct_tokens():
    """The cond layout itself: `t` moves the action_chunk tokens' conditioning and leaves the
    direct stream's input tokens (video_context) untouched.
    """
    trunk = build_algorithm(mixed_cfg()).predictor
    assert trunk.t_free.tolist() == [True] * 3 + [False] * 4
    torch.manual_seed(3)
    act = {"action": torch.randn(2, 3, 2)}
    lo, _ = trunk._build_cond(torch.zeros(2), act, 2, torch.device("cpu"), torch.float32)
    hi, _ = trunk._build_cond(torch.ones(2), act, 2, torch.device("cpu"), torch.float32)
    moved = [not torch.equal(lo[:, i], hi[:, i]) for i in range(lo.shape[1])]
    assert moved == [False] * 3 + [True] * 4


def test_per_timestep_sigreg_trains_end_to_end():
    cfg = lewm_cfg()
    cfg.losses[1]["statistic"] = "per_timestep"
    algo = build_algorithm(cfg)
    out = algo.loss(make_batch(action_rows=3))
    assert any(k.startswith("loss/sigreg") for k in out)
    out["loss"].backward()


def flow_overlap_cfg():
    """Pure flow policy whose context reaches index 0: the video token at step 0 spans
    [0, 1)s and time-overlaps the first chunk actions, exercising the role rule."""
    cfg = lewm_cfg()
    cfg.predict = OmegaConf.create({
        "action_chunk": {"from": "action", "type": "flow", "dim": 2, "index": "0..3",
                         "fps": 1.0},
    })
    cfg.losses = [{"type": "flow", "stream": "action_chunk", "weight": 1.0}]
    return cfg


def test_clean_tokens_never_attend_noisy_tokens():
    cfg = flow_overlap_cfg()
    cfg.model.clean_attends_noisy = False
    mask = build_algorithm(cfg).predictor.dense_mask
    video, chunk = slice(0, 3), slice(3, 7)
    assert not torch.any(mask[video, chunk])
    assert torch.all(mask[chunk, video])
    assert torch.equal(mask[chunk, chunk], torch.ones(4, 4, dtype=torch.bool).tril())


def test_flow_arms_are_unmasked_by_default():
    """No causality and no role rule leaves nothing for a mask to say, so the trunk builds
    none at all rather than an all-True (N, N) tensor SDPA would have to multiply through."""
    cfg = flow_overlap_cfg()
    del cfg.model.causal
    trunk = build_algorithm(cfg).predictor
    assert trunk._unmasked
    assert trunk._mask() is None
    assert not hasattr(trunk, "dense_mask")
    algo = build_algorithm(cfg)
    algo.loss(make_batch(action_rows=7))["loss"].backward()


def test_block_size_needs_causal_to_mean_anything():
    """block_size partitions the time axis, and unmasked there is no time rule left for it
    to partition -- so a run that sets one without `causal: true` gets nothing. The role
    rule is kept on so there is still a mask to compare."""
    def mask_for(causal, block_size):
        cfg = flow_overlap_cfg()
        cfg.model.causal = causal
        cfg.model.clean_attends_noisy = False
        cfg.predict.action_chunk.block_size = block_size
        return build_algorithm(cfg).predictor.dense_mask

    assert not torch.equal(mask_for(True, 1), mask_for(True, 4))
    assert torch.equal(mask_for(False, 1), mask_for(False, 4))
    # and what survives unmasked is the role rule alone: clean tokens see nothing noisy,
    # everything else is open
    video, chunk = slice(0, 3), slice(3, 7)
    mask = mask_for(False, 1)
    assert not torch.any(mask[video, chunk])
    assert torch.all(mask[chunk, :]) and torch.all(mask[video, video])


def test_a_direct_stream_refuses_to_build_unmasked():
    cfg = lewm_cfg()
    del cfg.model.causal
    with pytest.raises(AssertionError, match="need `model.causal: true`"):
        build_algorithm(cfg)


def test_clean_attends_noisy_restores_time_only_mask():
    cfg = flow_overlap_cfg()
    cfg.model.clean_attends_noisy = True
    mask = build_algorithm(cfg).predictor.dense_mask
    assert mask[2, 3]
    assert not torch.any(mask[2, 4:])
    assert not torch.any(mask[:2, 3:])                     # past video is untouched either way


def test_flow_time_is_beta_skewed_toward_the_noise_end():
    """t=0 is noise here and t=1 is data, the mirror of pi0's axis, so the Beta(1.5, 1.0)
    schedule has to come out skewed LOW. Sampling it unmirrored would spend the extra
    samples on the clean end -- the exact opposite of the point -- and nothing else in
    training would notice."""
    t = sample_flow_time(20000, "cpu")
    assert t.min() >= 0.0 and t.max() <= 1.0
    assert (t < 0.25).float().mean() > 2 * (t > 0.75).float().mean()
    assert abs(t.mean().item() - 0.4) < 0.02       # 1 - alpha/(alpha+beta), scaled


def dz_cfg():
    """mixed_cfg with DreamZero's token in/out on the flow stream."""
    cfg = mixed_cfg()
    cfg.predict.action_chunk.token_in = {"type": "mlp", "mlp_dim": 32}
    cfg.predict.action_chunk.token_out = {"type": "mlp", "mlp_dim": 8}
    return cfg


def test_mlp_token_io_trains_and_keeps_the_stream_in_raw_space():
    algo = build_algorithm(dz_cfg())
    trunk = algo.predictor
    assert isinstance(trunk.noisy_proj["action_chunk"], MLPInputLayer)
    assert isinstance(trunk.heads["action_chunk"], MLPFinalLayer)
    # the direct stream declared nothing either, so it keeps the (now default) MLP head
    assert isinstance(trunk.heads["future_video"], MLPFinalLayer)
    assert trunk.time_concat == {"action_chunk"}
    out = algo.loss(make_batch(action_rows=7))
    out["loss"].backward()
    assert torch.isfinite(out["loss"])


def test_mlp_token_in_folds_the_timestep_into_the_tokens():
    """token_in is handed the bare timestep slot, never the concatenated cond, so if it did
    not fold `t` in itself the action tokens would enter the trunk t-invariant."""
    algo = build_algorithm(dz_cfg())
    trunk = algo.predictor
    torch.manual_seed(1)
    x = torch.randn(2, 4, 2)
    layer = trunk.noisy_proj["action_chunk"]
    spread = (layer(x, trunk.t_embedder(torch.zeros(2)))
              - layer(x, trunk.t_embedder(torch.ones(2)))).abs().max()
    assert spread > 1e-3

    randomize(algo)
    tokens = {"video_context": torch.randn(2, 3, 32), "action_chunk": x}
    cond = {"action": torch.randn(2, 3, 2)}
    lo = trunk(tokens, torch.zeros(2), cond=cond)
    hi = trunk(tokens, torch.ones(2), cond=cond)
    assert not torch.equal(lo["action_chunk"], hi["action_chunk"])
    assert torch.equal(lo["future_video"], hi["future_video"])


def test_both_token_out_kinds_are_zero_at_init():
    """Either decoder starts the velocity at exactly zero, so swapping token_out varies the
    I/O structure and not the starting point. DreamZero's own decoder is not zero-init; this
    departs from it deliberately, because an A2A seed puts x0 near x1 and a nonzero velocity
    at init pushes the endpoint off an answer that is already nearly right."""
    torch.manual_seed(0)
    tokens = {"video_context": torch.randn(2, 3, 32), "action_chunk": torch.randn(2, 4, 2)}
    cond = {"action": torch.randn(2, 3, 2)}
    t = torch.full((2,), 0.3)
    for cfg in (mixed_cfg(), dz_cfg()):
        model = build_algorithm(cfg)
        out = model.predictor(tokens, t, cond=cond)["action_chunk"]
        assert torch.count_nonzero(out) == 0
        # zero output, but still a live path: the head's own weight takes gradient
        out.sum().backward()
        head = model.predictor.heads["action_chunk"]
        weight = head.mlp[-1].weight if hasattr(head, "mlp") else head.linear.weight
        assert weight.grad is not None and weight.grad.abs().sum() > 0


def test_mlp_token_io_is_the_default_with_no_config_at_all():
    """A stream that declares no token_in/token_out at all -- not even dz_cfg's explicit
    block -- still gets DreamZero's MLP pair, at mlp_dim: model_dim. Every predicted
    stream leaves the trunk the same way unless a config opts a stream out."""
    trunk = build_algorithm(mixed_cfg()).predictor
    assert isinstance(trunk.noisy_proj["action_chunk"], MLPInputLayer)
    assert isinstance(trunk.heads["action_chunk"], MLPFinalLayer)
    assert isinstance(trunk.heads["future_video"], MLPFinalLayer)
    assert trunk.noisy_proj["action_chunk"].mlp[0].w.in_features == 2 * MODEL["model_dim"]


def test_adaln_is_still_available_by_declaring_it_explicitly():
    cfg = mixed_cfg()
    cfg.predict.action_chunk.token_in = {"type": "adaln"}
    cfg.predict.action_chunk.token_out = {"type": "adaln"}
    cfg.predict.future_video.token_out = {"type": "adaln"}
    trunk = build_algorithm(cfg).predictor
    assert isinstance(trunk.noisy_proj["action_chunk"], InputLayer)
    assert isinstance(trunk.heads["action_chunk"], FinalLayer)
    assert isinstance(trunk.heads["future_video"], FinalLayer)
    assert "action_chunk" not in trunk.time_concat


def test_clean_conditioning_streams_enter_through_an_mlp_not_a_bare_linear():
    """Clean/conditioning token streams (via: tokens) are not configurable via token_in --
    they always get the same MLP lift the noisy default uses, minus the timestep
    concatenation, since they are never denoised."""
    trunk = build_algorithm(mixed_cfg()).predictor
    proj = trunk.clean_proj["video_context"]
    assert not isinstance(proj, torch.nn.Linear)
    out = proj(torch.randn(2, 3, 32))
    assert out.shape == (2, 3, MODEL["model_dim"])


@pytest.mark.parametrize("block, message", [
    ({"type": "mlp"}, "must declare its `mlp_dim`"),
    ({"type": "adaln", "mlp_dim": 8}, "no mlp_dim to set"),
    ({"type": "conv", "mlp_dim": 8}, "unknown type"),
    ({"type": "mlp", "mlp_dim": 8, "hidden": 4}, "unknown keys"),
])
def test_token_io_rejects_malformed_blocks(block, message):
    cfg = mixed_cfg()
    cfg.predict.action_chunk.token_out = block
    with pytest.raises(ValueError, match=message):
        build_algorithm(cfg)


def test_token_io_is_rejected_where_there_is_no_projection_to_configure():
    cfg = mixed_cfg()
    cfg.conditioning.video_context.token_in = {"type": "mlp", "mlp_dim": 8}
    with pytest.raises(AssertionError, match="conditioning stream"):
        build_algorithm(cfg)
    cfg = mixed_cfg()
    cfg.predict.future_video.token_in = {"type": "mlp", "mlp_dim": 8}
    with pytest.raises(AssertionError, match="adds no tokens of its own"):
        build_algorithm(cfg)


def test_direct_input_tokens_are_exempt_from_role_rule():
    """video_context feeds the direct head, so its tokens are prediction sites and keep
    the time-based mask even though they are a clean stream."""
    cfg = mixed_cfg()
    cfg.model.clean_attends_noisy = False
    cfg.predict.action_chunk.index = "0..3"
    mask = build_algorithm(cfg).predictor.dense_mask
    assert mask[2, 3]                                      # video step 0 sees action 0
    assert not torch.any(mask[:2, 3:])
