"""`model.experts`: a true Mixture-of-Transformers trunk. Each token stream names the
expert whose parameters process it; one masked self-attention spans every token. With
the experts' weights tied the trunk is bit-for-bit the shared DiT; with them free, an
expert's loss reaches another expert's weights only through the attention mask."""


import pytest
import torch

from thesis.algorithms import build_algorithm
from thesis.algorithms.block import DiTBlock, MoTBlock
from thesis.algorithms.generic_vit_predictor import build_expert_ids
from thesis.algorithms.masking import block_causal_mask, block_intervals
from thesis.algorithms.rope import RopeND, axes_from_positions, make_pos_ids

from test_generic_vit_predictor import randomize
from test_selfwam_attends import make_batch, mse_cfg, selfwam_cfg

# the MoT-flow split: the world model's tokens in one expert, the policy's in the other
EXPERTS = {"state": ["video_context", "future_video"],
           "action": ["action_clean", "action_chunk"]}


def mot_cfg(experts=EXPERTS, base=selfwam_cfg):
    cfg = base()
    cfg.model.experts = experts
    return cfg


def tie_experts(mot, shared):
    """Copy the shared trunk's weights into EVERY expert of the MoT trunk (and its
    encoders / projections / heads verbatim), so the two compute the same function."""
    src = shared.state_dict()
    dst = {}
    for key in mot.state_dict():
        if key in src:
            dst[key] = src[key]
            continue
        # predictor.blocks.<i>.<module>.<e>.<rest> <- predictor.blocks.<i>.<shared name>.<rest>
        head, _, rest = key.partition(".blocks.")
        layer, module, expert, *tail = rest.split(".")
        shared_module = {"qkv": "sa.qkv", "q_norm": "sa.q_norm", "k_norm": "sa.k_norm",
                         "out_proj": "sa.out_proj"}.get(module, module)
        dst[key] = src[".".join([head, "blocks", layer, shared_module, *tail])]
    mot.load_state_dict(dst)


def test_experts_build_mot_blocks_with_the_declared_table():
    trunk = build_algorithm(mot_cfg()).predictor
    assert trunk.expert_names == ["state", "action"]
    assert all(isinstance(b, MoTBlock) for b in trunk.blocks)
    owner = {n: trunk.expert_names[int(trunk.all_expert_ids[trunk.stream_slices[n]][0])]
             for n in trunk.stream_order}
    assert owner == {"video_context": "state", "action_clean": "action",
                     "action_chunk": "action", "future_video": "state"}
    # each stream's tokens all sit in one expert
    for n in trunk.stream_order:
        assert len(set(trunk.all_expert_ids[trunk.stream_slices[n]].tolist())) == 1


def test_without_experts_the_trunk_is_the_shared_dit_unchanged():
    trunk = build_algorithm(selfwam_cfg()).predictor
    assert trunk.expert_names == [] and not hasattr(trunk, "all_expert_ids")
    assert all(isinstance(b, DiTBlock) for b in trunk.blocks)


@pytest.mark.parametrize("experts, match", [
    ({"only": ["all"]}, "one expert is the shared trunk"),
    ({"state": ["video_context", "future_video"], "action": ["action_chunk"]}, "unassigned"),
    ({"state": ["all"], "action": ["action_chunk"]}, "both 'state' and 'action'"),
    ({"state": ["video_context", "future_video"], "action": ["action_clean", "typo"]},
     "not a token stream"),
    ({"state": ["all"], "action": []}, "lists no streams"),
])
def test_expert_table_is_validated_at_build(experts, match):
    with pytest.raises(ValueError, match=match):
        build_algorithm(mot_cfg(experts))


def test_group_aliases_resolve_like_attends():
    names, ids = build_expert_ids(["ctx", "clean", "pol", "fut"], ["ctx", "clean"],
                                  ["pol", "fut"], {"a": ["context"], "b": ["noisy"]})
    assert names == ["a", "b"] and ids.tolist() == [0, 0, 1, 1]


def test_tied_experts_reproduce_the_shared_trunk():
    """The routing is an exact re-arrangement: with every expert holding the shared
    weights, velocities agree with the shared DiT on the same tokens and timestep."""
    torch.manual_seed(0)
    shared = build_algorithm(selfwam_cfg())
    randomize(shared)
    mot = build_algorithm(mot_cfg())
    tie_experts(mot, shared)
    torch.manual_seed(1)
    tokens = {"video_context": torch.randn(2, 3, 32), "action_clean": torch.randn(2, 4, 2),
              "action_chunk": torch.randn(2, 4, 2), "future_video": torch.randn(2, 1, 32)}
    t = torch.rand(2)
    with torch.no_grad():
        a = shared.predictor(dict(tokens), t)
        b = mot.predictor(dict(tokens), t)
    for name in a:
        assert torch.allclose(a[name], b[name], atol=1e-5), name


def test_tied_block_matches_dit_block_with_sources_rope_and_mask():
    """Block-level twin of the above, covering the cross-attention branch, RoPE and a
    mask -- the paths the trunk-level fixture has no source for."""
    hidden, heads, n = 64, 4, 6
    torch.manual_seed(0)
    dit = DiTBlock(hidden, heads, source_dims=[4])
    randomize(dit)
    mot = MoTBlock(hidden, heads, 2, source_dims=[4])
    state = {}
    for key, value in dit.state_dict().items():
        module, _, rest = key.partition(".")
        if module == "sa":
            module, _, rest = rest.partition(".")
        for e in range(2):
            state[f"{module}.{e}.{rest}"] = value
    mot.load_state_dict(state)
    indices = torch.arange(n)
    pos = make_pos_ids(indices.float() / 2.0)
    axes = axes_from_positions(pos, {"time": {"share": 1.0, "period": "auto"}})
    freqs = RopeND(hidden // heads, axes)(pos)
    mask = block_causal_mask(*block_intervals(indices, fps=2.0))
    x, cond = torch.randn(2, n, hidden), torch.randn(2, hidden)
    sources = [torch.randn(2, 3, 4)]
    routes = (torch.tensor([0, 2, 4]), torch.tensor([1, 3, 5]))
    with torch.no_grad():
        a = dit(x, cond, sources=sources, freqs=freqs, attn_mask=mask)
        b = mot(x, cond, sources=sources, freqs=freqs, attn_mask=mask, routes=routes)
        c = mot(x, cond[:, None].expand(2, n, hidden), sources=sources, freqs=freqs,
                attn_mask=mask, routes=routes)
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(a, c, atol=1e-5)


def expert_params(trunk, expert):
    e = trunk.expert_names.index(expert)
    return [(name, p) for name, p in trunk.named_parameters()
            if name.startswith("blocks.") and name.split(".")[3] == str(e)]


def test_losses_reach_experts_only_through_the_mask():
    """Under the selfwam mask nobody attends the policy's chunk, so the future-video
    loss leaves the action expert's weights untouched -- and the action loss reaches
    the state expert only because the policy reads the context tokens it processes."""
    algo = build_algorithm(mot_cfg({"state": ["video_context", "future_video", "action_clean"],
                                    "action": ["action_chunk"]}))
    trunk = algo.predictor
    randomize(trunk)
    torch.manual_seed(1)
    tokens = {"video_context": torch.randn(2, 3, 32), "action_clean": torch.randn(2, 4, 2),
              "action_chunk": torch.randn(2, 4, 2), "future_video": torch.randn(2, 1, 32)}
    t = torch.rand(2)

    trunk.zero_grad()
    trunk(dict(tokens), t)["future_video"].square().sum().backward()
    assert all(p.grad is None or not p.grad.any() for _, p in expert_params(trunk, "action"))
    assert any(p.grad is not None and p.grad.any() for _, p in expert_params(trunk, "state"))

    trunk.zero_grad()
    trunk(dict(tokens), t)["action_chunk"].square().sum().backward()
    assert any(p.grad is not None and p.grad.any() for _, p in expert_params(trunk, "action"))
    assert any(p.grad is not None and p.grad.any() for _, p in expert_params(trunk, "state"))


def test_subset_integration_matches_full_pass_under_experts():
    """The routes follow every subset layout: a policy-only pass and a futures-only
    pass each reproduce the full pass bit-for-bit."""
    algo = build_algorithm(mot_cfg())
    randomize(algo.predictor)
    obs = make_batch()
    x0 = {"action_chunk": torch.randn(2, 4, 2), "future_video": torch.randn(2, 1, 32)}
    clean, sources, cond, _ = algo._condition(obs)
    full = algo.predictor.rollout(clean, sources, cond=cond, num_steps=2, x0=dict(x0))
    for stream in ("action_chunk", "future_video"):
        needed = algo.predictor.required_streams([stream])
        sub = algo.predictor.rollout({k: v for k, v in clean.items() if k in needed}, sources,
                                     cond=cond, num_steps=2, x0={stream: x0[stream]},
                                     streams=[stream])
        assert set(sub) == {stream}
        assert torch.equal(sub[stream], full[stream])


def test_trains_with_and_without_gradient_checkpointing():
    algo = build_algorithm(mot_cfg())
    randomize(algo.predictor)
    batch = make_batch()
    torch.manual_seed(3)
    plain = algo.loss(batch)
    plain["loss"].backward()
    grads = {n: p.grad.clone() for n, p in algo.named_parameters() if p.grad is not None}
    algo.zero_grad()
    algo.predictor.gradient_checkpointing = True
    torch.manual_seed(3)
    ckpt = algo.loss(batch)
    ckpt["loss"].backward()
    assert torch.allclose(plain["loss"], ckpt["loss"])
    for n, p in algo.named_parameters():
        if n in grads:
            assert torch.allclose(grads[n], p.grad, atol=1e-6), n


def test_mse_stream_routes_its_query_tokens():
    algo = build_algorithm(mot_cfg(base=mse_cfg))
    out = algo.loss(make_batch())
    out["loss"].backward()
    assert algo.predictor.query_embed["future_video"].grad is not None
    pred = algo.predict(make_batch())
    assert pred["future_video"].shape == (2, 1, 32)
    assert algo.predict(make_batch(), streams=["action_chunk"]).keys() == {"action_chunk"}


def test_experts_double_the_trunk_and_nothing_else():
    shared = build_algorithm(selfwam_cfg())
    mot = build_algorithm(mot_cfg())
    count = lambda m, pred: sum(p.numel() for n, p in m.named_parameters() if pred(n))
    in_blocks = lambda n: n.startswith("predictor.blocks.")  # not the ViT encoder's blocks
    assert count(mot, in_blocks) == 2 * count(shared, in_blocks)
    assert count(mot, lambda n: not in_blocks(n)) == count(shared, lambda n: not in_blocks(n))
    assert mot.summary()["experts"] == ["state", "action"]
    assert shared.summary()["experts"] is None
