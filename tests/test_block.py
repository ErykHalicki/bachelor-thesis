import torch

from thesis.algorithms.block import DiTBlock
from thesis.algorithms.masking import block_causal_mask, block_intervals
from thesis.algorithms.rope import RopeND, axes_from_positions, make_pos_ids

HIDDEN, HEADS, N = 64, 4, 6


def test_identity_at_init_without_sources():
    torch.manual_seed(0)
    block = DiTBlock(HIDDEN, HEADS)
    x = torch.randn(2, N, HIDDEN)
    cond = torch.randn(2, HIDDEN)
    assert torch.allclose(block(x, cond), x, atol=1e-6)


def test_cross_attention_ignores_rope():
    """Cross-attention must read its sources position-free.

    At init the zero-init adaRMS gates close the self-attention and MLP branches, so the
    block output is exactly `x + cross_attention` and the cross branch is directly
    observable. Feeding two different position fields must not move it. This used to fail:
    the block passed `freqs` into cross-attention, which rotated Q against unrotated K and
    so encoded the query's absolute position -- not an offset, and not what RoPE means
    anywhere else in the model.
    """
    torch.manual_seed(0)
    block = DiTBlock(HIDDEN, HEADS, source_dims=[4]).eval()
    x, cond = torch.randn(2, N, HIDDEN), torch.randn(2, HIDDEN)
    sources = [torch.randn(2, 5, 4)]

    def run(scale):
        pos = make_pos_ids(torch.arange(N).float() * scale)
        axes = axes_from_positions(pos, {"time": {"share": 1.0, "period": "auto"}})
        return block(x, cond, sources=sources,
                     freqs=RopeND(HIDDEN // HEADS, axes)(pos))

    assert torch.equal(run(1.0), run(7.0))


def test_cross_attention_branch_active_at_init():
    torch.manual_seed(0)
    block = DiTBlock(HIDDEN, HEADS, source_dims=[4, 6])
    x = torch.randn(2, N, HIDDEN)
    cond = torch.randn(2, HIDDEN)
    sources = [torch.randn(2, 1, 4), torch.randn(2, 5, 6)]
    out = block(x, cond, sources=sources)
    assert out.shape == x.shape
    assert not torch.allclose(out, x, atol=1e-6)


def test_full_forward_with_rope_and_mask():
    torch.manual_seed(0)
    block = DiTBlock(HIDDEN, HEADS, source_dims=[4])
    indices = torch.arange(N)
    pos = make_pos_ids(indices.float() / 2.0)
    axes = axes_from_positions(pos, {"time": {"share": 1.0, "period": "auto"}})
    freqs = RopeND(HIDDEN // HEADS, axes)(pos)
    mask = block_causal_mask(*block_intervals(indices, fps=2.0))

    x = torch.randn(2, N, HIDDEN)
    cond = torch.randn(2, HIDDEN)
    out = block(x, cond, sources=[torch.randn(2, 3, 4)], freqs=freqs, attn_mask=mask)
    assert out.shape == (2, N, HIDDEN)


def test_gates_open_after_training_step():
    torch.manual_seed(0)
    block = DiTBlock(HIDDEN, HEADS)
    x = torch.randn(2, N, HIDDEN)
    cond = torch.randn(2, HIDDEN)

    opt = torch.optim.SGD(block.parameters(), lr=0.1)
    loss = (block(x, cond) - torch.ones_like(x)).pow(2).mean()
    loss.backward()
    opt.step()

    assert not torch.allclose(block(x, cond), x, atol=1e-6)


def test_kv_cache_through_block():
    torch.manual_seed(0)
    block = DiTBlock(HIDDEN, HEADS, source_dims=[4])
    x = torch.randn(1, N, HIDDEN)
    cond = torch.randn(1, HIDDEN)
    src = [torch.randn(1, 3, 4)]

    out = block(x, cond, sources=src, use_kv_cache=True)
    garbage = [torch.randn(1, 3, 4)]
    assert torch.allclose(block(x, cond, sources=garbage, use_kv_cache=True), out)
    assert torch.allclose(block(x, cond, sources=None, use_kv_cache=True), out)

    block.clear_cache()
    assert not torch.allclose(block(x, cond, sources=garbage, use_kv_cache=True), out)