import torch

from thesis.algorithms.attention import CrossAttention, SelfAttention
from thesis.algorithms.masking import (
    block_causal_flex_mask,
    block_causal_mask,
    block_intervals,
)
from thesis.algorithms.rope import RopeND, axes_from_positions, make_pos_ids

HIDDEN, HEADS, N = 64, 4, 6


def test_self_attention_respects_causal_mask():
    """Perturbing a token must not change outputs of tokens whose block precedes it."""
    torch.manual_seed(0)
    sa = SelfAttention(HIDDEN, HEADS)
    start, end = block_intervals(range(N), fps=1.0)
    mask = block_causal_mask(start, end)

    x = torch.randn(1, N, HIDDEN)
    perturbed = x.clone()
    perturbed[0, 4] += 1.0
    out, out_p = sa(x, attn_mask=mask), sa(perturbed, attn_mask=mask)

    assert torch.allclose(out[0, :4], out_p[0, :4], atol=1e-6)
    assert not torch.allclose(out[0, 4:], out_p[0, 4:], atol=1e-6)


@torch.no_grad()
def test_self_attention_flex_matches_dense():
    """flex_attention has no CPU backward, so this parity check runs without grad."""
    torch.manual_seed(0)
    sa = SelfAttention(HIDDEN, HEADS)
    start, end = block_intervals(range(N), fps=1.0, block_size=2)
    x = torch.randn(2, N, HIDDEN)
    dense_out = sa(x, attn_mask=block_causal_mask(start, end))
    flex_out = sa(x, attn_mask=block_causal_flex_mask(start, end, flex_block_size=1))
    assert torch.allclose(dense_out, flex_out, atol=1e-5)


def test_self_attention_with_rope_freqs():
    """Shared time axis end to end: freqs from the same ids the mask is built from."""
    sa = SelfAttention(HIDDEN, HEADS)
    pos = make_pos_ids(torch.arange(N).float() / 2.0)
    axes = axes_from_positions(pos, {"time": {"share": 1.0, "period": "auto"}})
    freqs = RopeND(HIDDEN // HEADS, axes)(pos)
    x = torch.randn(2, N, HIDDEN)
    out = sa(x, freqs=freqs)
    assert out.shape == (2, N, HIDDEN)
    assert not torch.allclose(out, sa(x))


def test_cross_attention_multiple_sources():
    ca = CrossAttention(HIDDEN, HEADS, source_dims=[4, 6])
    x = torch.randn(2, N, HIDDEN)
    sources = [torch.randn(2, 1, 4), torch.randn(2, 5, 6)]
    assert ca(x, sources).shape == (2, N, HIDDEN)


def test_cross_attention_none_source_skipped():
    torch.manual_seed(0)
    ca = CrossAttention(HIDDEN, HEADS, source_dims=[4, 6])
    x = torch.randn(2, N, HIDDEN)
    sources = [torch.randn(2, 1, 4), None]
    assert ca(x, sources).shape == (2, N, HIDDEN)


def test_cross_attention_padding_masked_out():
    """Content of padded source tokens must not affect the output."""
    torch.manual_seed(0)
    ca = CrossAttention(HIDDEN, HEADS, source_dims=[4, 6])
    x = torch.randn(2, N, HIDDEN)
    state = torch.randn(2, 1, 4)
    lang = torch.randn(2, 5, 6)
    pad = torch.zeros(2, 5, dtype=torch.bool)
    pad[:, -2:] = True

    out = ca(x, [state, lang], source_masks=[None, pad])
    lang_altered = lang.clone()
    lang_altered[:, -2:] += 10.0
    out_altered = ca(x, [state, lang_altered], source_masks=[None, pad])
    assert torch.allclose(out, out_altered, atol=1e-6)


def test_cross_attention_kv_cache():
    torch.manual_seed(0)
    ca = CrossAttention(HIDDEN, HEADS, source_dims=[4])
    x = torch.randn(1, N, HIDDEN)
    src = [torch.randn(1, 3, 4)]

    out = ca(x, src, use_kv_cache=True)
    garbage = [torch.randn(1, 3, 4)]
    assert torch.allclose(ca(x, garbage, use_kv_cache=True), out)
    assert torch.allclose(ca(x, None, use_kv_cache=True), out)

    ca.clear_cache()
    assert not torch.allclose(ca(x, garbage, use_kv_cache=True), out)
