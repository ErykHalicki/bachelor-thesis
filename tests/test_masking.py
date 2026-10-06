import torch

from thesis.algorithms.masking import (
    block_causal_flex_mask,
    block_causal_mask,
    block_intervals,
)


def make_mask(*modalities):
    """Each modality is (indices, fps, block_size); concatenated in sequence order."""
    starts, ends = zip(*(block_intervals(idx, fps, bs) for idx, fps, bs in modalities))
    return block_causal_mask(torch.cat(starts), torch.cat(ends))


def test_block_size_one_is_causal():
    mask = make_mask((range(5), 1.0, 1))
    assert torch.equal(mask, torch.ones(5, 5, dtype=torch.bool).tril())


def test_within_block_full_attention():
    mask = make_mask((range(4), 1.0, 2))
    expected = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ], dtype=torch.bool)
    assert torch.equal(mask, expected)


def test_negative_indices_block_anchoring():
    """Blocks are anchored at t=0: [-2,-1] and [0,1] pair up, no straddling."""
    mask = make_mask(([-2, -1, 0, 1], 1.0, 2))
    assert mask[0, 1] and mask[1, 0]
    assert mask[2, 3] and mask[3, 2]
    assert not mask[0, 2] and not mask[1, 2]


def test_cross_rate_alignment():
    """Video at 1 step/s, actions at 3 steps/s, sequence [video | action]."""
    mask = make_mask((range(2), 1.0, 1), (range(6), 3.0, 1))
    video, action = 0, 2

    assert mask[action + 2, video + 0]
    assert not mask[action + 2, video + 1]
    assert mask[action + 3, video + 1]

    assert mask[video + 0, action + 2]
    assert not mask[video + 0, action + 3]
    assert torch.all(mask[video + 1, action : action + 6])


def test_boundary_exactness_across_fps():
    """5 fps video vs 30 fps actions: boundaries equal in exact math (1.6 s) never attend."""
    mask = make_mask(([7, 8], 5.0, 1), ([47, 48], 30.0, 1))
    video, action = 0, 2

    assert not mask[action + 0, video + 1]
    assert not mask[video + 0, action + 1]
    assert mask[action + 1, video + 1]
    assert mask[video + 1, action + 0]


def test_spatial_tokens_share_their_timestep_block():
    spatial = 3
    indices = torch.tensor([0, 1]).repeat_interleave(spatial)
    mask = make_mask((indices, 1.0, 1))
    assert torch.all(mask[:spatial, :spatial])
    assert not torch.any(mask[:spatial, spatial:])
    assert torch.all(mask[spatial:, :])


def test_always_visible_token_via_inf_start():
    start, end = block_intervals(range(3), 1.0)
    start = torch.cat([start, torch.tensor([-torch.inf], dtype=torch.float64)])
    end = torch.cat([end, torch.tensor([torch.inf], dtype=torch.float64)])
    mask = block_causal_mask(start, end)
    assert torch.all(mask[:, 3])
    assert torch.all(mask[3, :])


def test_flex_mask_matches_dense():
    start, end = block_intervals(torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]), 2.0, 2)
    dense = block_causal_mask(start, end)
    flex = block_causal_flex_mask(start, end, flex_block_size=1)
    assert torch.equal(flex.to_dense().squeeze().bool(), dense)


def role_rule_masks():
    """Clean context [-1, 0] at 1 fps, noisy chunk [0, 1, 2] at 3 fps: the clean token
    at step 0 spans [0, 1)s and overlaps the whole chunk under the time predicate."""
    s0, e0 = block_intervals([-1, 0], 1.0)
    s1, e1 = block_intervals([0, 1, 2], 3.0)
    start, end = torch.cat([s0, s1]), torch.cat([e0, e1])
    blocked_q = torch.tensor([True, True, False, False, False])
    noisy_kv = ~blocked_q
    return start, end, blocked_q, noisy_kv


def test_role_rule_blocks_clean_to_noisy_only():
    start, end, blocked_q, noisy_kv = role_rule_masks()
    base = block_causal_mask(start, end)
    mask = block_causal_mask(start, end, blocked_q, noisy_kv)
    assert torch.all(base[1, 2:])
    assert not torch.any(mask[:2, 2:])
    assert torch.equal(mask[:, :2], base[:, :2])
    assert torch.equal(mask[2:], base[2:])


def test_role_rule_flex_matches_dense():
    start, end, blocked_q, noisy_kv = role_rule_masks()
    dense = block_causal_mask(start, end, blocked_q, noisy_kv)
    flex = block_causal_flex_mask(start, end, blocked_q, noisy_kv, flex_block_size=1)
    assert torch.equal(flex.to_dense().squeeze().bool(), dense)
