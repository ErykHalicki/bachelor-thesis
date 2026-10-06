import pytest
import torch

from thesis.algorithms.vit import ViT

ROPE = {"time": {"share": 0.5, "period": "auto"},
        "height": {"share": 0.25, "period": "auto"},
        "width": {"share": 0.25, "period": "auto"}}


def _vit(**kw):
    kw = {"in_channels": 3, "dim": 32, "patch_size": 8, "depth": 1, "num_heads": 4,
          "rope": ROPE, "img_size": 16, "frames": 3, **kw}
    return ViT(**kw)


def test_rope_ladder_is_fixed_at_build_time():
    """The ladder must not depend on how many frames a call happens to carry."""
    vit = _vit()
    built = {k: v.clone() for k, v in vit.rope.named_buffers()}
    assert built, "the ladder should exist before any forward pass"

    for frames in (1, 2, 3):
        vit(torch.randn(2, frames, 3, 16, 16))
        assert all(torch.equal(v, built[k]) for k, v in vit.rope.named_buffers())


def test_output_shape_follows_the_frame_count():
    vit = _vit()
    for frames in (1, 3):
        out = vit(torch.randn(2, frames, 3, 16, 16))
        assert out.shape == (2, frames * 2 * 2, 32)     # 16/8 = 2x2 patch grid


def test_geometry_beyond_the_declared_bound_is_an_error():
    """Silently rescaling would put the extra positions on a ladder never trained on."""
    vit = _vit()
    with pytest.raises(AssertionError, match="RoPE ladder was built for at most"):
        vit(torch.randn(1, 4, 3, 16, 16))               # more frames than declared
    with pytest.raises(AssertionError, match="RoPE ladder was built for at most"):
        vit(torch.randn(1, 2, 3, 32, 32))               # bigger grid than declared


def test_non_square_img_size_for_a_stitched_stream():
    vit = _vit(patch_size=16, img_size=[32, 64], frames=2)
    assert vit.grid == (2, 4)
    assert vit(torch.randn(2, 2, 3, 32, 64)).shape == (2, 2 * 2 * 4, 32)


def test_geometry_and_rope_must_be_declared():
    # `frames` moved to its own assertion when it became attend_across_time-conditional;
    # a missing img_size now names img_size alone
    with pytest.raises(AssertionError, match="img_size"):
        _vit(img_size=None)
    with pytest.raises(AssertionError, match="`rope:` block"):
        _vit(rope=None)
