"""The encoding cache must reproduce the live encoder path: same windows, same clamping,
same augmentation coherence -- else training silently optimizes against embeddings the
validation/rollout path never produces. These tests pin that equivalence with a stub
frozen encoder and a synthetic source, plus the identity machinery (fingerprints,
storage placement, stride resolution) around it.
"""

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from thesis.utils.augment import Augmenter
from thesis.utils.enc_cache import (EncodingCache, StreamStore, augment_like_draw,
                                   build_encoding_cache, cache_fingerprint,
                                   quantize_windows, resolve_stride)


class StubEncoder(nn.Module):
    """Deterministic stand-in for a frozen backbone: (B, T, C, H, W) uint8 ->
    (B, T*tokens, dim). Linear in the pixels, so quantization error stays interpretable.
    """

    frozen = True
    raw_steps_per_index = 1

    def __init__(self, tokens=3, dim=8):
        super().__init__()
        torch.manual_seed(7)
        self.proj = nn.Linear(12 * 12 * 3, tokens * dim)
        self.tokens, self.dim = tokens, dim

    def forward(self, frames):
        x = frames.float().flatten(2) / 255.0
        return self.proj(x).reshape(frames.shape[0], -1, self.dim)


class FakeSource:
    """Two tiny episodes of random frames, stride-2 anchors -- read_frame/anchor-space
    shaped exactly like LeRobotSource's cache-build surface."""

    _anchor_stride = 2
    _episodes = [0, 1]

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.frames = {
            (ep, t): {"cam": torch.randint(0, 256, (3, 12, 12), generator=g,
                                           dtype=torch.uint8)}
            for ep in (0, 1) for t in range(13)
        }
        self.space = [(0, [2, 4, 6]), (1, [0, 2, 4, 6, 8])]

    def cache_anchor_space(self):
        return self.space

    def read_frame(self, ep, t):
        return self.frames[(ep, t)]


STREAMS = {
    "ctx": {"field": "cam", "offsets": [-4, 0], "encoder": "enc"},
    "fut": {"field": "cam", "offsets": [2, 4], "encoder": "enc"},
}


class FakeAlgo:
    def __init__(self):
        self.encoders = {"enc": StubEncoder()}


def build(tmp_path, draws=1, augment=None):
    return build_encoding_cache(
        FakeSource(), FakeAlgo(), STREAMS, augment, draws=draws, store_on="memory",
        directory=tmp_path / "cache", fingerprint="testfp", device=torch.device("cpu"),
        encode_batch=4, log=lambda *a: None,
    )


def test_lookup_matches_live_encode(tmp_path):
    cache = build(tmp_path)
    src, enc = FakeSource(), StubEncoder()
    for stream, (ep, anchor) in (("ctx", (0, 4)), ("fut", (1, 8)), ("ctx", (1, 0))):
        offsets = STREAMS[stream]["offsets"]
        # anchor 0 with offset -4 clamps onto the episode's first frame, like lerobot
        frames = torch.stack(
            [src.frames[(ep, max(0, anchor + o))]["cam"] for o in offsets]
        )
        with torch.no_grad():
            live = enc(frames.unsqueeze(0))
        got = cache.lookup(stream, torch.tensor([[ep, anchor, 0]]), "cpu")
        rel = (got - live).norm() / live.norm()
        assert rel < 0.02, f"{stream} ({ep},{anchor}): rel err {rel:.4f}"


def test_missing_window_is_a_hard_error(tmp_path):
    cache = build(tmp_path)
    with pytest.raises(KeyError, match="episode 0, anchor 5"):
        cache.lookup("ctx", torch.tensor([[0, 5, 0]]), "cpu")


def test_pixels_can_ride_along_coherently(tmp_path):
    """A pixel window augmented via augment_like_draw must match the augmentation baked
    into the cached draw it rides next to -- the property that keeps a pixel-decoder's
    targets geometrically coherent with its cached conditioning latents.
    """
    augment = {"streams": {"cam": {"random_crop": {"scale": [0.5, 0.8]}}}}
    cache = build(tmp_path, draws=2, augment=augment)
    src, enc = FakeSource(), StubEncoder()
    ep, anchor, k = 0, 4, 1
    frames = torch.stack(
        [src.frames[(ep, max(0, anchor + o))]["cam"] for o in STREAMS["ctx"]["offsets"]]
    )
    pixel_side = augment_like_draw(
        Augmenter(augment["streams"]), "cam", frames, ep, anchor, k
    )
    with torch.no_grad():
        live = enc(pixel_side.unsqueeze(0))
    got = cache.lookup("ctx", torch.tensor([[ep, anchor, k]]), "cpu")
    rel = (got - live).norm() / live.norm()
    assert rel < 0.02, f"cached draw and re-derived pixel draw disagree: {rel:.4f}"


def test_draws_are_deterministic_and_distinct(tmp_path):
    augment = {"streams": {"cam": {"random_crop": {"scale": [0.6, 0.9]}}}}
    a = build(tmp_path / "a", draws=2, augment=augment)
    b = build(tmp_path / "b", draws=2, augment=augment)
    key = torch.tensor([[0, 4, 0]])
    assert torch.equal(a.lookup("ctx", key, "cpu"), b.lookup("ctx", key, "cpu"))
    other = torch.tensor([[0, 4, 1]])
    assert not torch.equal(a.lookup("ctx", key, "cpu"), a.lookup("ctx", other, "cpu"))


def test_save_load_roundtrip_and_fingerprint_guard(tmp_path):
    cache = build(tmp_path)
    out = tmp_path / "saved"
    cache.save(out)
    again = EncodingCache.load(out, "testfp", store_on="memory")
    key = torch.tensor([[1, 6, 0]])
    assert torch.equal(cache.lookup("fut", key, "cpu"),
                       again.lookup("fut", key, "cpu"))
    with pytest.raises(ValueError, match="fingerprint"):
        EncodingCache.load(out, "otherfp")


def test_quantization_error_is_small():
    z = torch.randn(4, 6, 16)
    q, scale, zero = quantize_windows(z)
    back = torch.as_tensor(q).float() * torch.as_tensor(scale).float() \
        + torch.as_tensor(zero).float()
    rel = (back - z).norm(dim=-1) / z.norm(dim=-1)
    assert rel.max() < 0.05


def test_resolve_stride():
    assert resolve_stride(None, 30) == 1
    assert resolve_stride({"a": {"every": 3}}, 30) == 3
    assert resolve_stride({"a": {"fps": 10}}, 30) == 3
    assert resolve_stride({"a": {"every": 4}, "b": {"fps": 5}}, 30) == 12
    with pytest.raises(ValueError, match="does not divide"):
        resolve_stride({"a": {"fps": 7}}, 30)
    with pytest.raises(ValueError, match="exactly one"):
        resolve_stride({"a": {"every": 2, "fps": 15}}, 30)


def test_fingerprint_tracks_every_input():
    base = dict(streams_info=STREAMS, encoder_specs={"enc": {"type": "vit"}},
                augment_cfg=None, stride=6, draws=2, dataset_id={"repo_id": "x"})
    fp = cache_fingerprint(**base)
    assert fp == cache_fingerprint(**base)
    for change in ({"draws": 3}, {"stride": 3},
                   {"augment_cfg": {"streams": {"cam": {}}}},
                   {"dataset_id": {"repo_id": "y"}}):
        assert cache_fingerprint(**{**base, **change}) != fp


def test_fingerprint_ignores_encoders_the_cache_never_runs():
    """The arms of a sweep declare extra encoders -- an action codec, a pixel decoder --
    that touch no cached byte. Hashing the whole map would split one shared cache into one
    per arm, which is the whole cost the cache exists to avoid."""
    base = dict(streams_info=STREAMS, augment_cfg=None, stride=6, draws=2,
                dataset_id={"repo_id": "x"})
    used_only = cache_fingerprint(encoder_specs={"enc": {"type": "vit"}}, **base)
    with_codec = cache_fingerprint(
        encoder_specs={"enc": {"type": "vit"},
                       "act_enc": {"type": "chunk_mlp", "in_dim": 7}},
        **base,
    )
    assert used_only == with_codec
    assert cache_fingerprint(encoder_specs={"enc": {"type": "vit", "depth": 4}},
                             **base) != used_only


def test_cache_dir_is_named_by_contents_alone():
    """Two runs whose cached bytes match must resolve to one directory, so the first to
    build it serves the rest."""
    assert (EncodingCache.cache_dir("/tmp/x", "abc123")
            == EncodingCache.cache_dir("/tmp/x", "abc123"))
    assert (EncodingCache.cache_dir("/tmp/x", "abc123")
            != EncodingCache.cache_dir("/tmp/x", "def456"))


def test_store_placement(monkeypatch):
    import thesis.utils.enc_cache as m
    monkeypatch.setattr(m, "_available_ram_bytes", lambda: 1000)
    assert EncodingCache._store_in_ram("memory", 10 ** 9)
    assert not EncodingCache._store_in_ram("disk", 1)
    assert EncodingCache._store_in_ram("auto", 400)
    assert not EncodingCache._store_in_ram("auto", 600)
    with pytest.raises(ValueError, match="store_on"):
        EncodingCache._store_in_ram("ram", 1)


def test_sorted_keys_invariant(tmp_path):
    cache = build(tmp_path)
    for store in cache.stores.values():
        assert isinstance(store, StreamStore)
        keys = store.keys.numpy()
        assert (np.diff(keys) > 0).all(), "packed keys must be strictly ascending"


@pytest.mark.skipif(not os.path.exists("/proc/meminfo"), reason="reads /proc/meminfo (Linux only)")
def test_available_ram_respects_cgroup_limit(monkeypatch, tmp_path):
    """Inside a pod, /proc/meminfo describes the HOST; the cgroup limit is the number
    that decides whether a cache fits in RAM (the run39 OOM, 2026-08-12)."""
    from thesis.utils import enc_cache

    limit = tmp_path / "memory.max"
    usage = tmp_path / "memory.current"
    limit.write_text("1000000\n")
    usage.write_text("250000\n")
    monkeypatch.setattr(enc_cache, "_CGROUP_MEM", ((str(limit), str(usage)),))
    host = enc_cache._available_ram_bytes()
    assert host == 750000

    limit.write_text("max\n")
    assert enc_cache._available_ram_bytes() > 1 << 30
    limit.write_text(str(1 << 62) + "\n")
    assert enc_cache._available_ram_bytes() > 1 << 30

    monkeypatch.setattr(enc_cache, "_CGROUP_MEM", ((str(tmp_path / "nope"), str(tmp_path / "nope2")),))
    assert enc_cache._available_ram_bytes() > 1 << 30
