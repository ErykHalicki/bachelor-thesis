import torch

from thesis.utils.normalize import Normalizer, NormalizedSource, compute_stats


class _ToyDataset:
    def __init__(self, values):
        self.values = values
        self.provided_modalities = {"x"}

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return {"x": self.values[idx]}


def test_compute_stats_matches_manual_mean_std():
    values = [torch.tensor([[float(i), float(2 * i)]]) for i in range(10)]
    stats = compute_stats(_ToyDataset(values), keys=["x"])
    stacked = torch.cat(values)
    assert torch.allclose(torch.tensor(stats["x"]["mean"]), stacked.mean(0), atol=1e-4)
    assert torch.allclose(torch.tensor(stats["x"]["std"]), stacked.std(0, unbiased=False), atol=1e-4)


def test_normalize_unnormalize_roundtrip_mean_std():
    stats = {"x": {"mean": [1.0, 2.0], "std": [2.0, 4.0], "min": [-1.0, -2.0], "max": [3.0, 6.0]}}
    norm = Normalizer(stats, method="mean_std")
    raw = torch.tensor([5.0, 10.0])
    normalized = norm.normalize("x", raw)
    assert torch.allclose(norm.unnormalize("x", normalized), raw)


def test_normalize_unnormalize_roundtrip_min_max():
    stats = {"x": {"mean": [1.0, 2.0], "std": [2.0, 4.0], "min": [-1.0, -2.0], "max": [3.0, 6.0]}}
    norm = Normalizer(stats, method="min_max")
    raw = torch.tensor([0.0, 1.0])
    normalized = norm.normalize("x", raw)
    assert normalized.abs().max() <= 1.0 + 1e-5
    assert torch.allclose(norm.unnormalize("x", normalized), raw, atol=1e-5)


def test_normalize_unnormalize_roundtrip_percentile():
    stats = {"x": {"q_lo": [-2.0, 0.0], "q_hi": [2.0, 10.0]}}
    norm = Normalizer(stats, method="percentile")
    raw = torch.tensor([0.0, 5.0])
    normalized = norm.normalize("x", raw)
    assert torch.allclose(normalized, torch.tensor([0.0, 0.0]), atol=1e-5)
    assert torch.allclose(norm.unnormalize("x", normalized), raw, atol=1e-5)


def test_percentile_does_not_clamp_outliers():
    """The tails outside q_lo/q_hi must survive normalization, and round-trip."""
    stats = {"x": {"q_lo": [-2.0, 0.0], "q_hi": [2.0, 10.0]}}
    norm = Normalizer(stats, method="percentile")
    outlier = torch.tensor([100.0, -100.0])
    normalized = norm.normalize("x", outlier)
    assert torch.equal(normalized, torch.tensor([50.0, -21.0]))
    assert torch.allclose(norm.unnormalize("x", normalized), outlier)


def test_compute_stats_adds_quantiles_when_requested():
    values = [torch.tensor([[float(i)]]) for i in range(101)]
    stats = compute_stats(_ToyDataset(values), keys=["x"], percentiles=[1, 99])
    assert "q_lo" in stats["x"] and "q_hi" in stats["x"]
    assert abs(stats["x"]["q_lo"][0] - 1.0) < 1e-6
    assert abs(stats["x"]["q_hi"][0] - 99.0) < 1e-6


def test_unknown_key_passes_through_unchanged():
    norm = Normalizer({"x": {"mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]}})
    raw = torch.tensor([42.0])
    assert torch.equal(norm.normalize("y", raw), raw)


def test_normalized_source_wraps_batches():
    values = [torch.tensor([[float(i)]]) for i in range(20)]
    source = NormalizedSource(_ToyDataset(values), keys=["x"])
    assert len(source) == 20
    batch = source[0]
    assert set(batch) == {"x"}
    assert source.provided_modalities == {"x"}
