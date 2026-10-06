import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from thesis.datasets import build_dataset
from thesis.datasets.lerobot import split_episodes
from thesis.experiments.eval.offline import OfflineEval


def test_split_is_disjoint_and_covers_every_episode():
    train = split_episodes(range(20), 0.25, "train")
    val = split_episodes(range(20), 0.25, "val")
    assert len(val) == 5
    assert len(train) == 15
    assert not set(train) & set(val)
    assert sorted(train + val) == list(range(20))


def test_split_is_stable_across_calls_but_moves_with_the_seed():
    first = split_episodes(range(20), 0.25, "val", seed=0)
    assert first == split_episodes(range(20), 0.25, "val", seed=0)
    assert first != split_episodes(range(20), 0.25, "val", seed=1)


def test_split_draws_rather_than_slicing_the_tail():
    """Episodes are recorded in session order, so a tail slice would hold out a
    correlated block rather than a sample of the dataset."""
    val = split_episodes(range(100), 0.1, "val")
    assert val != list(range(90, 100))


def test_split_applies_within_an_explicit_episode_list():
    episodes = [3, 5, 8, 11, 20]
    train = split_episodes(episodes, 0.2, "train")
    val = split_episodes(episodes, 0.2, "val")
    assert len(val) == 1
    assert sorted(train + val) == episodes


def test_a_nonzero_fraction_always_holds_out_at_least_one_episode():
    # 0.1 of 5 rounds to 0, but asking for a split and getting none would silently
    # score the training data
    assert len(split_episodes(range(5), 0.1, "val")) == 1


def test_split_never_holds_out_the_last_episode():
    assert split_episodes(range(3), 0.99, "train") == split_episodes(range(3), 0.99, "train")
    assert len(split_episodes(range(3), 0.99, "train")) == 1


def test_zero_fraction_leaves_training_whole_and_refuses_a_val_split():
    assert split_episodes(range(10), 0.0, "train") == list(range(10))
    with pytest.raises(ValueError, match="holds out no episodes"):
        split_episodes(range(10), 0.0, "val")


def test_out_of_range_fraction_is_rejected():
    for bad in (-0.1, 1.0, 2.0):
        with pytest.raises(ValueError, match="validation_split"):
            split_episodes(range(10), bad, "train")


def test_unknown_split_name_is_rejected():
    with pytest.raises(ValueError, match="split must be"):
        split_episodes(range(10), 0.1, "test")


def test_split_is_rejected_for_backends_without_episodes():
    cfg = OmegaConf.create({"backend": "dummy", "n": 4, "obs_dim": 2, "target_dim": 1})
    with pytest.raises(ValueError, match="only the lerobot backend"):
        build_dataset(cfg, split="val")


class _CountingModel(torch.nn.Module):
    """Reports a per-item loss of 1.0 and counts the items it saw, so the eval's
    averaging and its batch cap can both be checked."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.seen = 0

    def loss(self, batch):
        self.seen += len(batch["x"])
        return {
            "loss": torch.tensor(1.0) + self.weight.sum(),
            "loss/term": torch.tensor(0.25),
            "ignored": torch.tensor(99.0),
        }


class _FakeSource:
    def __init__(self, n=10):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"x": torch.full((3,), float(idx))}


@pytest.fixture
def offline_cfg(monkeypatch):
    import thesis.datasets as datasets_mod

    calls = {}

    def fake_build(cfg, norm_stats_override=None, augment=True, split=None):
        calls.update(augment=augment, split=split, stats=norm_stats_override)
        return _FakeSource()

    monkeypatch.setattr(datasets_mod, "build_dataset", fake_build)
    return calls


def test_offline_eval_averages_loss_over_the_held_out_split(offline_cfg):
    model = _CountingModel()
    model.norm_stats = {"x": {"mean": [0.0], "std": [1.0]}}
    cfg = OmegaConf.create({"backend": "offline", "dataset": {}, "batch_size": 4})
    result = OfflineEval(cfg).run(model)
    assert result.metrics["loss"] == pytest.approx(1.0)
    assert result.metrics["loss/term"] == pytest.approx(0.25)
    assert result.metrics["holdout_samples"] == 10
    assert "ignored" not in result.metrics


def test_offline_eval_scores_the_val_split_unaugmented_with_training_stats(offline_cfg):
    model = _CountingModel()
    model.norm_stats = {"x": {"mean": [0.0], "std": [1.0]}}
    OfflineEval(OmegaConf.create({"backend": "offline", "dataset": {}})).run(model)
    assert offline_cfg["split"] == "val"
    assert offline_cfg["augment"] is False
    assert offline_cfg["stats"] == model.norm_stats


def test_max_batches_caps_the_work(offline_cfg):
    model = _CountingModel()
    cfg = OmegaConf.create(
        {"backend": "offline", "dataset": {}, "batch_size": 2, "max_batches": 3}
    )
    result = OfflineEval(cfg).run(model)
    assert model.seen == 6
    assert result.metrics["holdout_samples"] == 6


def _capped_sample(offline_cfg, seed):
    """Which dataset items a capped run actually scores. _FakeSource returns its index
    as data, so the batch contents name them."""
    seen = []

    class _Recorder(_CountingModel):
        def loss(self, batch):
            seen.extend(batch["x"][:, 0].tolist())
            return super().loss(batch)

    cfg = OmegaConf.create({
        "backend": "offline", "dataset": {}, "batch_size": 2, "max_batches": 2, "seed": seed,
    })
    OfflineEval(cfg).run(_Recorder())
    return seen


def test_a_capped_run_samples_the_split_rather_than_its_first_episodes(offline_cfg):
    # samples are laid out episode by episode, so scoring items 0..3 would score
    # the lowest-indexed held-out episode and nothing else
    assert _capped_sample(offline_cfg, seed=0) != [0.0, 1.0, 2.0, 3.0]


def test_the_capped_subset_is_the_same_for_every_checkpoint(offline_cfg):
    assert _capped_sample(offline_cfg, seed=0) == _capped_sample(offline_cfg, seed=0)
    assert _capped_sample(offline_cfg, seed=0) != _capped_sample(offline_cfg, seed=7)


def test_eval_restores_training_mode_and_leaves_the_global_rng_alone(offline_cfg):
    model = _CountingModel()
    model.train()
    torch.manual_seed(1234)
    before = torch.randn(4)
    torch.manual_seed(1234)
    OfflineEval(OmegaConf.create({"backend": "offline", "dataset": {}})).run(model)
    assert torch.equal(torch.randn(4), before)
    assert model.training


def test_empty_holdout_is_an_error_not_a_zero(monkeypatch):
    import thesis.datasets as datasets_mod

    monkeypatch.setattr(
        datasets_mod, "build_dataset",
        lambda cfg, norm_stats_override=None, augment=True, split=None: _FakeSource(0),
    )
    cfg = OmegaConf.create({"backend": "offline", "dataset": {"repo_id": "x/y"}})
    with pytest.raises(ValueError, match="no samples"):
        OfflineEval(cfg).run(_CountingModel())



pytest.importorskip("lerobot")


@pytest.fixture
def split_root(tmp_path):
    """Ten single-frame-per-step episodes, each carrying its own episode index as data,
    so a built split can be traced back to the episodes it actually opened."""
    from lerobot.datasets import LeRobotDataset

    root = tmp_path / "split_arm"
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["j.pos"]},
        "action": {"dtype": "float32", "shape": (1,), "names": ["j.pos"]},
    }
    ds = LeRobotDataset.create("test/split_arm", fps=10, features=features, root=root)
    for episode in range(10):
        for _ in range(4):
            ds.add_frame({
                "observation.state": np.array([float(episode)], dtype=np.float32),
                "action": np.array([float(episode)], dtype=np.float32),
                "task": "fake",
            })
        ds.save_episode()
    ds.finalize()
    return root


def _split_cfg(root, **overrides):
    return OmegaConf.create({
        "backend": "lerobot",
        "repo_id": "test/split_arm",
        "root": str(root),
        "conditioning": {"observation.state": {"index": "0..0"}},
        "predict": {"action": {"index": "0..0"}},
        **overrides,
    })


def _episodes_in(dataset):
    return {
        float(dataset[i]["observation.state"].reshape(-1)[0]) for i in range(len(dataset))
    }


def test_train_and_val_sources_open_disjoint_episodes(split_root):
    cfg = _split_cfg(split_root, validation_split=0.2)
    train = _episodes_in(build_dataset(cfg, split="train"))
    val = _episodes_in(build_dataset(cfg, split="val"))
    assert len(val) == 2
    assert not train & val
    assert train | val == {float(e) for e in range(10)}


def test_no_split_keeps_every_episode(split_root):
    cfg = _split_cfg(split_root, validation_split=0.2)
    assert len(_episodes_in(build_dataset(cfg))) == 10
