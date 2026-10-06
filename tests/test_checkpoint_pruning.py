"""Checkpoint artifact pruning exercised against a fake wandb module: default saves keep
only the newest committed version, in-flight (uncommitted) uploads are
never counted, the `best`-tagged version is never pruned, save_intermittent=True keeps
everything, and pruning failures stay non-fatal."""

import sys
import types

import pytest
import torch
import torch.nn as nn

from thesis.utils.checkpoint import save_checkpoint


class FakeVersion:
    def __init__(self, version, state="COMMITTED", type="model", aliases=()):
        self.version = version
        self.state = state
        self.type = type
        self.aliases = list(aliases)
        self.deleted = False

    def delete(self, delete_aliases=False):
        self.deleted = True


def make_fake_wandb(versions):
    wandb = types.ModuleType("wandb")
    wandb.run = types.SimpleNamespace(id="abc123", entity="ent", project="proj")

    class Artifact:
        def __init__(self, name, type):
            self.name, self.type = name, type

        def add_file(self, path):
            self.path = path

    wandb.Artifact = Artifact
    wandb.logged = []

    def log_artifact(artifact, aliases=None):
        artifact.aliases = aliases
        wandb.logged.append(artifact)

    wandb.log_artifact = log_artifact

    class FakeRun:
        def logged_artifacts(self):
            return list(versions)

    class Api:
        def run(self, path):
            assert path == "ent/proj/abc123"
            return FakeRun()

    wandb.Api = Api
    return wandb


@pytest.fixture
def model_and_opt():
    model = nn.Linear(2, 2)
    return model, torch.optim.SGD(model.parameters(), lr=0.1)


def run_save(tmp_path, monkeypatch, versions, **kwargs):
    fake = make_fake_wandb(versions)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    model, opt = nn.Linear(2, 2), None
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    save_checkpoint(model, opt, tmp_path, step=5, **kwargs)
    return fake


def test_prunes_all_but_newest_committed(tmp_path, monkeypatch):
    versions = [FakeVersion(f"v{i}") for i in range(5)]
    fake = run_save(tmp_path, monkeypatch, versions)
    assert len(fake.logged) == 1
    assert [v.version for v in versions if v.deleted] == ["v0", "v1", "v2", "v3"]
    assert [v.version for v in versions if not v.deleted] == ["v4"]


def test_uncommitted_versions_are_not_counted_or_deleted(tmp_path, monkeypatch):
    # v4 is still uploading: the survivor must be v3, and the in-flight
    # version must never be touched
    versions = [FakeVersion(f"v{i}") for i in range(4)] + [FakeVersion("v4", state="PENDING")]
    run_save(tmp_path, monkeypatch, versions)
    assert [v.version for v in versions if v.deleted] == ["v0", "v1", "v2"]


def test_non_model_artifacts_are_never_touched(tmp_path, monkeypatch):
    # run.logged_artifacts() also returns wandb's own history/events artifacts
    history = FakeVersion("v9", type="run_table")
    versions = [FakeVersion(f"v{i}") for i in range(4)] + [history]
    run_save(tmp_path, monkeypatch, versions)
    assert not history.deleted
    assert [v.version for v in versions if v.deleted] == ["v0", "v1", "v2"]


def test_best_version_is_never_pruned(tmp_path, monkeypatch):
    best = FakeVersion("v0", aliases=["best"])
    versions = [best] + [FakeVersion(f"v{i}") for i in range(1, 5)]
    run_save(tmp_path, monkeypatch, versions)
    assert not best.deleted
    assert [v.version for v in versions if v.deleted] == ["v1", "v2", "v3"]


def test_best_save_writes_best_pt_and_tags_the_alias(tmp_path, monkeypatch):
    fake = run_save(tmp_path, monkeypatch, [], best=True)
    assert (tmp_path / "best.pt").exists()
    assert not (tmp_path / "model.pt").exists()
    assert fake.logged[0].aliases == ["best"]


def test_final_save_tags_the_alias_and_writes_model_pt(tmp_path, monkeypatch):
    fake = run_save(tmp_path, monkeypatch, [], final=True)
    assert (tmp_path / "model.pt").exists()
    assert fake.logged[0].aliases == ["final"]


def test_final_version_is_never_pruned(tmp_path, monkeypatch):
    # a resumed run keeps saving after the original run's end-of-training checkpoint;
    # that `final` version must survive pruning exactly as `best` does
    final = FakeVersion("v0", aliases=["final"])
    versions = [final] + [FakeVersion(f"v{i}") for i in range(1, 5)]
    run_save(tmp_path, monkeypatch, versions)
    assert not final.deleted
    assert [v.version for v in versions if v.deleted] == ["v1", "v2", "v3"]


def test_ordinary_save_carries_no_alias(tmp_path, monkeypatch):
    fake = run_save(tmp_path, monkeypatch, [])
    assert (tmp_path / "model.pt").exists()
    assert fake.logged[0].aliases is None


def test_save_intermittent_keeps_every_version(tmp_path, monkeypatch):
    versions = [FakeVersion(f"v{i}") for i in range(5)]
    fake = run_save(tmp_path, monkeypatch, versions, save_intermittent=True)
    assert len(fake.logged) == 1
    assert not any(v.deleted for v in versions)


def test_pruning_failure_is_non_fatal(tmp_path, monkeypatch, capsys):
    fake = make_fake_wandb([])
    fake.Api = None
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    path = save_checkpoint(model, opt, tmp_path, step=5)
    assert path.exists()
    assert "non-fatal" in capsys.readouterr().out
