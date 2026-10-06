"""Upload-safety logic for the b601 recorder: how a local recording folder is classified
against its Hub copy, and which classifications are allowed to push.

The local and remote sides are both real lerobot-shaped meta/episodes parquet files, so
the fingerprinting that decides "is the Hub a prefix of this?" is exercised for real; only
the Hub round trip itself is stubbed."""

import types

import pytest

pytest.importorskip("lerobot")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

record = pytest.importorskip("thesis.scripts.b601.record")
common = pytest.importorskip("thesis.scripts.b601.common")
session = pytest.importorskip("thesis.utils.session")


def episode_row(index, length, *, content=0.0, chunk=0, file=0):
    """One row of lerobot's per-episode metadata. `content` stands in for the frames
    themselves: it moves the per-feature stats, as recording a different take would."""
    return {
        "episode_index": index,
        "tasks": ["pick up the charger"],
        "length": length,
        "dataset_from_index": 0,
        "dataset_to_index": length,
        "data/chunk_index": chunk,
        "data/file_index": file,
        "videos/observation.images.zed_left/chunk_index": chunk,
        "videos/observation.images.zed_left/file_index": file,
        "videos/observation.images.zed_left/from_timestamp": 0.0,
        "videos/observation.images.zed_left/to_timestamp": length / 30,
        "stats/action/mean": [content, content + 1.0],
        "stats/action/std": [0.5, 0.25],
        "stats/observation.state/mean": [content - 1.0, content],
        "meta/episodes/chunk_index": chunk,
        "meta/episodes/file_index": file,
    }


def write_episodes(root, rows, *, chunk=0, file=0):
    path = root / "meta" / "episodes" / f"chunk-{chunk:03d}" / f"file-{file:03d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def make_dataset(tmp_path, name, episodes):
    """A dataset folder holding `episodes` as (length, content) pairs, plus the stand-in
    for the LeRobotDataset object compare_with_hub is handed."""
    root = tmp_path / name
    write_episodes(root, [episode_row(i, length, content=c) for i, (length, c) in enumerate(episodes)])
    return types.SimpleNamespace(
        root=root,
        num_episodes=len(episodes),
        meta=types.SimpleNamespace(total_episodes=len(episodes)),
    )


REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Stub the Hub round trip, so classification is tested without network access. The
    remote side is still fingerprinted from real parquet files."""

    def set_remote(episodes=(), no_repo=False, unreachable=None):
        def fetch(_repo_id):
            if unreachable is not None:
                raise record.HubUnreachable(unreachable)
            if no_repo:
                return None
            remote_root = tmp_path / "remote"
            write_episodes(
                remote_root,
                [episode_row(i, length, content=c) for i, (length, c) in enumerate(episodes)],
            )
            fingerprints = record.episode_fingerprints(record.episode_metadata_files(remote_root))
            return record.RemoteEpisodes(REVISION, fingerprints)

        monkeypatch.setattr(record, "fetch_remote_episodes", fetch)

    return set_remote


def test_no_repo_on_hub_is_safe(hub, tmp_path):
    hub(no_repo=True)
    c = record.compare_with_hub(make_dataset(tmp_path, "ds", [(10, 1.0), (20, 2.0)]), "me/ds")
    assert c.status == record.HUB_NO_REPO
    assert c.safe_to_push


def test_hub_holding_an_exact_prefix_is_safe_to_append(hub, tmp_path):
    hub(episodes=[(1625, 1.0), (121, 2.0)])
    local = [(1625, 1.0), (121, 2.0), (469, 3.0), (311, 4.0), (438, 5.0), (456, 6.0)]
    c = record.compare_with_hub(make_dataset(tmp_path, "ds", local), "me/ds")
    assert c.status == record.HUB_BEHIND
    assert c.safe_to_push
    assert (c.local_episodes, c.remote_episodes) == (6, 2)
    assert c.revision == REVISION


def test_identical_dataset_is_in_sync(hub, tmp_path):
    hub(episodes=[(10, 1.0), (20, 2.0)])
    c = record.compare_with_hub(make_dataset(tmp_path, "ds", [(10, 1.0), (20, 2.0)]), "me/ds")
    assert c.status == record.HUB_IN_SYNC
    assert c.safe_to_push


def test_hub_with_extra_episodes_is_unsafe(hub, tmp_path):
    hub(episodes=[(10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0)])
    c = record.compare_with_hub(make_dataset(tmp_path, "ds", [(10, 1.0), (20, 2.0)]), "me/ds")
    assert c.status == record.HUB_AHEAD
    assert not c.safe_to_push


def test_same_counts_and_lengths_but_different_takes_is_diverged(hub, tmp_path):
    # what counting frames could never see: two sessions with the same episode
    # count and lengths, of different data
    hub(episodes=[(10, 1.0), (20, 2.0)])
    c = record.compare_with_hub(make_dataset(tmp_path, "ds", [(10, 1.0), (20, 9.0)]), "me/ds")
    assert c.status == record.HUB_DIVERGED
    assert not c.safe_to_push
    assert c.first_divergent_episode == 1


def test_larger_local_but_divergent_prefix_is_not_treated_as_append(hub, tmp_path):
    # the trap the episode-count check fell into: local is bigger, so it looks like
    # a clean append, but the shared episodes are different recordings
    hub(episodes=[(10, 1.0), (20, 2.0)])
    local = [(10, 7.0), (20, 8.0), (30, 3.0), (40, 4.0)]
    c = record.compare_with_hub(make_dataset(tmp_path, "ds", local), "me/ds")
    assert c.status == record.HUB_DIVERGED
    assert not c.safe_to_push
    assert c.first_divergent_episode == 0


def test_unreachable_hub_is_never_mistaken_for_an_empty_one(hub, tmp_path):
    hub(unreachable="connection reset")
    c = record.compare_with_hub(make_dataset(tmp_path, "ds", [(10, 1.0)]), "me/ds")
    assert c.status == record.HUB_UNKNOWN
    assert not c.safe_to_push
    assert "connection reset" in (c.error or "")


def test_unreadable_local_metadata_is_unknown_rather_than_raising(hub, tmp_path):
    hub(episodes=[(10, 1.0)])
    dataset = make_dataset(tmp_path, "ds", [(10, 1.0)])
    record.episode_metadata_files(dataset.root)[0].write_bytes(b"not a parquet file")
    c = record.compare_with_hub(dataset, "me/ds")
    assert c.status == record.HUB_UNKNOWN
    assert not c.safe_to_push


def test_fingerprints_follow_episode_content_not_its_file_layout(tmp_path):
    # re-chunking a dataset must not make it look like different data
    one = tmp_path / "one"
    write_episodes(one, [episode_row(i, 10 * i + 5, content=float(i)) for i in range(3)])
    split = tmp_path / "split"
    write_episodes(split, [episode_row(i, 10 * i + 5, content=float(i), file=1) for i in range(2)], file=1)
    write_episodes(split, [episode_row(2, 25, content=2.0, chunk=1)], chunk=1)

    assert record.episode_fingerprints(record.episode_metadata_files(one)) == record.episode_fingerprints(
        record.episode_metadata_files(split)
    )


def test_fingerprints_are_ordered_by_episode_index_across_files(tmp_path):
    root = tmp_path / "ds"
    # a resumed session opens a new metadata file, and the Hub may serve them in any order
    write_episodes(root, [episode_row(2, 30, content=3.0)], file=1)
    write_episodes(root, [episode_row(0, 10, content=1.0), episode_row(1, 20, content=2.0)], file=0)
    reference = tmp_path / "ref"
    write_episodes(reference, [episode_row(i, 10 * (i + 1), content=float(i + 1)) for i in range(3)])

    assert record.episode_fingerprints(record.episode_metadata_files(root)) == record.episode_fingerprints(
        record.episode_metadata_files(reference)
    )


@pytest.mark.parametrize(
    "status",
    [
        record.HUB_NO_REPO,
        record.HUB_IN_SYNC,
        record.HUB_BEHIND,
        record.HUB_AHEAD,
        record.HUB_DIVERGED,
        record.HUB_UNKNOWN,
    ],
)
def test_every_status_reports_all_three_counts_and_a_verdict(status):
    c = record.HubComparison(
        status,
        local_episodes=6,
        remote_episodes=2,
        revision=REVISION,
        first_divergent_episode=1,
        error="boom",
    )
    lines = record.describe_hub_comparison(c, "me/ds", recorded_episodes=2)
    assert len(lines) >= 2, f"{status} produced no verdict line"
    assert "local: 6" in lines[0]
    assert "recorded this session: 2" in lines[0]
    unsafe = status in (record.HUB_AHEAD, record.HUB_DIVERGED)
    assert ("WARNING" in " ".join(lines)) == unsafe


def test_comparable_statuses_name_the_commit_they_were_compared_against():
    for status in (record.HUB_IN_SYNC, record.HUB_BEHIND, record.HUB_AHEAD, record.HUB_DIVERGED):
        c = record.HubComparison(
            status, 6, remote_episodes=2, revision=REVISION, first_divergent_episode=1
        )
        body = " ".join(record.describe_hub_comparison(c, "me/ds", recorded_episodes=2)[1:])
        assert REVISION[:7] in body, f"{status} did not name the commit"


def test_behind_message_counts_the_gap_not_the_session():
    c = record.HubComparison(record.HUB_BEHIND, 6, remote_episodes=2, revision=REVISION)
    lines = record.describe_hub_comparison(c, "me/ds", recorded_episodes=2)
    body = " ".join(lines[1:])
    assert "gains 4 episode(s)" in body
    assert "for 6 total" in body


# confirm/confirm_phrase live in common.py, so the stdin they read has to be
# stubbed there
def test_confirm_treats_anything_but_yes_as_no(monkeypatch):
    monkeypatch.setattr(session, "flush_pending_stdin", lambda: None)
    for answer, expected in [("y", True), ("yes", True), ("YES", True),
                             ("", False), ("n", False), ("sure", False)]:
        monkeypatch.setattr(session, "prompt", lambda _text, a=answer: a)
        assert record.confirm("go?") is expected


def test_confirm_phrase_rejects_a_reflexive_yes(monkeypatch):
    monkeypatch.setattr(session, "flush_pending_stdin", lambda: None)
    for answer, expected in [("overwrite", True), ("y", False), ("", False), ("Overwrite", False)]:
        monkeypatch.setattr(session, "prompt", lambda _text, a=answer: a)
        assert record.confirm_phrase("risky", "overwrite") is expected


def test_archive_moves_the_folder_aside_instead_of_deleting(tmp_path):
    root = tmp_path / "me" / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text('{"total_episodes": 3}')

    archived = record.archive_dataset_root(root)

    assert not root.exists()
    assert archived.parent == root.parent
    assert archived.name.startswith("ds.superseded-")
    assert (archived / "meta" / "info.json").read_text() == '{"total_episodes": 3}'
