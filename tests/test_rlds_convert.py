"""RLDS -> LeRobot v3 conversion on a synthetic dataset, read back through LeRobotV3Source."""
import importlib.util
import io
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
PIL = pytest.importorskip("PIL.Image")
if shutil.which("ffmpeg") is None:
    pytest.skip("needs ffmpeg", allow_module_level=True)

from thesis.datasets.mixture.source import LeRobotV3Source  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "convert_rlds", Path(__file__).resolve().parents[1] / "scripts" / "oxe" / "convert_rlds.py")
rlds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rlds)


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _ld(field, payload):
    return _varint(field << 3 | 2) + _varint(len(payload)) + payload


def _feature(kind, values):
    if kind == "bytes":
        inner = b"".join(_ld(1, v) for v in values)
        return _ld(1, inner)
    if kind == "float":
        return _ld(2, _ld(1, np.asarray(values, "<f4").tobytes()))
    packed = b"".join(_varint(int(v) & (2 ** 64 - 1)) for v in values)
    return _ld(3, _ld(1, packed))


def _example(feats):
    entries = b"".join(_ld(1, _ld(1, k.encode()) + _ld(2, _feature(*v))) for k, v in feats.items())
    return _ld(1, entries)


def _jpeg(rgb):
    b = io.BytesIO()
    PIL.fromarray(rgb).save(b, "PNG")
    return b.getvalue()


FEATURES = {"featuresDict": {"features": {
    "steps": {"pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
              "sequence": {"feature": {"featuresDict": {"features": {
                  "is_first": {"pythonClassName": "x.Scalar", "tensor": {"dtype": "bool"}},
                  "is_last": {"pythonClassName": "x.Scalar", "tensor": {"dtype": "bool"}},
                  "language_instruction": {"pythonClassName": "x.Text", "text": {}},
                  "language_embedding": {"pythonClassName": "x.Tensor", "tensor": {
                      "shape": {"dimensions": ["4"]}, "dtype": "float32"}},
                  "action": {"pythonClassName": "x.Tensor", "tensor": {
                      "shape": {"dimensions": ["3"]}, "dtype": "float32"}},
                  "observation": {"featuresDict": {"features": {
                      "image": {"pythonClassName": "x.Image", "image": {
                          "shape": {"dimensions": ["48", "64", "3"]}, "dtype": "uint8"}},
                      "depth": {"pythonClassName": "x.Tensor", "tensor": {
                          "shape": {"dimensions": ["48", "64"]}, "dtype": "float32"}},
                      "state": {"pythonClassName": "x.Tensor", "tensor": {
                          "shape": {"dimensions": ["5"]}, "dtype": "float32"}},
                  }}}}}}}},
    "episode_metadata": {"featuresDict": {"features": {
        "file_path": {"pythonClassName": "x.Text", "text": {}}}}},
}}}


def _episodes(rng, lengths):
    eps = []
    for e, T in enumerate(lengths):
        imgs = [np.full((96, 128, 3), (10 * t + 40 * e) % 256, np.uint8) for t in range(T)]
        eps.append({
            "steps/is_first": ("int", [1] + [0] * (T - 1)),
            "steps/is_last": ("int", [0] * (T - 1) + [1]),
            "steps/language_instruction": ("bytes", [f"task {e % 2}".encode()] * T),
            "steps/language_embedding": ("float", rng.normal(size=4 * T)),
            "steps/action": ("float", rng.normal(size=3 * T)),
            "steps/observation/image": ("bytes", [_jpeg(i) for i in imgs]),
            "steps/observation/depth": ("float", rng.normal(size=48 * 64 * T)),
            "steps/observation/state": ("float", np.arange(5 * T) + 100 * e),
            "episode_metadata/file_path": ("bytes", [f"/data/ep{e}.h5".encode()]),
        })
    return eps


def _write_shard(path, eps):
    with open(path, "wb") as f:
        for ep in eps:
            rec = _example(ep)
            f.write(struct.pack("<Q", len(rec)) + b"\0" * 4 + rec + b"\0" * 4)


# episode index within its shard, which the synthetic content is keyed by
SOURCE_EP = [0, 1, 0]


def _source(tmp_path, monkeypatch, lengths):
    rng = np.random.default_rng(0)
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    shards = []
    for i, ls in enumerate(lengths):
        name = f"robotics/toy/0.1.0/toy-train.tfrecord-{i:05d}-of-{len(lengths):05d}"
        _write_shard(src / name.rsplit("/", 1)[-1], _episodes(rng, ls))
        shards.append({"name": name, "size": (src / name.rsplit("/", 1)[-1]).stat().st_size})
    monkeypatch.setattr(rlds, "_get", lambda url, *a, **k: json.dumps(
        FEATURES if url.endswith("features.json") else {"description": "toy"}).encode())
    monkeypatch.setattr(rlds, "download", lambda obj, dst: shutil.copy(
        src / obj["name"].rsplit("/", 1)[-1], dst) and dst)
    return shards


def _converter(tmp_path, shards, out="out", **kw):
    return rlds.Converter("toy", tmp_path / out, tmp_path / "tmp", fps=5, fit=(32, 32),
                          encoder="libx264", cq=10, version="robotics/toy/0.1.0/",
                          shards=shards, **kw)


@pytest.fixture
def converted(tmp_path, monkeypatch):
    lengths = [[7, 4], [9]]
    out = _converter(tmp_path, _source(tmp_path, monkeypatch, lengths)).run()
    return out, sum(lengths, [])


def test_tables_and_columns_survive(converted):
    out, lengths = converted
    s = LeRobotV3Source(out)
    assert list(s.lengths) == lengths
    state = s.column("observation.state")
    for e, T in enumerate(lengths):
        ep = state[s.first_row[e]:s.first_row[e] + T]
        np.testing.assert_array_equal(ep.reshape(-1), np.arange(5 * T) + 100 * SOURCE_EP[e])
    info = json.loads((out / "meta/info.json").read_text())
    assert "observation.images.image" in info["features"]
    assert info["features"]["observation.images.image"]["shape"] == [24, 32, 3]
    assert any("depth" in d for d in info["rlds"]["dropped"])
    assert any("embedding" in d for d in info["rlds"]["dropped"])
    assert "observation.depth" not in info["features"]
    assert s.column("action").shape == (sum(lengths), 3)


def test_frames_follow_their_episode(converted):
    out, lengths = converted
    s = LeRobotV3Source(out)
    for e, T in enumerate(lengths):
        frames = s.frames("observation.images.image", e, list(range(T))).float()
        level = frames.mean(dim=(1, 2, 3)).numpy()
        expected = [(10 * t + 40 * SOURCE_EP[e]) % 256 for t in range(T)]
        np.testing.assert_allclose(level, expected, atol=4)


def test_tasks_and_metadata(converted):
    import pyarrow.parquet as pq
    out, _ = converted
    eps = pq.read_table(out / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert [e["tasks"] for e in eps] == [["task 0"], ["task 1"], ["task 0"]]
    assert eps[2]["meta.episode_metadata.file_path"] == "/data/ep0.h5"
    assert [e["rlds.record"] for e in eps] == [0, 1, 0]


def test_a_crashed_run_resumes_from_its_last_checkpoint(tmp_path, monkeypatch):
    """A crash part-way through shard 3 leaves files written after the shard-2 checkpoint;
    the rerun deletes them, continues at shard 3, and matches an uninterrupted run."""
    import pyarrow.parquet as pq

    shards = _source(tmp_path, monkeypatch, [[7, 4], [9], [5, 6]])
    clean = _converter(tmp_path, shards, out="clean", checkpoint_gb=0).run()

    real = rlds.Converter._episode

    def crash_at_episode_4(self, rec, p, videos, data, ep_index, *a):
        if ep_index == 4:
            raise RuntimeError("simulated crash")
        return real(self, rec, p, videos, data, ep_index, *a)

    monkeypatch.setattr(rlds.Converter, "_episode", crash_at_episode_4)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _converter(tmp_path, shards, out="resumed", checkpoint_gb=0).run()
    partial = tmp_path / "resumed/toy"
    assert (partial / ".progress.json").exists()
    monkeypatch.setattr(rlds.Converter, "_episode", real)
    resumed = _converter(tmp_path, shards, out="resumed", checkpoint_gb=0).run()

    assert not (resumed / ".progress.json").exists()
    a, b = LeRobotV3Source(clean), LeRobotV3Source(resumed)
    assert list(a.lengths) == list(b.lengths) == [7, 4, 9, 5, 6]
    np.testing.assert_array_equal(a.column("observation.state"), b.column("observation.state"))
    for e, T in enumerate(a.lengths):
        x = a.frames("observation.images.image", e, list(range(T))).float()
        y = b.frames("observation.images.image", e, list(range(T))).float()
        assert (x - y).abs().mean() < 1.0
    ea = pq.read_table(clean / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    eb = pq.read_table(resumed / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert [e["rlds.shard"] for e in ea] == [e["rlds.shard"] for e in eb]
    # one data file per checkpointed span, nothing left over from the crash
    assert len(list((resumed / "data").rglob("*.parquet"))) == 3


def test_a_finished_dataset_is_skipped(converted, monkeypatch):
    out, _ = converted
    stamp = (out / "meta/info.json").stat().st_mtime_ns
    monkeypatch.setattr(rlds, "download", lambda *a: pytest.fail("downloaded again"))
    rlds.Converter("toy", out.parent, out.parent.parent / "tmp", fps=5).run()
    assert (out / "meta/info.json").stat().st_mtime_ns == stamp
