"""Raw episode access to a LeRobot v3.0 dataset at its native rate.

A source knows nothing about windows, normalization or embodiments. It exposes the
episode table, whole low-dimensional columns as arrays, and frames by index.
"""

import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _read_parquets(directory, columns=None):
    files = sorted(Path(directory).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {directory}")
    tables = [pq.read_table(f, columns=columns) for f in files]
    return pa.concat_tables(tables)


def _to_array(column):
    column = column.combine_chunks()
    t = column.type
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return column.flatten().to_numpy(zero_copy_only=False).reshape(len(column), -1)
    return column.to_numpy(zero_copy_only=False).reshape(len(column), -1)


class LeRobotV3Source:
    """One dataset directory. Low-dim columns are loaded once and kept as arrays, which
    forked DataLoader workers share. Video decoders are opened lazily per process."""

    MAX_OPEN_DECODERS = 16

    def __init__(self, root):
        self.root = Path(root).expanduser()
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        if self.info["codebase_version"] != "v3.0":
            raise ValueError(f"{self.root} is {self.info['codebase_version']}, expected v3.0")
        self.fps = float(self.info["fps"])
        self.video_keys = [k for k, f in self.info["features"].items() if f["dtype"] == "video"]

        episodes = _read_parquets(self.root / "meta" / "episodes").sort_by("episode_index")
        names = [c for c in episodes.column_names if not c.startswith("stats/")]
        self._episodes = {c: episodes[c].to_numpy(zero_copy_only=False) for c in names
                          if c != "tasks"}
        self.first_row = self._episodes["dataset_from_index"].astype(np.int64)
        self.lengths = self._episodes["length"].astype(np.int64)
        self._columns = {}
        self._decoders = {}
        self._pid = None

    @property
    def num_episodes(self):
        return len(self.lengths)

    @property
    def total_frames(self):
        return int(self.first_row[-1] + self.lengths[-1])

    def column(self, name):
        """A whole low-dim column as (total_frames, width) float32, in dataset row order."""
        if name not in self._columns:
            table = _read_parquets(self.root / "data", columns=[name, "index"]).sort_by("index")
            array = _to_array(table[name]).astype(np.float32)
            if len(array) != self.total_frames:
                raise ValueError(f"{self.root}: column '{name}' has {len(array)} rows, "
                                 f"episodes cover {self.total_frames}")
            self._columns[name] = array
        return self._columns[name]

    def _decoder(self, video_key, episode):
        if self._pid != os.getpid():
            self._decoders, self._pid = {}, os.getpid()
        chunk = int(self._episodes[f"videos/{video_key}/chunk_index"][episode])
        file = int(self._episodes[f"videos/{video_key}/file_index"][episode])
        key = (video_key, chunk, file)
        decoder = self._decoders.pop(key, None)
        if decoder is None:
            from torchcodec.decoders import VideoDecoder

            path = self.root / self.info["video_path"].format(
                video_key=video_key, chunk_index=chunk, file_index=file)
            decoder = VideoDecoder(str(path), seek_mode="approximate", num_ffmpeg_threads=1)
            if len(self._decoders) >= self.MAX_OPEN_DECODERS:
                self._decoders.pop(next(iter(self._decoders)))
        self._decoders[key] = decoder
        return decoder

    def frames(self, video_key, episode, frame_idxs):
        """uint8 (len(frame_idxs), 3, H, W) for episode-relative frame indices."""
        offset = self._episodes[f"videos/{video_key}/from_timestamp"][episode] * self.fps
        idxs = (int(round(offset)) + np.asarray(frame_idxs)).tolist()
        return self._decoder(video_key, episode).get_frames_at(indices=idxs).data
