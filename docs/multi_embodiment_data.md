# Multi-embodiment data layer: motivation and plan

Status: first version implemented 2026-11-11 on branch `multi-embodiment` (eryk-pc worktree
`~/School/thesis/scratch/multiemb/repo`). The sections from "Motivation" on are the original
proposal; where the implementation differs, the list below wins. Results of the first run are
in `docs/.claude/multi_embodiment_wm_report.tex`.

## What was built, and where it departs from the proposal

```
src/thesis/datasets/mixture/reencode.py     NVENC re-encode to 224x224 GOP-2 h264 + episode-table repair
src/thesis/datasets/mixture/source.py       LeRobotV3Source: raw columns and frames by index
src/thesis/datasets/mixture/embodiment.py   embodiment files (configs/mixture/embodiments/*.yaml)
src/thesis/datasets/mixture/clock.py        step times, frame offsets, action chunks
src/thesis/datasets/mixture/manifest.py     per-(source, embodiment) split and statistics
src/thesis/datasets/mixture/stream.py       Member, CameraPolicy, MixtureStream, FixedWindows
src/thesis/datasets/mixture/batch.py        Batch and collate
src/thesis/wm/model.py          encoder, SIGReg, predictor, WorldModel
src/thesis/wm/embodiment.py     per-embodiment action normalization and projection
src/thesis/wm/camera_embedding.py  the five view-identity strategies
src/thesis/wm/probe.py          pixel decoder probe and PSNR
src/thesis/wm/train.py          training loop and validation
```

- **Action windows follow the shown frames.** The chunk for step k is the native actions
  between the frame shown at step k and the one shown at step k + 1, not the nominal
  `[t_k, t_{k+1})`. The two differ whenever `fps / f_wm` is not an integer.
- **K is per embodiment, not shared.** `K_e = round(fps_e / f_wm)`. Each embodiment owns its
  projection, so nothing needs a common K, and with an integer ratio the chunk is the
  measured actions with no resampling. Only uneven gaps are resampled, by `kind`
  (`delta`, `absolute`, `discrete`).
- **The manifest does not depend on the clock.** Statistics are per action/state dimension
  over all native frames. Anchors are derived from episode lengths at load.
- **Padding, not bucketing.** The predictor pads each step to the batch's largest view count
  and masks empty slots. Sequences are at most a few dozen tokens and the encoder is over
  90% of the step, so bucketing has nothing to win yet.
- **Windows must fit inside an episode.** There is no `valid` mask; episodes shorter than a
  window are skipped (26 of 11,991 in the first mixture).
- **Not built:** the conversion registry, a state token (state is in the batch, unused),
  `view_dropout`, per-source `f_wm` overrides, stream fast-forward on resume (a resumed run
  reseeds with `seed + step`).
- **New: episode-table repair.** The lerobot/* Open-X v3.0 conversions record the episode
  that crosses a video-file rollover in the old file, past its end. `reencode.py` re-points
  those episodes and checks that episodes tile every file exactly.

## Motivation

### The research question

RoboJEPA (Zholus et al., 2026, arXiv 2610.10515) trains an action-conditioned latent world model on a 23-dataset,
12-platform robot mixture and shows that imagination error follows a power law in compute. Its design choices make it
expensive:

- The encoder (V-JEPA 2.1-G) is **frozen**, and the predictor models the **full patch grid**: 300 tokens per view per
  frame at 240x320, so a 16-frame two-view clip is 16 x 602 tokens.
- Planning runs CEM with thousands of predictor rollouts over that dense sequence.

The hypothesis is that this is more expensive than it needs to be. A frozen encoder was never trained for next-latent
prediction, so the predictor has to model a lot of content it cannot use. A LeWorldModel-style model (Maes et al., 2026,
arXiv 2603.19312) trains the encoder end to end with SIGReg and represents each frame with a single [CLS]-derived token.
LeWM matches or beats DINO-WM on its benchmarks with about 200x fewer tokens and plans about 48x faster. Its own future
work is "pre-training on large, diverse video datasets". The plan is to scale that recipe:

- Initialize the encoder from LeVJEPA (arXiv 2608.27395), a SIGReg-trained video encoder, and keep it **unfrozen**.
- One token per camera per world-model step.
- One token per step for the actions, projected from the whole action chunk between consecutive world-model frames
  (RoboJEPA already does this with a per-embodiment linear map over the concatenated chunk).
- The predictor sequence shrinks about 100x relative to RoboJEPA, and encoder outputs can be cached during planning.

The comparison against RoboJEPA's published scaling curve needs multi-embodiment training data. The starting target is
the Open-X action subset RoboJEPA uses (2,677 action hours, about 70-210 GB of MP4 at native resolution), with DROID and
RoboCasa added because they are RoboJEPA's evaluation domains and the only high-rate sources where action chunking is
non-trivial.

### Why the current loader is not enough

`src/thesis/mixturesets/lerobot.py` is a good single-dataset adapter (column remaps, name-based slices, fork-safe decoding,
fast parquet stats), but it is built around one dataset:

- **Windows are in frames, not time.** `raw_index` offsets become seconds through that dataset's fps, so one spec covers
  different spans of time on a 5 fps and a 30 fps source.
- **No action chunking across a stride.** Actions between sampled frames cannot be gathered into one per-step chunk.
- **Fixed shapes per field.** Different action widths and camera sets break `default_collate`.
- **One stats table.** `NormalizedSource` pools statistics per field across everything it wraps, which mixes units across
  embodiments.
- **No mixture.** The training loop is `DataLoader(shuffle=True)` over one source, with no weights.
- **The encoding cache and its fingerprint assume one repo.** It is irrelevant for an unfrozen encoder anyway.

Upstream LeRobot does not fill the gap. In v0.7.0, `make_dataset` still raises `NotImplementedError` for
`MultiLeRobotDataset`, that class keeps only the features shared by all datasets and uses the first dataset's fps, and
`aggregate.py` requires an identical fps, `robot_type` and features. RoboJEPA's repo ships converters only for DROID and
AgiBot.

This codebase is scaffolding, so the plan replaces the data layer rather than extending it, with no backward
compatibility.

## Design principles

1. **Time is defined once, in seconds.** A single clock defines what a sample means. Sources never see windows.
2. **Sources are raw.** A source only reads arrays at native rate. All semantics live in declarative embodiment files.
3. **No padding of action or state widths.** Each embodiment keeps its native action and state space, and the model
   owns one input projection (and, later, one output head) per embodiment, as in RoboJEPA.
4. **Fixed K action sub-steps per world-model step.** Each embodiment's action trajectory is resampled in time to exactly
   K samples per step, so the chunk shape is `(K, D_e)` for every source regardless of frame rate.
5. **An infinite, weighted stream.** Training is step-based; epochs have no meaning across a mixture.
6. **Normalization belongs to the model.** Stats are computed offline, stored per embodiment, and applied on device.
7. **Camera count and type do not matter to the data layer.** Each embodiment brings however many cameras it has. Every
   view carries an identity, and the model decides, per experiment, how that identity becomes a learned embedding. The
   default is one learned embedding per (embodiment, camera).

## Components

```
data/source/base.py, lerobot.py   raw episode access at native rate
data/embodiments/*.yaml           what each source's columns mean
data/convert.py                   small registry of vectorized conversions
data/manifest.py                  offline per-(source, embodiment) index + stats
data/clock.py                     world-model time grid, frame selection, action resampling
data/stream.py                    weighted infinite IterableDataset + fixed validation sets
data/batch.py                     batch dataclass and grouped collate
model/embodiment.py               per-embodiment normalizer + projection modules
model/camera_embedding.py         swappable view-identity -> learned embedding strategies
model/bucketing.py                group samples by token layout for per-bucket attention
```

### 1. Source

Interface: `episodes()`, `fps`, `length(ep)`, `read(ep, frame_idxs, columns) -> dict[str, np.ndarray]`.

No windowing, normalization or augmentation. `LeRobotSource` keeps what already works: lazy per-process opening, clearing
the inherited torchcodec decoder cache after fork, pruning unread parquet columns, and raising `tolerance_s` for
re-encoded videos. A RoboJEPA-format (HDF5 plus MP4) source or an OCBench source only needs `read`.

### 2. Embodiment file

Declares, per source, which columns form the action and state vectors, how each group behaves in time, and which camera
fills each role:

```yaml
name: widowx_bridge
action:
  - {from: action, dims: [0, 1, 2], kind: delta}              # eef translation
  - {from: action, dims: [3, 4, 5], kind: delta, convert: euler_xyz}
  - {from: action, dims: [6], kind: absolute, convert: invert_clip01}   # gripper, 1 = closed
state:
  - {from: observation.state, dims: [0, 1, 2, 3, 4, 5, 6]}
cameras:                                    # any number, named freely per embodiment
  - {name: over_shoulder, from: observation.images.image_0}
  - {name: side,          from: observation.images.image_1, role: exterior}
  - {name: wrist,         from: observation.images.image_2, role: wrist}
```

- `D_e` (action width) and `S_e` (state width) are whatever the groups add up to. Nothing is padded to a shared width.
- `kind` controls resampling (section 4). Gripper channels are usually `absolute` even when the arm is `delta`.
- `convert` is optional. With per-embodiment projections, conversions are not needed to share a space. They are kept for
  consistency within one platform family (quaternion to Euler, gripper direction) and for readable planning outputs.
- Multiple embodiment files may point at one source (DROID's cartesian, joint and velocity action variants, as in
  RoboJEPA), each becoming its own embodiment id.
- Cameras are a list of any length. `name` is local to the embodiment. `role` is an optional cross-embodiment tag
  (`exterior`, `wrist`, `head`, ...) that only matters if an experiment chooses role-based camera embeddings
  (section 7). Nothing requires two embodiments to have the same cameras.

### 3. Manifest

A `build-manifest` command, run once per (source, embodiment):

- episode lengths, fps, video hours and action hours;
- the seeded train/val episode split (shuffled, not a tail slice, as `split_episodes` does now);
- the anchor table: valid start times per episode for the run's clock;
- per-dimension normalization statistics (global quantiles over all frames, read straight from parquet), computed on the
  **resampled** action chunks so they match what the model sees;
- a fingerprint over repo id, revision, embodiment file and clock.

Training reads manifests only. That removes the stats pass at startup and the rank-0 stats broadcast, and gives the
mixture the hours it weights by.

### 4. Clock

```yaml
clock:
  f_wm: 3.0        # world-model steps per second
  context: 4       # steps of history
  horizon: 8       # steps predicted
  substeps: 10     # K action samples per step (10 = 30 Hz action resolution at 3 Hz)
```

For an anchor time `t0`, step times are `t_k = t0 + k / f_wm`.

- **Frames.** The nearest native frame to each `t_k`, from each camera the run's camera selection keeps (section 6).
- **Actions.** For step k, the action trajectory over `[t_k, t_{k+1})` is resampled to K evenly spaced sub-steps:
  - `kind: delta`: cumulatively sum to a trajectory, resample it at the sub-step boundaries, and difference again. The
    total motion over the step is preserved exactly, whatever the source rate.
  - `kind: absolute`: linearly interpolate (sample-and-hold for discrete channels such as the gripper).

  The result is `(T, K, D_e)` for every source. A 30 Hz, 15 Hz or 5 Hz source at `f_wm = 3` all give K = 10 sub-steps
  with the same time meaning per index. The per-step action token is a projection of the flattened `K x D_e` chunk.
  Non-integer frame-rate ratios need no special case.
- **State.** Sampled at each `t_k` like frames.
- **Rate rule.** A source is eligible only if `fps >= f_wm`. The manifest builder rejects the rest (for example OXE RoboNet
  at 1 fps) instead of leaving them in the mixture at a low weight.
- **Low-rate sources.** A 5 Hz source resampled to K = 10 contains interpolated sub-steps rather than measured ones.
  `src_fps` is emitted per sample so the model can be conditioned on it if this turns out to matter.
- **Episode boundaries.** Windows that run past either end of an episode are clamped and marked with a per-step `valid`
  flag (RoboJEPA's per-frame mask), not dropped, so short episodes still contribute.

An optional per-source `f_wm` override (RoboJEPA uses per-dataset rates to equalize motion speed) is supported only
together with a per-sample `dt` conditioning input, so mixed time scales are explicit to the model.

### 5. Stream

```yaml
mixture:
  weights: {by: action_hours, temperature: 0.5}    # p_i proportional to hours_i ** temperature
  overrides: {bridge_v2: 2.0}
  sources:
    - {source: lerobot/bridge_v2, embodiment: widowx_bridge}
    - ...
```

- **Temperature weighting.** `temperature: 1` reproduces RoboJEPA's rule. Their Table 8 weights are proportional to video
  hours, with each action-source variant counted separately. Lower values flatten the mixture, so one number shrinks
  the giant subsets (AgiBot, MT-Opt) and lifts small clean ones (FMB, RT-1). `overrides` multiply individual sources
  before renormalizing.
- **Infinite `IterableDataset`.** Each (rank, worker) pair gets its own seeded generator. Each draw picks a source by
  weight, then an anchor uniformly from that source's anchor table. Determinism comes from (seed, rank, worker, draw
  counter), so a resumed run can fast-forward its streams.
- **Validation.** A fixed, seeded, finite list of windows per source from held-out episodes, so validation loss is
  reported per embodiment. A whole source can also be held out for zero-shot transfer to an unseen embodiment.

### 6. Batch and collate

Views are the unit for cameras, embodiments are the unit for actions and state. Neither is padded.

```
views       uint8 (N, T, C, H, W)        every kept view of every sample, flattened; N = sum of views per sample
view_sample int   (N,)                   which sample each view belongs to
view_id     struct of int (N,)           emb_id, camera index within the embodiment, role id (or -1)
valid       bool  (B, T)
emb_id      int   (B,)
src_fps     f32   (B,)
groups      {emb_id: {idx: (b_e,), action: (b_e, T, K, D_e), state: (b_e, T, S_e)}}
```

- **Views.** All views go through the encoder as one flat batch of N clips. The encoder neither knows nor cares how many
  views a sample had. A sample with three cameras contributes three rows, a sample with one contributes one. Frames are
  resized to the run's resolution at decode time, so the views share `(C, H, W)`.
- **Action and state groups.** `idx` lists which batch rows belong to that embodiment. Inside the model each group is
  normalized with its embodiment's stats, projected to the model width by its embodiment's projection, and scattered
  back into a dense `(B, T, D)` action-token tensor.
- **Predictor input.** Per step, a sample contributes `V_b` view tokens plus one action token (plus a state token if
  used), so sequence lengths differ across the batch. The predictor runs on **buckets**: samples are grouped by token
  layout (in practice, by view count), and each bucket is one dense attention call with only the frame-causal mask. The
  MLP, norms and projections still run on all tokens at once; only attention is split. There is no padding, and RoPE
  time positions are per sample, so bucketing changes nothing about a sample's positions.

  Measured on the V100 (2026-11-11, `~/School/thesis/scratch/packbench` on eryk-pc), fp16 forward + backward, 16
  steps, view counts uniform in {1, 2, 3}, compared with a best case where every sample has the same mean length:

  | Predictor          | Batch | Same shape (best case) | Bucketed by V | Padded to max | Packed, SDPA bool mask |
  |--------------------|-------|------------------------|---------------|---------------|------------------------|
  | ViT-S (384 x 12)   | 64    | 15.9 ms                | 30.6 ms       | 19.9 ms       | 52.4 ms                |
  | ViT-S (384 x 12)   | 256   | 46.4 ms                | 57.4 ms       | 59.3 ms       | 546.5 ms               |
  | ViT-B (768 x 12)   | 64    | 36.5 ms                | 47.1 ms       | 45.8 ms       | 104.5 ms               |
  | ViT-B (768 x 12)   | 256   | 118.5 ms               | 133.7 ms      | 155.5 ms      | 1082.4 ms              |

  - Packing into one sequence with a dense boolean mask computes all `N^2` attention scores across the whole batch, so
    it grows with batch size squared: 9-12x slower than the best case at batch 256. Do not use it.
  - FlexAttention, which would skip the masked blocks, does not compile on the V100. Triton fails on sm_70 with
    "Unsupported conversion from f16 to f16". It remains an option on newer GPUs (Sockeye's V100s have the same
    problem).
  - Bucketing costs 13-24% over the best case at batch 256. At batch 64 the per-bucket kernel launches dominate the
    small ViT-S (+93%), so keep micro-batches large or capture the step with CUDA graphs.
  - Padding costs 25-33% here, and that grows as view counts spread out. It is the simple fallback if bucketing becomes
    awkward, at the cost of breaking the no-padding principle.
- **Camera selection per experiment.** The stream applies a run-level `cameras:` policy before decoding, so unselected
  views are never decoded:

  ```yaml
  cameras:
    keep: all                 # all | roles: [exterior, wrist] | names: {droid_cartesian: [left, wrist]}
    max_views: 3              # optional cap; which views are kept is drawn at random per sample when over the cap
    view_dropout: 0.0         # optional: probability of dropping each view (at least one kept), as augmentation
  ```

  The default keeps every camera of every embodiment.

Samples from different embodiments with different camera counts still share every optimizer step.

### 7. Model side: per-embodiment modules

`model/embodiment.py` holds, per embodiment id:

- normalization buffers loaded from the manifest (quantiles over `K x D_e` and `S_e`), so they are saved with the
  checkpoint and nothing has to be broadcast;
- an input projection `Linear(K * D_e -> D)` for actions and `Linear(S_e -> D)` for state;
- later, an output head for action prediction or decoding.

**Camera embeddings** are a swappable strategy chosen per experiment. The data layer always emits the full `view_id`
(embodiment, camera index, role). The strategy decides which table entry each view uses, and adds that learned vector
to the view's token:

```yaml
model:
  camera_embedding: per_embodiment_camera   # default
  # per_embodiment_camera  one vector per (embodiment, camera)  - cameras never share, even across embodiments
  # per_role               one vector per role                   - exterior/wrist/head shared across embodiments
  # per_role_plus_offset   per_role + a zero-initialised per-(embodiment, camera) offset
  # per_index              one vector per camera index           - order-based, ignores identity
  # none                   no camera embedding                   - views are an unordered set
```

Each strategy is a small module mapping `view_id -> table index` plus an `nn.Embedding`, so adding one is a few lines and
needs no change to the data layer. The table sizes come from the manifests, which list every (embodiment, camera) in the
mixture. `per_role_plus_offset` is RoboJEPA's arrangement (a shared view encoding plus a per-camera bias) and is the
natural way to let unseen embodiments start from the shared role vectors.

Planning (CEM) samples in each embodiment's own native action space, normalizes, and projects, which falls out of this
structure directly. Evaluation passes an embodiment name rather than a stats dict.

Augmentation stays on device and applies to `cams` per batch, as the current device-side augmenter does.

## What is dropped from the current layer

- `index` / `raw_index` frame-offset specs, `columns` remaps and `slices`. The clock and embodiment files replace them.
- `NormalizedSource` and computing stats at dataset build time.
- `drop_boundary`, replaced by the `valid` mask.
- The `"last"` goal-frame special case. A goal frame becomes another window time, `t = episode_end`.
- The encoding cache. It only pays off with a frozen encoder. If one comes back, it should be keyed by
  `(source, episode, frame)` at frame level so overlapping windows share entries.

## Risks and open points

- **SIGReg and BatchNorm statistics with mixed batches.** LeWM's projector uses BatchNorm, and SIGReg tests the batch
  of embeddings for an isotropic Gaussian. In a mixed batch the regularizer acts on the mixture, which is the intended
  behaviour. If micro-batches ever become single-embodiment, both would see one embodiment at a time and push each
  embodiment's latents to be Gaussian separately. Keep batches mixed.
- **Interpolated sub-steps on low-rate sources.** Resampling a 5 Hz source to 30 Hz-equivalent sub-steps invents
  structure. Watch per-embodiment validation loss, and try `src_fps` conditioning if low-rate sources hurt.
- **Unseen embodiments and cameras.** With `per_embodiment_camera`, a camera that never appeared in training has no
  trained embedding. Zero-shot transfer experiments should use `per_role`, `per_role_plus_offset` or `none`.
- **Bucketing overhead.** With many distinct view counts in a mixture, buckets get small and launches add up. Cap
  `max_views`, or have the stream draw each micro-batch so its view counts fall into a few buckets.
- **SIGReg over views.** SIGReg sees one embedding per view, so embodiments with many cameras contribute more rows to
  the regularizer than single-camera ones. If that skews the latent distribution, weight rows by `1 / V_b`.
- **Embodiments with few samples.** A projection per embodiment for a small source may be poorly trained. Temperature
  weighting helps; merging near-identical platforms into one embodiment file is the other lever.
- **Decode throughput.** Many sources per worker, each with its own open video handles. Measure samples per second per
  worker early, before scaling the mixture.
- **Action conventions.** The embodiment files are where the real data cleaning happens: about one file per Open-X
  subset. RoboJEPA's Appendix B.4 and its repo's `app/robo_jepa/docs/action_spaces/` are the references.
- **DROID frame rate.** The RoboJEPA repo's `docs/data.md` says stored DROID videos are 15 fps, while the paper's
  Appendix B.2 uses 60 Hz. Read the fps from the data, never from a config.

## Order of work

1. Source interface and `LeRobotSource` read path. One Open-X subset end to end.
2. Clock, with tests: delta resampling preserves total motion, and absolute resampling matches the source at the native
   sample times.
3. Embodiment files and the conversion registry for two or three platforms (Bridge V2, RT-1, FMB).
4. Manifest builder, with stats on resampled chunks.
5. Stream, grouped collate, and the per-embodiment model modules. Train a small LeWM-style model on that subset to
   check throughput and per-embodiment validation loss.
6. Remaining Open-X embodiment files, then DROID and RoboCasa.
