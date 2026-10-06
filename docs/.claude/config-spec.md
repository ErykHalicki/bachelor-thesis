# Config spec

The reference for writing a config under `configs/`. Start at [configs/README.md](../../configs/README.md).

## The modality spec

`algorithm.conditioning` and `algorithm.predict` describe every stream the model consumes and produces. The spec is the single source of truth shared across layers: the algorithm builds its networks from it, and dataset configs interpolate it (`conditioning: ${algorithm.conditioning}`) so the data layer loads exactly the steps the model needs. One entry per stream:

```yaml
conditioning:
  video_context:
    from: observation.images.front  # raw batch field(s) this entry reads (default: the entry name)
    stitch: horizontal              # only with a list `from`: how the views are joined
    role: action                    # marks the entry carrying the robot's actions (see below)
    encoder: video                  # `encoders:` entry that produces trunk-ready tokens (default: passthrough)
    via: tokens                     # tokens = time axis | cross = cross-attention source | cond = per-frame AdaLN conditioning | null = encoded but hidden from the predictor (loss terms and stream-mode seeds only)
    dim: 768                        # feature dim entering the trunk
    index: "-12..-1"                # window of step indices around decision time t=0
    raw_index: "-24..-1"            # raw rows those steps are built from (default: `index`)
    fps: 2.5                        # steps per second; converts indices to seconds
    grid: [16, 16]                  # spatial tokens per step (default [1, 1])
    block_size: 2                   # steps per causal attention block (default 1)
```

`index` accepts a list of ints or inclusive ranges: `"0..47"`, `"-12..-1"`, `"-8..-2, 0"`.

Who reads which keys:

| keys | consumer |
| --- | --- |
| `dim`, `via`, `index`, `fps`, `grid`, `block_size`, `type`, `input` | `GenericViTTrunk` (token layout, RoPE, attention mask, cond slots) |
| `from`, `stitch`, `encoder` | `PredictiveModel` base (routes raw batch fields through encoders) |
| `from`, `raw_index` | dataset backends (which raw columns and time steps to load) |
| `role` | algorithms + eval (which stream is the action chunk) |

## Stitched streams

A list `from` makes one stream out of several fields: they are read with the same window and joined into a single image before the encoder, `stitch: horizontal` (default, side by side) or `vertical`.

```yaml
camera_context:
  from: [observation.images.zed_left, observation.images.wrist_cam]
  stitch: horizontal
  encoder: vjepa
  grid: [14, 28]            # the stitched frame's patch grid, not one view's
```

This is not the same as one stream per camera. The backbone attends across the views inside its own forward, the pair carries one set of token positions rather than two, and the views are joined at pixels, so a temporal encoder sees them as one scene. Views are resized to the first one's frame size before joining, so cameras of different native resolutions compose without the config matching them up.

The stitch happens in the model, not the data layer: datasets and rollout drivers still read each camera as its own field (a robot has no stitched column), and every listed field must be read with the same window — an entry reading one of them elsewhere with a different window is a build-time error.

A `[H, W]` `crop_size` on the encoder keeps a wide stitch at its own aspect; a square crop squashes each view instead, for half the tokens.

## `index` vs `raw_index`

`index` is the model's step axis: one entry per token step, and what `fps`, `block_size`, and the attention mask are stated in. `raw_index` is the rows the dataset loads, and defaults to `index` — for every stream whose step *is* a row, they are the same list and neither side has to know the difference.

They come apart when an encoder builds one step out of several rows. A V-JEPA 2.1 tubelet is 2 frames (`frames_per_step: 2`), so 5 tubelet steps are built from 10 frames, and only `raw_index` can say which 10. That is also where a stream's own sampling rate lives: sampling a 30 fps recording at 5 fps is a `raw_index` strided by 6.

```yaml
index: "-4..0"                                              # 5 tubelets, 2.5 per second
raw_index: "-54, -48, -42, -36, -30, -24, -18, -12, -6, 0"  # 10 frames, 5 per second
```

`len(raw_index)` must equal `len(index) * frames_per_step`, checked at build time. With `frames_per_step: 1` (the default) each raw frame is duplicated into its own tubelet, which needs no `raw_index` — and gives the backbone no motion within a tubelet.

## The attention mask

Two independent rules, both off by default, so a trunk with neither builds no mask at all and hands SDPA `None` rather than an all-True `(N, N)` tensor.

`model.causal: true` makes self-attention block-causal on the shared time axis: a token attends wherever the key's block starts before its own block ends, so streams at different rates align in seconds with no index arithmetic (`algorithms/masking.py`). `block_size` partitions that axis and therefore means nothing without it.

`model.clean_attends_noisy: false` restores the role rule: conditioning tokens are pure K/V context — no head reads their outputs — so they never attend noise-seeded flow tokens, even when their time blocks overlap (a context step at index 0 spans `[0, 1/fps)` and would otherwise see the start of the action chunk). Tokens feeding a `direct` head are prediction sites and are exempt.

**A `direct` stream requires `causal: true`** and refuses to build without it: its input window overlaps its own target window one step later, so unmasked the token at index i attends the clean encoded observation at i + 1 — the value it is trained to predict. No shipped arm uses a `direct` stream; the flow arms leave `causal` off, since DreamZero's action register is bidirectional within a chunk and the conditioning is a fixed window a policy has no reason to order causally.

Cross-attention sources (`via: cross`) are off the token axis entirely and always visible.

## Per-stream experts (`model.experts`)

Off by default: the trunk is one shared DiT stack and every token goes through the same weights. `model.experts` turns it into a true Mixture-of-Transformers (Liang et al. 2024): a mapping from expert name to the token streams it owns, e.g.

```yaml
model:
  experts:
    state: [scene_context, wrist_context, state_context, state_future, scene_future, wrist_future]
    action: [action_clean, action]
```

Each expert then has its own QKV and output projections, QK-norms, MLP, cross-attention branch and adaRMS modulation in every block (`MoTBlock`), while a single self-attention over the union of tokens — under the mask above — is the only place the experts meet. An expert's weights receive gradient from another expert's loss only through the keys/values its tokens contribute where `attends:` lets the other expert read them.

Rules, all checked at build: every `via: tokens` and predicted token stream is named in exactly one expert (a stream left out or listed twice fails; the `all` / `context` / `noisy` groups of `attends:` are accepted); at least two experts (one is the shared trunk — spell that by omitting the key); `via: cross` / `cond` / `null` entries are not tokens and cannot be listed. Parameter names change (`blocks.<i>.qkv.<e>.…` instead of `blocks.<i>.sa.qkv.…`), so a shared-trunk checkpoint does not load into an expert arm or vice versa. No shipped arm declares experts; `tests/test_mot_experts.py` shows the shape.

## `role: action`

Stream names are arbitrary labels with one exception: a few things have to know which
stream holds the robot's actions — how long a predicted chunk is, which prediction a
rollout executes and which statistics unnormalize it, and which conditioning stream a
world-model rollout (`rollout_latent`) feeds candidate actions through. `role: action` marks that entry, in `conditioning:`,
`predict:`, or both.

An entry reading the `action` field claims the role by convention when nothing declares
it, so configs written before the role keep working. Declaring it twice in one spec is a
build-time error, and a model whose `predict:` has no action stream is a world model:
`ChunkDriver` refuses it by name. `PlanDriver` (`eval.goal_image`) is the on-robot path for
one, but it needs a model exposing `plan()`, and none ships in this repo.

An action entry in `conditioning:` is a past-action history, and its offsets must all be
negative — a rollout only knows actions it has already executed. It is the one conditioning
stream a robot cannot supply as an observation column, so the driver replays what it
commanded itself, padding with normalized zero until an episode is old enough to fill the
window (`experiments/eval/chunking.py`).

The shipped flow arms declare that entry `via: null`, so no action history reaches the
trunk — DreamZero conditions on observations alone, and the only action information an a2a
arm gets is its own flow seed. It cannot simply be deleted: the driver reads it to size the
executed-action buffer that seed is built from, and to know `action` is a column no robot
provides. `dims_*.yaml` cuts the state to the current step for the same reason, matching
their one state token per chunk.

## Encoders

`algorithm.encoders` declares named encoder modules; a spec entry's `encoder:` key names the
one that turns its raw field into trunk-ready tokens. Types: `vit` (from scratch, trained end
to end), `vjepa2` (pretrained V-JEPA 2.1 video backbone, any published size via `arch`),
`resnet` (torchvision ResNet, optional ImageNet weights), `linear` / `mlp` (projection
heads), `chunk_mlp` (chunk-level codec, `in_steps` rows in, `out_steps` out), `reshape`
(the same regrouping with no network: an exact, parameter-free bijection between rows and
features), `attentive_pool` (learned-query pooling, the CLS a backbone never had),
`pixel_decoder` (latents back to an image — see Pixel probes below), `smolvlm` (stub). Two
keys every entry understands make them composable:

- `frozen: true|false` — no grads, permanent eval mode. Pretrained types default frozen,
  from-scratch types and heads default trainable, and either can be flipped per run.
- `input: <encoder>` — chaining: this encoder consumes another entry's output instead of the
  stream's raw field, which feeds the chain's root. Cycles and unknown names are build-time
  errors, so a chain always bottoms out at real data.

```yaml
encoders:
  vjepa: { type: vjepa2, arch: vit_large, checkpoint: /path/to/vjepa21.pt, crop_size: 96 }
  head:  { type: linear, input: vjepa, in_dim: 1024, out_dim: 256 }
conditioning:
  pixel_context: { from: observation.images.pixels, encoder: head, dim: 256, ... }
```

A loss naming a stream sees the chain's final output, so "frozen vjepa2.1 -> trainable linear
-> sigreg on the projection" is pure config: put `sigreg` on `pixel_context` above. Frozen
stages run under no_grad whenever nothing upstream of them is trainable.

## How a predicted stream is produced (`type`)

Each `predict:` entry declares `type`, and that is the only paradigm-specific thing in the
model — a DiT is this ViT with noise-seeded predicted tokens, LeWorldModel is the same ViT
whose predicted tokens are the previous latents, and the two can be mixed in one `predict:`.

| `type` | seeded input | target | AdaLN cond | tokens |
| --- | --- | --- | --- | --- |
| `flow` (default) | noise `x_t` | `x1 - x0` (a velocity) | timestep + `cond` streams | its own positions |
| `direct` | reads its `input` stream's tokens | its own encoded `index` window | `cond` streams only | none |

A `direct` stream adds no tokens: it names an `input` token stream, reads a head off that
stream's token slice, and its `index` is the *target* window, paired positionally with the
input window (equal lengths). So `input: "-2..0"` with `index: "-1..1"` means frame -2 predicts
-1 ... and frame 0 predicts +1 — autoregressive teacher forcing, with the token count identical
at train and eval (eval reads the last position).

AdaLN conditioning is a concatenation of fixed slots (timestep first when any stream is `flow`,
then one per `via: cond` stream), each zeroed where it does not apply. The timestep slot is
zeroed on a direct stream's input tokens, so `t` never reaches the direct path.

### Projections into and out of the trunk (`token_in` / `token_out`)

A predicted stream may replace its default projections. Declaring neither keeps `mlp` at
`mlp_dim: model_dim`, so every predicted stream — flow or direct, action, state, or a vision
latent token — leaves the trunk the same way unless a config opts a stream out explicitly.

```yaml
predict:
  action:
    token_in:  {type: mlp, mlp_dim: 512}   # DreamZero's action encoder
    token_out: {type: mlp, mlp_dim: 64}    # DreamZero's action decoder
```

| `type` | in | out |
| --- | --- | --- |
| `mlp` (default) | linear lift, the **timestep embedding concatenated** onto it, then an MLP of inner width `mlp_dim` back to the trunk width | an MLP through an `mlp_dim` bottleneck: no norm, no modulation, output projection zero-init |
| `adaln` | `InputLayer`: linear lift, RMSNorm modulated by the full cond | `FinalLayer`: modulate then project, zero-init so the velocity is exactly 0 at init |

`mlp` is DreamZero's pair (itself pi0's encoder), and the difference that matters is where `t`
enters: `adaln` gives it exactly one path in, `mlp` gives it two — concatenated at the input
*and* the adaLN cond every block still applies (for a `direct` stream, whose timestep slot is
always zeroed, `mlp`'s only path for `t` is moot and it reduces to a plain bottlenecked MLP).
Declaring `type: mlp` explicitly still requires `mlp_dim`; only the *absence* of a `token_in`/
`token_out` block falls back to `mlp_dim: model_dim`. `mlp_dim` is rejected on `adaln`; on
`token_out` it also bounds the **rank** of the prediction, so it must be read against that
stream's `dim` rather than copied from DreamZero (64 expands into a per-step action row and
constrains nothing; against a whole chunk in one token it is a hard bottleneck). `token_in` is
meaningless on a `direct` stream (it contributes no tokens of its own) and on a conditioning
stream (its own MLP, never read by a head — see below); both raise.

The shipped flow arms declare `mlp` explicitly with their own `mlp_dim`s (unaffected by the
default). `mlp`'s output projection is zero-initialized here (unlike DreamZero's), so swapping
between the two varies the I/O structure and not the starting point: either decoder emits
exactly zero velocity at init, which is nearly the right answer under an A2A seed. On a
one-token action stream set `token_out`'s `mlp_dim` to at least the stream's `dim` —
DreamZero's 64 expands into a per-step action row and constrains nothing, but would confine a
whole chunk to a 64-dim subspace.

Clean/conditioning streams (`via: tokens`) are not configurable this way — every one enters
through the same `make_mlp(dim, model_dim, model_dim)` lift, unconditionally, with no timestep
concatenated (they are never denoised, so `t` has nothing to tell them). `via: cond` streams
(the pooled AdaLN sources) are unaffected and still enter through a plain linear.

The dummy algorithm keeps a minimal list form (`conditioning: [observation]`) where entries are the batch field names themselves; only the build-time modality check reads that form.

## A2A flow sources (`source:`)

A flow stream may declare a `source:` block replacing its Gaussian x0 with an informed
history seed (A2A flow matching, arXiv:2602.07322): training interpolates `x_t` between the
seed and the encoded target, rollout Euler-integrates from the seed. `v = 0` then already
yields persistence / repeat-the-last-chunk, so the field only has to learn the delta, and
the source shares the target's scale instead of transporting unit noise onto it.

```yaml
predict:
  scene_future:                      # stream mode: reuse an encoded conditioning stream
    type: flow
    source:
      stream: scene_context          # must be a conditioning stream (history, no leak)
      rows: "-1"                     # its index STEPS to take (negatives from the end)
      fill: repeat_last              # widen to the stream's token count (or `tile`)
      noise_std: 0.1                 # gaussian noise on the seed, in flow space
  action:                            # field mode: read a raw window of a batch field
    type: flow
    source:
      from: action
      index: "-36..-1"               # raw rows; dataset backends load them like any window
      encoder: null                  # optional chain into the stream's flow space
      raw_noise_std: 0.0             # noise BEFORE the encoder (paper's history corruption)
      noise_std: 0.1
      # tile: 36                     # repeat the raw window before encoding (e.g. index "-1")
```

The declared `noise_std` / `raw_noise_std` apply at eval and rollout exactly as in
training — the endpoint every metric scores integrates from the same x0 distribution the
flow loss trained against.

Field-mode rows must be history a rollout can know (`<= 0`, `<= -1` for the action field);
they widen the union window the dataset loads and the history eval drivers buffer. A seed is
a live part of the graph: when it runs through a jointly trained encoder, the flow loss
shapes that encoder through x0 as well as x1. Nothing on this path is stop-gradded — a
codec latent is kept from collapsing by `ae_recon` and `sigreg`, which ground it, rather
than by cutting gradients.

Nothing currently shipped seeds from anything but noise (`flow_wam_b601` — dz_noise, the
default): x0 is Gaussian, matching DreamZero. World-model rollouts
(`rollout_final_state`/`rollout_latent`) do not thread seeds and integrate from noise regardless.

Similarly, nothing shipped currently uses `adaln` (`flow_common` sets `mlp` explicitly on
`token_in`/`token_out`, DreamZero's own projections, which is also the default now) — see the
table above for what declaring `token_in`/`token_out: {type: adaln}` on a predicted stream
would restore.

### Comparing across datasets (`units: raw`)

An `integration` term takes `units: normalized` (default) or `units: raw`. Normalized is MSE
in the space the model flows in; raw rescales the per-dim error by the normalizer's own slope
(`span / 2` for `percentile`, `std` for `mean_std`) so the number is in the field's real units
— degrees, for a joint-position action column.

This matters because **a normalized metric is only comparable within one set of normalization
stats.** Recomputing stats over more episodes rescales every one of them: the b601 pusht repull
from 141 to 161 episodes widened the action percentile range by 1.2–2.6× per dimension, which
divides a normalized MSE by ~2.7× on average (6.5× on wrist_roll) for a policy of *identical*
physical accuracy. A sweep that changes dataset and then compares `loss/int_action` against the
old numbers is reading a unit change as an improvement.

`units: raw` needs the model to carry `norm_stats` and only applies to a stream reading a
normalized field; on a codec-latent stream there is no physical unit and the term raises. The
label gets a `_raw` suffix (`loss/int_action_raw`), so both can be reported side by side —
which is what `flow_common.yaml` and `flow_wam_common.yaml` now do, both at `weight: 0.0`.

## The batch contract between datasets and algorithms

These are the assumptions both sides rely on. Breaking any of them breaks training or, worse, silently trains on leaked future data.

- **Fields are keyed by `from`.** A batch is a flat dict whose keys are the spec entries' `from` values (entry name if `from` is omitted). Entry names themselves are arbitrary labels and never appear in a batch.
- **Shared fields carry the union window.** When several entries read one field (`video_context` takes tubelets -12..-1 of the front camera, `future_video` takes 0..3), the dataset emits a single tensor covering the sorted union of their windows, ascending in time. Each reader slices its own window back out; the algorithm precomputes those selectors at build time.
- **Slicing happens before encoding.** Each entry's raw window goes through its encoder separately. Encoding the whole union once and splitting the latents would let a temporal encoder (VJEPA2) blend future frames into the clean context latents, a train-time leak that inference cannot reproduce.
- **Row scale per index step.** One index step spans `raw_steps_per_index` rows of an encoded field (VJEPA2: 2 frames per tubelet) — those rows are the entry's `raw_index`, which is what a backend loads — and grid-area rows of a passthrough token field, which is a spatial fan-out and needs no `raw_index`. Entries without `index` (language strings) pass their whole field through unsliced.
- **Conditioning lives strictly in negative time, predicted streams at >= 0.** This is what the block-causal mask anchors on, and it gives the union fields their inference property: conditioning rows are always a prefix of the union, so an obs dict at rollout time carries only the past rows and the same selectors still apply. Cross sources are off the time axis and exempt (the current state can sit at index 0).
- **Datasets deliver float32.** Float64 columns are cast on load; model code never casts dtype or device.

## Augmentation

`dataset.augment` adds training-time augmentation. It is a pure tensor transform over the batch dict — it knows nothing about backends or modalities, so one config form works for every source:

```yaml
augment:
  seed: null                      # null: global RNG; an int makes the run reproducible
  streams:
    "observation.images.*":       # field name, or an fnmatch glob over field names
      random_crop:    {scale: [0.9, 1.0]}
      rotation:       {degrees: 3.0, p: 0.5}
      color_jitter:   {brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.02}
      gaussian_noise: {std: 0.01}
    observation.state: {gaussian_noise: {std: 0.01}}
    action:            {gaussian_noise: {std: 0.005}}
```

| op | params | notes |
| --- | --- | --- |
| `random_crop` | `scale` (side fraction, or `[low, high]`) | resizes back, so field shape is unchanged |
| `rotation` | `degrees`, `padding_mode` | about the image center, aspect-corrected |
| `color_jitter` | `brightness`, `contrast`, `saturation`, `hue` | magnitudes: factor drawn from 1 +/- m; `hue` is in turns |
| `gaussian_noise` | `std` | images and vector streams alike |

Every op also takes `p` (default 1.0), the probability it fires. Ops run in the order of the table above regardless of how the config lists them, so noise is never smoothed by a later interpolation.

Rules worth knowing:

- **Parameters are drawn once per field per item** and shared across the field's whole time window, so a video context is augmented coherently. Different fields draw independently: two cameras get different crops.
- **Where it runs is `experiment.augment_on`.** `device` (the default in `experiment/base.yaml`) applies the Augmenter per sample to each micro-batch once it is on the accelerator, with the batch-size probe including it; `workers` wraps the source so DataLoader workers augment items on the CPU. Same ops, same per-item draws; measured on pnpt 256² windows the image ops cost ~200 ms per sample on one CPU thread (color_jitter ~90 ms) with multi-second tails under contention, and ~3 ms on an H100. The train log's `aug` time is launch time only on CUDA. Fields served as encoding-cache keys are never augmented here; their draws are baked into the cache.
- **Augmentation runs outside normalization.** Image streams are not normalized, so they arrive raw; state/action noise is therefore specified in normalized units and needs no retuning per dataset.
- **Image ops need a `(..., C, H, W)` field** and work on floats in [0, 1], casting back to the field's dtype (uint8 streams are scaled by 255 both ways) — so a magnitude means the same thing whichever dtype a backend serves. Putting an image op on a vector field is a build-time error, as is a `streams` pattern that matches no field.
- **Training only.** Held-out eval and the modality check build the source with `augment=False`.

## Losses

The algorithm spec's `losses:` is a list of terms, each `{type, weight, ...}`, summed to the
training loss (default when omitted: a single prediction term over every predicted stream).
`flow` and `prediction` name the same MSE(`pred`, `target`) objective for the two stream types —
`type: flow` (velocity vs `x1 - x0`) and `type: direct` (next latent vs next latent) — over
their target streams (`stream`/`streams`, else all predicted streams marked
`type: flow`/`prediction`); differing per-stream weights are just multiple terms. `sigreg`
regularizes the encoder latents of its `streams` toward an isotropic Gaussian (collapse
prevention for jointly-trained encoders). `ae_recon` (`{type: ae_recon, stream, decoder,
weight, norm: l1|mse}`) decodes one predict stream's encoder latent back through a named
`encoders:` entry and penalizes the error against the stream's raw window — the
reconstruction grounding of a codec-latent flow stream (A2A's L_AE).

`integration` (`{type: integration, stream, weight, num_steps, decoder, norm: l1|mse}`) is
the inference-consistency term: it Euler-solves the flow stream *with gradients* from the
same x0 eval would use (its `source:` seed, else noise) and scores the ENDPOINT, where every
other term scores a velocity at a random `t`. Name a `decoder:` to score in raw units (the
decoded chunk vs `ctx["raw"]`), omit it to score in flow space against the encoded target.
`num_steps` defaults to the algorithm's `num_flow_steps`, which is the point — supervise the
solve that actually runs. **It costs `num_steps` extra predictor passes per training step,
and the backward runs the whole chain**, so activation memory grows with `num_steps` too.
Every `integration` term shares one solve, so their `num_steps` must agree. Nothing shipped
scores this at nonzero weight.

Losses read a per-call context (encoder `latent`, predictor `pred`, regression `target`, plus
`raw` windows and the `integrated` endpoint for the terms that ask for them), so a term can
only touch its intended artifact — `sigreg` on a stream always means that stream's encoder
latent, never a prediction. See `algorithms/losses.py`.

## Pixel probes

Two algorithms exist only to answer what a latent is holding, by decoding it back to an
image with a `pixel_decoder` (learnable per-patch queries cross-attending the latent; see
`algorithms/pixel_decoder.py`). Neither ever feeds a policy — they are diagnostics, and both
run under the `decoder` experiment (`experiments/decoder.py`), which adds one key:
`encoder_init: <run id | .pt>`, the finished run being probed. Their eval is the
`reconstruction` backend (`experiments/eval/reconstruction.py`): the held-out loss, a PSNR
per stream, and a panel image per decoded stream logged to wandb. No probe configs ship in
`configs/`; a probe needs `experiment/decoder.yaml`, an `eval/` config with
`backend: reconstruction`, and a `latent_decoder` or `wam_decoder` algorithm config.

| algorithm | decodes | panel columns |
| --- | --- | --- |
| `latent_decoder` | an encoder's output | `frame \| decoded latent` |
| `wam_decoder` | a frozen world model's encoded **and predicted** latents | `frame \| decoded latent \| decoded prediction` |

Each panel row is one held-out sample, drawn from a fixed seeded subset so successive
evals show the same frames. Reading the columns pairwise is the point: frame against
decoded latent is what the encoder threw away, decoded latent against decoded prediction is
what the predictor got wrong.

`encoder_init` brings across both halves of the model being probed, so a probe config never
restates one: the **weights** (every parameter the two share, by name — encoders and trunk
— plus the normalization stats they trained under) and, for a run id, the **stored
algorithm config**, which supersedes what the probe composed. That is the same guarantee
post-hoc eval gives: the probe measures what actually trained, not what the yaml says
today. A local `.pt` carries no config, so there the composed one stands, as it does when
wandb cannot be reached (with a warning).

`latent_decoder` reuses the two spec sections directly — `conditioning:` is the latent,
`predict:` is the frame it came from, and each predict entry names both (`input:` a
conditioning stream, `decoder:` an `encoders:` entry). That is the arm for a stack no run
declares, such as a bare pretrained V-JEPA grid.

`wam_decoder` declares nothing but the probe: it composes the WAM's algorithm config
(`defaults: [flow_wam_b601, _self_]`) and adds its decoders to that `encoders:` map, a `decode:`
section, and its own losses.

```yaml
decode:
  scene: {stream: scene_future, decoder: decode_scene}   # a predict stream: all 3 columns
losses:
  - {type: prediction, stream: scene, weight: 1.0}       # only decoders may be named
```

Naming a *conditioning* stream instead renders the two columns an encoder probe can show,
so the encoder-only question needs no separate run — it is already the middle column.

Only the decoders train. `freeze_world_model` (default true) freezes everything else and
pins the trunk and encoders in eval mode, overriding whatever `frozen:` the probed run's
config declared: those encoders were legitimately training there, and here a decoder's
gradient would move the representation being measured. Decoders train on the *encoded*
latent, never on a prediction, or the third column would flatter the model. A decoder's
`in_dim` and `tokens_per_step` must match its stream's `dim` and grid area; both are
checked at build time and in `tests/test_configs.py`.

## Experiment-owned knobs

Training tuning lives in `experiment/`, not in the algorithm spec. `execute_len` in the
algorithm yaml sets how much of a predicted action chunk eval executes per replan.

`execute_len`, `num_flow_steps` and `cfg_scale` are also eval knobs (`eval/b601.yaml`),
where `null` keeps the algorithm yaml's value. A post-hoc eval replaces the whole algorithm
section with the config its run stored, so editing the algorithm yaml after training does
nothing — the eval override is what changes a finished run, and it is the only one that
reaches an inference server, which never sees the client's algorithm config at all.

Every key of a real-robot rollout eval: [lerobot-eval-knobs.md](lerobot-eval-knobs.md).
