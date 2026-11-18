"""Shared base for the predictor algorithms (dit_predictor, vit_predictor).

Builds the encoders, reads the modality spec (`conditioning`/`predict`), slices each entry's
window out of its raw batch field, routes conditioning streams by `via`, and runs the
`losses:` terms. A subclass supplies the predictor, `_forward_train` and the eval rollouts.
The batch contract is documented in docs/.claude/config-spec.md.
"""

import fnmatch
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.spec import (
    LAST, action_entry, flow_source_entries, parse_index, raw_index, spec_fields, split_index,
)
from .base import BaseAlgorithm
from .encoders import build_encoder, resolve_chains
from .losses import build_loss_term


def stitch_views(views, mode="horizontal"):
    """Several (B, T, C, H, W) camera windows -> one wider (or taller) window.

    Views are resized to the first one's frame size before joining. Flat (B, T, D) streams
    join on the feature axis instead: two 7-dim state columns become one 14-dim vector per
    step, and only `horizontal` means anything there.
    """
    if mode not in ("horizontal", "vertical"):
        raise ValueError(f"unknown stitch '{mode}'; use 'horizontal' or 'vertical'")
    if views[0].ndim == 3:
        steps = {tuple(view.shape[:2]) for view in views}
        if len(steps) > 1:
            raise ValueError(
                f"flat streams stitch on the feature axis, so they must share a window: "
                f"got {sorted(steps)}"
            )
        if mode != "horizontal":
            raise ValueError(
                f"stitch '{mode}' needs a spatial axis; a flat stream only has features, "
                f"so it joins horizontally or not at all"
            )
        return torch.cat(views, dim=-1)
    dim = -1 if mode == "horizontal" else -2
    size = views[0].shape[-2:]
    out = []
    for view in views:
        if view.shape[-2:] != size:
            resized = F.interpolate(
                view.flatten(0, 1).float(), size=size, mode="bilinear", align_corners=False
            )
            if view.dtype == torch.uint8:
                resized = resized.round().clamp(0, 255)
            view = resized.to(view.dtype).unflatten(0, view.shape[:2])
        out.append(view)
    return torch.cat(out, dim=dim)


class PredictiveModel(BaseAlgorithm):
    def __init__(self, cfg):
        super().__init__()
        self.conditioning = cfg.conditioning
        self.predict_spec = cfg.predict

        self.encoder_specs = {
            name: dict(spec) for name, spec in (cfg.get("encoders") or {}).items()
        }
        self.encoders = nn.ModuleDict({
            name: build_encoder(spec) for name, spec in (cfg.get("encoders") or {}).items()
        })
        self._encoder_chains = resolve_chains(cfg.get("encoders") or {})

        # a field-mode source is registered as a pseudo entry so the dataset loads its rows
        self._flow_sources = {}
        for name, spec in cfg.predict.items():
            src = spec.get("source")
            if not src:
                continue
            if spec.get("type", "flow") != "flow":
                raise ValueError(f"stream '{name}' declares a `source:` but is not flow")
            if src.get("stream"):
                if src["stream"] not in cfg.conditioning:
                    raise ValueError(
                        f"stream '{name}' seeds from '{src['stream']}', which is not a "
                        f"conditioning stream (a seed is history; targets would leak)"
                    )
                enc = src.get("encoder")
                if enc is not None and enc not in self.encoders:
                    raise ValueError(f"source of '{name}' names unknown encoder '{enc}'")
            self._flow_sources[name] = src
        self._source_entries = flow_source_entries(cfg.predict)
        self.source_fields = {
            f for spec in self._source_entries.values() for f in spec_fields("", spec)
        }
        for pseudo, sspec in self._source_entries.items():
            enc = self._flow_sources[pseudo.removesuffix(".source")].get("encoder")
            if enc is not None and enc not in self.encoders:
                raise ValueError(f"source '{pseudo}' names unknown encoder '{enc}'")

        # entries sharing a batch field are served one tensor covering the UNION of their
        # windows, past first; each slices its own window back out before encoding
        entries = {**cfg.conditioning, **cfg.predict, **self._source_entries}
        field_union, field_has_last = {}, {}
        for name, spec in entries.items():
            enc = spec.get("encoder")
            assert enc is None or enc in self.encoders, f"'{name}' names unknown encoder '{enc}'"
            if "index" in spec:
                # raw rows, not index steps: a tubelet step spans several rows
                rel, has_last = split_index(raw_index(spec))
                for field in spec_fields(name, spec):
                    field_union.setdefault(field, set()).update(rel)
                    field_has_last[field] = field_has_last.get(field, False) or has_last
        # "last" is appended after the relative offsets, so it is always the final row
        field_union = {
            field: sorted(steps) + ([LAST] if field_has_last.get(field) else [])
            for field, steps in field_union.items()
        }
        self._last_fields = {f for f, hl in field_has_last.items() if hl}

        # An index step spans multiple rows in TIME (`raw_index`, e.g. 2 per VJEPA2 tubelet)
        # and in SPACE (grid area), and the two compose. "last" rows are addressed from the
        # END so the same selectors work at rollout, where the field lacks the future rows.
        self._selectors = {}
        self._raw_per_step = {}
        for name, spec in entries.items():
            if "index" not in spec:
                self._selectors[name] = None
                continue
            enc = spec.get("encoder")
            if enc is not None:
                raw_per_step = 1
                for stage in self._encoder_chains[enc]:
                    raw_per_step *= self.encoders[stage].raw_steps_per_index
                span = 1
            else:
                raw_per_step = 1
                grid = spec.get("grid", (1, 1))
                span = grid[0] * grid[1]
                if int(spec.get("views", 1)) > 1:
                    raise ValueError(
                        f"stream '{name}' has views but no encoder: a multi-view field is "
                        f"(B, T, V, ...) and only an encoder turns it into tokens"
                    )
            self._raw_per_step[name] = raw_per_step
            steps = len(parse_index(spec["index"]))
            raw = parse_index(raw_index(spec))
            if len(raw) != steps * raw_per_step:
                needed = steps * raw_per_step
                raise ValueError(
                    f"stream '{name}' consumes {raw_per_step} raw rows per index step, so "
                    f"its {steps} index steps need a `raw_index` naming the {needed} rows "
                    f"they are built from; got {len(raw)}"
                )
            fields = spec_fields(name, spec)
            unions = {f: field_union[f] for f in fields}
            if len({tuple(u) for u in unions.values()}) > 1:
                raise ValueError(
                    f"stream '{name}' stitches {fields}, but they carry different row "
                    f"layouts ({unions}); every stitched view must be read with one window"
                )
            union = field_union[fields[0]]
            n_last = raw.count(LAST)
            seen_last = 0
            entry_rows = []
            for i in raw:
                if i == LAST:
                    seen_last += 1
                    entry_rows.append(seen_last - n_last - 1)
                else:
                    entry_rows.append(union.index(i))
            rows = [e * span + j for e in entry_rows for j in range(span)]
            contiguous = rows == list(range(rows[0], rows[0] + len(rows)))
            stop = rows[0] + len(rows)
            self._selectors[name] = slice(rows[0], stop or None) if contiguous else rows

        # eval rollout geometry: obs_len is the SPAN back to the earliest conditioning
        # offset, and a window may be sparse, so drivers sample `field_offsets` out of a
        # dense obs_len history. Source seeds are history too, and must be buffered alike.
        cond_union = {}
        for name, spec in {**dict(cfg.conditioning), **self._source_entries}.items():
            if "index" in spec:
                rel, _ = split_index(raw_index(spec))          # "last" is not history
                for field in spec_fields(name, spec):
                    cond_union.setdefault(field, set()).update(rel)
        self.field_offsets = {f: sorted(steps) for f, steps in cond_union.items() if steps}
        self.obs_len = max(
            (1 - min(steps) for steps in self.field_offsets.values()), default=1
        )
        # both stay None for a model that predicts no actions, e.g. a world model
        action = action_entry(cfg.predict)
        self.action_stream, self.action_field = action or (None, None)
        if self.action_stream is not None:
            spec = cfg.predict[self.action_stream]
            # a codec-latent stream indexes latent steps, not actions: `chunk_len` and the
            # decoder's `out_dim` restore the executed length and raw action width
            self.chunk_len = int(spec.get("chunk_len") or len(parse_index(spec["index"])))
            dec = spec.get("decoder")
            dec_spec = self.encoder_specs.get(dec) or {} if dec else {}
            self.action_dim = int(dec_spec.get("out_dim") or spec["dim"])
            self.execute_len = int(cfg.get("execute_len", self.chunk_len))
            assert 1 <= self.execute_len <= self.chunk_len

        for pseudo, sspec in self._source_entries.items():
            floor = 0 if spec_fields(pseudo, sspec)[0] != self.action_field else -1
            bad = [i for i in parse_index(sspec["index"]) if i > floor]
            if bad:
                raise ValueError(
                    f"source of '{pseudo.removesuffix('.source')}' reads offsets {bad}, "
                    f"which a rollout does not know yet (max allowed: {floor})"
                )

        default_streams = [
            name for name, spec in cfg.predict.items()
            if spec.get("type", "flow") in ("flow", "prediction")
        ]
        loss_cfg = cfg.get("losses") or [
            {"type": "flow", "stream": name, "weight": 1.0} for name in default_streams
        ]
        stream_dims = {
            name: int(spec["dim"])
            for name, spec in {**dict(cfg.conditioning), **dict(cfg.predict)}.items()
            if "dim" in spec
        }
        stream_frames = {
            name: len(parse_index(spec["index"]))
            for name, spec in {**dict(cfg.conditioning), **dict(cfg.predict)}.items()
            if "index" in spec
        }
        # seconds, so streams at different rates land on identical floats for one moment
        tokens_per_step = {
            name: int(spec.get("views", 1)) * int(tuple(spec.get("grid", (1, 1)))[0])
            * int(tuple(spec.get("grid", (1, 1)))[1])
            for name, spec in {**dict(cfg.conditioning), **dict(cfg.predict)}.items()
            if "index" in spec
        }
        self._tokens_per_step = tokens_per_step
        stream_times = {
            name: tuple(
                LAST if isinstance(i, str) else i / float(spec.get("fps", 1.0))
                for i in parse_index(spec["index"])
            )
            for name, spec in {**dict(cfg.conditioning), **dict(cfg.predict)}.items()
            if "index" in spec
        }
        self.loss_terms = nn.ModuleList(
            [build_loss_term(term, default_streams, stream_dims, stream_frames, stream_times,
                             modules=self.encoders, tokens_per_step=tokens_per_step)
             for term in loss_cfg]
        )
        self._rollout_terms = [
            (term, float(t.get("weight", 1.0)) != 0.0)
            for term, t in zip(self.loss_terms, loss_cfg) if t.get("type") == "rollout"
        ]
        for term, _ in self._rollout_terms:
            if cfg.predict.get(term.stream, {}).get("type") != "direct":
                raise ValueError(f"rollout term names '{term.stream}', not a direct stream")
        for t in loss_cfg:
            if t.get("type") == "ht" and not cfg.predict.get(t["stream"], {}).get("scale"):
                raise ValueError(
                    f"ht term names '{t['stream']}', which declares no `scale:` head "
                    f"(token | pool | step) for its residual scale"
                )
        # weight decay on every >= 2-D trainable tensor but those matching `no_decay`
        # (fnmatch over parameter names); 1-D tensors (biases, norms) never decay
        self.optim_groups = dict(cfg.get("optim_groups") or {})
        # `weight: 0.0` declares a MONITORING term: evals report it, training never
        # pays for it -- loss() skips these in train mode and their artifacts (an
        # observed `integration` costs num_steps extra trunk passes) are not built
        scored = [float(term.get("weight", 1.0)) != 0.0 for term in loss_cfg]
        if loss_cfg and not any(scored):
            raise ValueError(
                "every `losses:` term has weight 0.0, so training has no objective. At "
                "least one term must be scored; weight 0.0 means eval-only."
            )

        def _raw_streams(entries):
            return {
                term["stream"] for term in entries
                if term.get("type") == "ae_recon"
                or (term.get("type") == "integration" and term.get("decoder"))
            }

        self._raw_loss_streams = _raw_streams(loss_cfg)
        self._raw_loss_streams_scored = _raw_streams(
            [term for term, keep in zip(loss_cfg, scored) if keep]
        )
        unknown = self._raw_loss_streams - set(cfg.predict)
        if unknown:
            raise ValueError(f"ae_recon names {sorted(unknown)}, not predict streams")

        # every integration term reads one shared solve, so num_steps is one setting
        self._integration = None
        int_terms = [term for term in loss_cfg if term.get("type") == "integration"]
        if int_terms:
            unknown = {t["stream"] for t in int_terms} - set(cfg.predict)
            if unknown:
                raise ValueError(f"integration names {sorted(unknown)}, not predict streams")
            not_flow = [t["stream"] for t in int_terms
                        if cfg.predict[t["stream"]].get("type", "flow") != "flow"]
            if not_flow:
                raise ValueError(
                    f"integration names {sorted(not_flow)}, which are not flow streams: "
                    f"there is no ODE to integrate"
                )
            solve = {int(t.get("num_steps") or cfg.get("num_flow_steps", 1))
                     for t in int_terms}
            if len(solve) > 1:
                raise ValueError(
                    f"integration terms disagree on the solve ({sorted(solve)} steps); every "
                    f"term reads the same endpoint, so num_steps must match"
                )
            num_steps = solve.pop()
            assert num_steps >= 1, "integration num_steps must be >= 1"
            self._integration = {
                "num_steps": num_steps,
                "streams": [t["stream"] for t in int_terms],
                # false when every integration term is observed-only: the solve then runs
                # in evals alone, and a training step costs exactly what it did before
                "scored": any(float(t.get("weight", 1.0)) != 0.0 for t in int_terms),
            }

        self._stream_axes = {}
        for name, spec in entries.items():
            if "index" not in spec:
                continue
            grid = tuple(spec.get("grid", (1, 1)))
            self._stream_axes[name] = (
                len(parse_index(spec["index"])),
                int(spec.get("views", 1)) * int(grid[0]) * int(grid[1]),
            )

        self.predictor = self._build_predictor(cfg)

    def _build_predictor(self, cfg):
        """Return the predictor nn.Module (built from cfg.model + the spec)."""
        raise NotImplementedError

    def _forward_train(self, clean, sources, cond, targets, seeds=None):
        """Turn the encoded context into a (pred, target) pair, both dicts keyed by predict
        stream, for the `losses:` terms. clean/sources/cond are the encoded conditioning
        streams routed by `via` (tokens / cross / cond); targets are the encoded predict
        streams. `seeds` (optional, per flow stream) are A2A source draws replacing the
        Gaussian x0; a paradigm without a flow x0 ignores them.
        """
        raise NotImplementedError

    @property
    def backbone(self):
        """The predictor the loss layer runs against."""
        return self.predictor

    def run_encoder(self, name, x, batch=None):
        """Run x through encoder `name`, first through the upstream chain its `input`
        links declare. A frozen stage runs under no_grad unless a trainable stage
        upstream needs gradients through it. A stage listing `meta_fields` (e.g. the
        embodiment id a per-embodiment projection selects its weights by) reads them
        from `batch` as keyword arguments.
        """
        return self._run_stages(self._encoder_chains[name], x, batch)

    def _run_stages(self, stages, x, batch=None):
        for stage in stages:
            encoder = self.encoders[stage]
            fields = getattr(encoder, "meta_fields", ())
            if fields and batch is None:
                raise ValueError(f"encoder '{stage}' needs batch fields {list(fields)}")
            kwargs = {f: batch[f] for f in fields}
            if encoder.frozen and not (torch.is_tensor(x) and x.requires_grad):
                with torch.no_grad():
                    x = encoder(x, **kwargs)
            else:
                x = encoder(x, **kwargs)
        return x

    def optim_params(self, lr=None):
        """Parameter groups from `optim_groups`; without it, every parameter in one group.

          no_decay  fnmatch patterns over parameter names kept at weight_decay 0, beside
                    every 1-D tensor (biases, norms); everything else >= 2-D decays
          lr_scale  {pattern: factor}: a parameter matching a pattern trains at
                    factor * `lr` (the optimizer's base LR), e.g. `encoders.levjepa.*: 0.1`
                    for an unfrozen encoder slower than the predictor; the first matching
                    pattern wins. The LR schedule multiplies every group alike.

        Frozen parameters are left out.
        """
        if not self.optim_groups:
            return self.parameters()
        patterns = list(self.optim_groups.get("no_decay") or [])
        scales = {str(k): float(v) for k, v in dict(self.optim_groups.get("lr_scale") or {}).items()}
        if scales and lr is None:
            raise ValueError("optim_groups.lr_scale needs the optimizer's base lr")
        groups = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            no_decay = param.ndim < 2 or any(fnmatch.fnmatchcase(name, p) for p in patterns)
            scale = next((v for p, v in scales.items() if fnmatch.fnmatchcase(name, p)), 1.0)
            groups.setdefault((no_decay, scale), []).append(param)
        unused = [p for p in scales
                  if not any(fnmatch.fnmatchcase(n, p) for n, _ in self.named_parameters())]
        if unused:
            raise ValueError(f"optim_groups.lr_scale patterns {unused} match no parameter")
        out = []
        for (no_decay, scale), params in sorted(groups.items(), key=lambda kv: kv[0]):
            group = {"params": params}
            if no_decay:
                group["weight_decay"] = 0.0
            if scale != 1.0:
                group["lr"] = lr * scale
            out.append(group)
        return out

    def cacheable_streams(self):
        """The spec entries whose frozen-root encoder output can be precomputed, as
        `{stream: {field, offsets, encoder}}`. A stream qualifies when its encoder chain
        starts frozen and it reads one un-stitched visual field with no `"last"` row.
        """
        entries = {**dict(self.conditioning), **dict(self.predict_spec)}
        out = {}
        for name, spec in entries.items():
            if "index" not in spec:
                continue
            fields = spec_fields(name, spec)
            enc = spec.get("encoder")
            rel, has_last = split_index(raw_index(spec))
            root = self.encoders[self._encoder_chains[enc][0]] if enc is not None else None
            # frozen alone is not enough: a probe run freezes everything, and the cache
            # holds visual windows only (supports_encoding_cache is per encoder type)
            if (root is not None and len(fields) == 1 and not has_last
                    and root.frozen and root.supports_encoding_cache):
                out[name] = {
                    "field": fields[0],
                    "offsets": parse_index(raw_index(spec)),
                    "encoder": self._encoder_chains[enc][0],
                }
        return out

    def raw_pixel_fields(self):
        """Fields some entry reads WITHOUT a cacheable frozen-root chain -- these must
        keep arriving as pixels even when other streams on the field are served from
        the encoding cache (the trainer builds the dataset in keys+pixels mode for the
        intersection).
        """
        cacheable = set(self.cacheable_streams())
        needed = set()
        entries = {**dict(self.conditioning), **dict(self.predict_spec), **self._source_entries}
        for name, spec in entries.items():
            if "index" in spec and name not in cacheable:
                needed.update(spec_fields(name, spec))
        return needed

    def attach_encoding_cache(self, cache):
        """Serve the cached streams from `cache` whenever a batch carries their
        `enc_cache/<field>` keys. Batches with pixel fields (validation, rollouts) keep
        the live path -- the dispatch is the batch's schema, decided per batch.
        """
        missing = set(self.cacheable_streams()) - set(cache.stores)
        if missing:
            raise ValueError(f"encoding cache lacks streams {sorted(missing)}")
        self._enc_cache = cache

    def _raw(self, name, spec, batch):
        """A stream's own raw window, sliced out of the shared field and stitched, but not
        yet encoded -- what its encoder consumes, and what a pixel probe compares against.
        """
        sel = self._selectors[name]
        views = [batch[field] for field in spec_fields(name, spec)]
        if sel is not None:
            views = [view[:, sel] for view in views]
        return views[0] if len(views) == 1 else stitch_views(views, spec.get("stitch", "horizontal"))

    def _encode(self, name, spec, batch):
        fields = spec_fields(name, spec)
        key_field = f"enc_cache/{fields[0]}"
        cache = getattr(self, "_enc_cache", None)
        if key_field in batch:
            # dispatch is per stream: a field can carry keys AND pixels, when another
            # entry reads it raw
            if cache is not None and name in cache.stores:
                device = next(self.parameters()).device
                x = cache.lookup(name, batch[key_field], device)
                return self._run_stages(self._encoder_chains[spec["encoder"]][1:], x)
            if fields[0] not in batch:
                raise ValueError(
                    f"batch carries '{key_field}' but stream '{name}' has neither a "
                    f"cache entry nor the field's pixels: attach a cache serving it "
                    f"(attach_encoding_cache) or feed pixel batches"
                )
        raw = self._raw(name, spec, batch)
        enc = spec.get("encoder")
        if not enc:
            return raw
        if int(spec.get("views", 1)) > 1:
            return self._encode_views(name, spec, enc, raw, batch)
        return self.run_encoder(enc, raw, batch)

    def _encode_views(self, name, spec, enc, raw, batch):
        """A (B, T, V, ...) multi-camera window -> (B, T * V * k, dim) tokens, time-major
        then view. A token the batch marks invalid (`valid.steps` / `valid.views`) is never
        encoded and comes out as zeros.

        With a frame-local chain (every stage's `frame_local`) the field's real frames are
        encoded once as single-frame clips and the stream slices its rows out of that, so a
        context window and its shift-by-one target share one encoder pass. Otherwise each
        present camera is encoded as its own (T, ...) clip, padded steps included."""
        views = int(spec["views"])
        if raw.shape[2] != views:
            raise ValueError(f"stream '{name}' declares {views} views, field has {raw.shape[2]}")
        fields = spec_fields(name, spec)
        block = spec.get("valid") or {}
        if (len(fields) == 1 and self._raw_per_step.get(name, 1) == 1
                and self._frame_local(enc)):
            full = self._frame_encoded(fields[0], enc, views, block, batch)
            sel = self._selectors[name]
            out = full if sel is None else full[:, sel]
            if out.shape[1] != raw.shape[1]:
                raise ValueError(f"stream '{name}': {raw.shape[1]} rows, sliced {out.shape[1]}")
            return out.reshape(out.shape[0], -1, out.shape[-1])
        B, T = raw.shape[:2]
        clips = raw.transpose(1, 2)                                  # (B, V, T, ...)
        field = block.get("views")
        keep = batch[field].bool() if field else torch.ones(B, views, dtype=torch.bool,
                                                            device=raw.device)
        z = self.run_encoder(enc, clips[keep], batch=self._rows(batch, keep))
        k = z.shape[1] // T
        out = z.new_zeros(B, views, T, k, z.shape[-1])
        out[keep] = z.view(-1, T, k, z.shape[-1])
        return out.transpose(1, 2).reshape(B, T * views * k, z.shape[-1])

    def _frame_local(self, enc):
        return all(getattr(self.encoders[s], "frame_local", False)
                   for s in self._encoder_chains[enc])

    def _frame_encoded(self, field, enc, views, block, batch):
        """(B, R, V, k, dim) latents of every row of a (B, R, V, ...) camera field, from one
        encoder pass over its real frames only (valid step AND valid camera slot), zeros
        elsewhere. Memoized per (field, encoder) while a loss context is being built, so
        every stream reading the field reuses the pass."""
        memo = getattr(self, "_frame_memo", None)
        if memo is not None and (field, enc) in memo:
            return memo[(field, enc)]
        x = batch[field]
        B, R = x.shape[:2]
        keep = torch.ones(B, R, views, dtype=torch.bool, device=x.device)
        if block.get("steps"):
            steps = batch[block["steps"]].bool()
            if steps.shape[1] != R:
                raise ValueError(f"'{block['steps']}' has {steps.shape[1]} rows, '{field}' {R}")
            keep &= steps[:, :, None]
        if block.get("views"):
            keep &= batch[block["views"]].bool()[:, None, :]
        rows = {k: v[:, None, None].expand(B, R, views)[keep] for k, v in batch.items()
                if torch.is_tensor(v) and v.ndim == 1 and v.shape[0] == B}
        z = self.run_encoder(enc, x[keep].unsqueeze(1), batch=rows)   # (M, k, dim)
        out = z.new_zeros(B, R, views, *z.shape[1:])
        out[keep] = z
        if memo is not None:
            memo[(field, enc)] = out
        return out

    @staticmethod
    def _rows(batch, keep):
        """The batch fields a per-view encoder pass may read, repeated per kept view."""
        B, V = keep.shape
        return {k: v.unsqueeze(1).expand(B, V, *v.shape[1:])[keep]
                for k, v in batch.items() if torch.is_tensor(v) and v.ndim >= 1
                and v.shape[0] == B and v.ndim <= 2}

    def _token_valid(self, name, spec, batch):
        """(B, N) bool validity of a stream's tokens from its `valid:` block, or None.

        `valid.steps` names a (B, rows) bool field laid out like the stream's own field
        (same rows, so the stream's selector slices it), `valid.views` a (B, views) bool
        field. A token is valid when its step and its camera slot both are.
        """
        block = spec.get("valid")
        if not block or "index" not in spec:
            return None
        steps = len(parse_index(spec["index"]))
        views = int(spec.get("views", 1))
        grid = tuple(spec.get("grid", (1, 1)))
        span = int(grid[0]) * int(grid[1])
        ref = next(v for v in batch.values() if torch.is_tensor(v))
        B = ref.shape[0]
        mask = torch.ones(B, steps, views, dtype=torch.bool, device=ref.device)
        if block.get("steps"):
            rows = batch[block["steps"]].bool()
            sel = self._selectors[name]
            if sel is not None:
                rows = rows[:, sel] if self._raw_per_step.get(name, 1) == 1 else \
                    rows[:, sel][:, ::self._raw_per_step[name]]
            mask &= rows[:, :steps, None]
        if block.get("views"):
            mask &= batch[block["views"]].bool()[:, None, :]
        return mask[..., None].expand(-1, -1, -1, span).reshape(B, -1)

    def _trunk_inputs(self, batch):
        """The identity fields and per-stream key validity the trunk reads, from `batch`."""
        meta = {f: batch[f] for f in getattr(self.predictor, "meta_fields", ())}
        valid = {}
        for name, spec in self.conditioning.items():
            if spec.get("via", "tokens") == "tokens":
                v = self._token_valid(name, spec, batch)
                if v is not None:
                    valid[name] = v
        return meta, valid

    def _condition(self, batch, token_streams=None):
        """Encode every conditioning stream once, routing it by `via`.

        `via: null` encodes the stream but feeds it to nothing in the predictor; it is
        read from ctx["latent"] by loss terms and by stream-mode `source:` blocks.
        `token_streams` (None = all) restricts which `via: tokens` streams are encoded
        AT ALL -- a subset integration's omitted clean streams then need no batch rows.
        cross/cond/null streams are always encoded.
        """
        clean, sources, cond, hidden = {}, {}, {}, {}
        for name, spec in self.conditioning.items():
            via = spec.get("via", "tokens")
            assert via in ("tokens", "cross", "cond", None), f"unknown via '{via}' for '{name}'"
            if via == "tokens" and token_streams is not None and name not in token_streams:
                continue
            enc = self._encode(name, spec, batch)
            {"cross": sources, "cond": cond, None: hidden}.get(via, clean)[name] = enc
        return clean, sources, cond, hidden

    def build_flow_seeds(self, batch, encoded, training=True):
        """The x0 seed of every flow stream with a `source:` block, in that stream's flow
        space, or None when no stream has one. `encoded` holds the already-encoded
        conditioning streams (stream-mode sources read them for free); field-mode sources
        slice their raw window out of `batch` and run it through their own encoder chain.
        """
        seeds = {}
        for name, src in self._flow_sources.items():
            seeds[name] = self._build_seed(name, src, batch, encoded, training)
        return seeds or None

    def _build_seed(self, name, src, batch, encoded, training):
        spec = self.predict_spec[name]
        grid = tuple(spec.get("grid", (1, 1)))
        span = int(grid[0]) * int(grid[1])
        n_tokens = len(parse_index(spec["index"])) * span
        # eval builds the seed EXACTLY as training does -- an eval-only off-switch for
        # the declared noise here would be a silent train/eval mismatch; do not add one
        if "noise_at_eval" in src:
            raise ValueError(
                f"source of '{name}' declares `noise_at_eval`, which no longer exists: "
                f"eval seeds always carry the same noise the training seeds did. Delete "
                f"the key (and lower `noise_std` if the intent was a quieter eval)."
            )

        if src.get("stream"):
            z = encoded[src["stream"]]
            rows = src.get("rows")
            if rows is not None:
                # index STEPS, negatives from the end, grid tokens kept together
                cspec = self.conditioning[src["stream"]]
                cgrid = tuple(cspec.get("grid", (1, 1)))
                cspan = int(cgrid[0]) * int(cgrid[1])
                steps = z.shape[1] // cspan
                sel = [int(r) % steps for r in parse_index(rows)]
                z = z.unflatten(1, (steps, cspan))[:, sel].flatten(1, 2)
            enc = src.get("encoder")
            if enc:
                # already encoded, so one adapter module rather than a chain from raw
                z = self.encoders[enc](z)
        else:
            raw = self._raw(f"{name}.source", self._source_entries[f"{name}.source"], batch)
            raw = raw.float()
            tile = int(src.get("tile") or 0)
            if tile:
                raw = raw.repeat(1, tile, 1)
            raw_std = float(src.get("raw_noise_std") or 0.0)
            if raw_std:
                raw = raw + raw_std * torch.randn_like(raw)
            enc = src.get("encoder")
            z = self.run_encoder(enc, raw, batch) if enc else raw

        fill, m = src.get("fill", "repeat_last"), z.shape[1]
        if m != n_tokens:
            if fill == "repeat_last":
                z = (z[:, -n_tokens:] if m > n_tokens
                     else torch.cat([z, z[:, -1:].expand(-1, n_tokens - m, -1)], dim=1))
            elif fill == "tile":
                z = z.repeat(1, -(-n_tokens // m), 1)[:, :n_tokens]
            else:
                raise ValueError(
                    f"source of '{name}' yields {m} tokens but the stream flows "
                    f"{n_tokens}; set `fill: repeat_last|tile` to widen it"
                )
        if z.shape[-1] != int(spec["dim"]):
            raise ValueError(
                f"source of '{name}' is {z.shape[-1]}-dim but the stream flows "
                f"{spec['dim']}-dim tokens; route it through a matching `encoder:`"
            )
        std = float(src.get("noise_std") or 0.0)
        if std:
            z = z + std * torch.randn_like(z)
        # `detach: true` stop-grads the seed: the flow loss then shapes the encoder only
        # through the target x1, not by dragging the seed's source (the context) toward
        # the future -- the temporal-collapse pull an unfrozen encoder acts on
        if src.get("detach"):
            z = z.detach()
        return z

    def _forward_integrate(self, clean, sources, cond, seeds, num_steps):
        """The differentiable ODE endpoint per flow stream, in flow space (decoders NOT
        applied). Same conditioning and same x0 as `_forward_train`.
        """
        raise NotImplementedError

    def _forward_rollout(self, stream, steps, prefix, clean, sources, cond, targets, pred,
                         meta, valid, target_valid):
        """A `rollout` term's autoregressive passes: a list (one per prefix) of per-step
        (pred, target, valid, copy) tuples, `copy` being the last true latent before the
        rollout (the no-motion baseline). See `losses.RolloutLoss`."""
        raise NotImplementedError

    def _build_context(self, batch):
        self._frame_memo = {}
        try:
            return self._build_context_inner(batch)
        finally:
            self._frame_memo = None

    def _build_context_inner(self, batch):
        """Assemble the artifacts the loss terms read: per-stream encoder `latent`, predictor
        `pred`, and the regression `target` (plus the `raw` windows for the terms scoring in
        raw space, and the `integrated` ODE endpoint when an integration term asks for it).
        Conditioning latents feed the predictor; predicted latents are the targets.
        """
        clean, sources, cond, hidden = self._condition(batch)
        latent = {**clean, **sources, **cond, **hidden}
        targets = {name: self._encode(name, spec, batch) for name, spec in self.predict_spec.items()}
        latent.update(targets)
        seeds = self.build_flow_seeds(batch, {**clean, **sources, **cond, **hidden}, training=True)
        meta, valid = self._trunk_inputs(batch)
        extra = {"meta": meta, "valid": valid} if (meta or valid) else {}
        # a paradigm with `scale:` streams also returns their raw residual scales
        pred, target, *scale = self._forward_train(clean, sources, cond, targets,
                                                   seeds=seeds, **extra)
        ctx = {"latent": latent, "pred": pred, "target": target}
        if scale and scale[0]:
            ctx["scale"] = scale[0]
        # the no-motion baseline of a `direct` stream: its input latent at each position,
        # read as a prediction of the next step (input position p predicts target p)
        copy = {name: latent[spec["input"]] for name, spec in self.predict_spec.items()
                if spec.get("type") == "direct" and spec.get("input") in latent
                and latent[spec["input"]].shape == target[name].shape}
        if copy:
            ctx["copy"] = copy
        target_valid = {}
        for name, spec in self.predict_spec.items():
            v = self._token_valid(name, spec, batch)
            if v is not None:
                target_valid[name] = v
        if target_valid:
            ctx["valid"] = target_valid
        # `valid.dims`: a (B, dim) bool field marking the coordinates a sample fills (a
        # mixture's zero-padded action width); the losses average over those alone
        dim_valid = {name: batch[spec["valid"]["dims"]].bool()
                     for name, spec in self.predict_spec.items()
                     if (spec.get("valid") or {}).get("dims")}
        if dim_valid:
            ctx["dim_valid"] = dim_valid
        if valid or target_valid:
            # every stream's token validity, for the terms scoring encoder latents (sigreg)
            ctx["latent_valid"] = {**valid, **target_valid}
        wanted = [term for term, scored in self._rollout_terms if scored or not self.training]
        if wanted:
            ctx["rollout"] = {
                term.label: self._forward_rollout(
                    term.stream, term.steps, term.prefix, clean, sources, cond, targets,
                    pred, meta, valid, target_valid.get(term.stream))
                for term in wanted
            }
        # observed (`weight: 0.0`) terms run in eval only, so in train mode build only what
        # the scored terms read -- see the `scored` note where the terms are built
        raw_streams = (self._raw_loss_streams_scored if self.training
                       else self._raw_loss_streams)
        if raw_streams:
            ctx["raw"] = {
                name: self._raw(name, self.predict_spec[name], batch).float()
                for name in raw_streams
            }
        if self._integration is not None and (
            self._integration["scored"] or not self.training
        ):
            ctx["integrated"] = self._forward_integrate(
                clean, sources, cond, seeds, self._integration["num_steps"],
            )
            # only `units: raw` terms read this; building it is a couple of tiny tensors
            scales = {n: self._unit_scale(n) for n in self._integration["streams"]}
            ctx["unit_scale"] = {n: v for n, v in scales.items() if v is not None}
        return ctx

    def _unit_scale(self, name):
        """Per-dim factor turning a NORMALIZED error on this stream into the field's own
        units, or None when the stream has no such field (a codec latent) or the model
        carries no stats.

        Normalization is `(x - q_lo) / span * 2 - 1` (percentile) or `(x - mean) / std`, so
        an error scales by `span / 2` or `std` respectively -- the inverse of the slope, and
        the reason a wider dataset silently lowers every normalized metric.
        """
        stats = getattr(self, "norm_stats", None)
        if not stats:
            return None
        spec = self.predict_spec[name]
        field = spec.get("from", name)
        entry = stats.get(field)
        if not entry:
            return None
        method = getattr(self, "norm_method", "mean_std")
        if method == "percentile":
            lo = torch.as_tensor(entry["q_lo"], dtype=torch.float32)
            hi = torch.as_tensor(entry["q_hi"], dtype=torch.float32)
            return (hi - lo).clamp_min(1e-8) / 2
        return torch.as_tensor(entry["std"], dtype=torch.float32).clamp_min(1e-8)

    def _variance_logs(self, latent, valid=None):
        """Share of variance carried along the time and grid axes, per stream FAMILY.

        Streams reading the same source field with matching grid span and latent width --
        a camera's context and its predicted future -- are concatenated along time and
        measured as ONE trajectory, so the estimate covers the full context+future token
        set rather than each half individually (grouping is automatic, from the specs'
        `from:` fields; no configuration). A family labels itself by its members' common
        prefix (scene_context + scene_future -> `var/scene/...`); a stream with no
        partner keeps its own name, exactly as before.

        Each value is the variance along that axis over the family's total: 1.0 is what
        independent tokens give, 0 means constant along the axis, i.e. collapsed. Tokens
        `valid` marks invalid (padded steps, empty camera slots) are left out of both.
        """
        valid = valid or {}
        streams = {**dict(self.conditioning), **dict(self.predict_spec)}
        groups = {}
        for name, (steps, span) in self._stream_axes.items():
            z = latent.get(name)
            if z is None or z.ndim != 3 or z.shape[1] != steps * span:
                continue
            z = z.detach().float().unflatten(1, (steps, span))
            m = valid.get(name)
            m = (torch.ones(z.shape[:3], dtype=torch.bool, device=z.device) if m is None
                 else m.bool().view(z.shape[:3]))
            spec = streams.get(name)
            field = spec.get("from", name) if spec is not None else name
            groups.setdefault((field, span, z.shape[-1]), []).append((name, z, m))
        logs = {}
        for (field, span, _), members in groups.items():
            names = [n for n, _, _ in members]
            z = torch.cat([g for _, g, _ in members], dim=1)
            m = torch.cat([g for _, _, g in members], dim=1)
            label = (names[0] if len(names) == 1
                     else os.path.commonprefix(names).rstrip("_") or field)
            real = z[m]
            if real.shape[0] < 2:
                continue
            total = real.var(dim=0).mean().clamp_min(1e-12)
            for axis, dim in (("temporal", 1), ("spatial", 2)):
                v = self._masked_var(z, m, dim) if z.shape[dim] > 1 else None
                if v is not None:
                    logs[f"var/{label}/{axis}"] = v / total
        return logs

    @staticmethod
    def _masked_var(z, m, dim):
        """Unbiased variance of (B, T, S, D) `z` along `dim` over the tokens `m` keeps,
        averaged over the slices holding at least two of them and over D; None when no
        slice does (a batch of single-camera clips has no spatial spread to measure)."""
        w = m.unsqueeze(-1).to(z.dtype)
        n = w.sum(dim, keepdim=True)
        mu = (z * w).sum(dim, keepdim=True) / n.clamp_min(1)
        var = ((z - mu).square() * w).sum(dim, keepdim=True) / (n - 1).clamp_min(1)
        ok = (n > 1).expand_as(var)
        return var[ok].mean() if ok.any() else None

    def loss(self, batch):
        ctx = self._build_context(batch)
        logs, total = {}, 0.0
        for term in self.loss_terms:
            # a `weight: 0.0` term is observed, not optimized: computing it during training
            # would cost real time and add a `loss/<label>` series that no gradient backs
            if self.training and term.weight == 0.0:
                continue
            val = term(ctx)
            total = total + term.weight * val
            logs[f"loss/{term.label}"] = val.detach()
            for key, value in getattr(term, "step_logs", {}).items():
                logs[f"loss/{term.label}/{key}"] = value
        var = self._variance_logs(ctx["latent"], ctx.get("latent_valid"))
        return {"loss": total, **logs, **var}

    def summary(self):
        return {**super().summary(), "obs_len": self.obs_len}
