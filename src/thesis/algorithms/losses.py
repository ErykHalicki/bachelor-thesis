"""Configurable training losses for the predictor algorithms.

A loss `term` is a small nn.Module built from one `losses:` config entry. Every term reads a
per-call context dict (see `PredictiveModel._build_context`) with three artifacts and never
anything else, so a term can only ever touch what it is meant to:

    ctx["latent"]      encoder output per encoder-backed stream (conditioning + predict)
    ctx["pred"]        predictor output per predict stream
    ctx["target"]      the regression target per predict stream
    ctx["raw"]         the normalized batch window per predict stream, for the terms that
                       score in raw space (`ae_recon`, a raw-space `integration`)
    ctx["integrated"]  the ODE endpoint per flow stream, present only when an `integration`
                       term asked for the (expensive) differentiable solve

A term's `weight` also decides WHEN it runs. `weight: 0.0` makes it an observed term: evals
compute and report it under the usual `loss/<label>` name, training skips it entirely and
does not build the artifacts only it reads. That is how an expensive diagnostic -- an
`integration` term, whose solve costs `num_steps` extra trunk passes -- can be watched on
the held-out split without slowing every training step or changing the objective.

`PredictionLoss` (config types `flow` and `prediction`) is the MSE of `pred` against `target`;
the two names name the same objective for the two predictor paradigms — for the flow-matching
`dit_predictor` the pred is a velocity and the target is `x1 - x0`, for the regression
`vit_predictor` the pred is the next latent and the target is the next latent. `sigreg`
regularizes `latent` toward an isotropic Gaussian. Because `sigreg` only indexes
`ctx["latent"]`, it structurally cannot be applied to a prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SIGReg", "PredictionLoss", "SIGRegLoss", "AEReconLoss", "IntegrationLoss",
           "build_loss_term"]


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (LeWorldModel, single-GPU): pushes embeddings
    toward an isotropic Gaussian via the Epps-Pulley squared ECF distance over random 1D
    projections. Reference: https://github.com/lucas-maes/le-wm/blob/main/module.py

    forward(embeddings: (B, N, D)) -> scalar. Tokens are flattened to (B*N, D) and fit as a
    single isotropic Gaussian, so the statistic carries no spatial detail: a CLS vector
    (B, 1, D) and a full patch grid (B, N, D) are both just bags of D-dim samples.

    Scaling: the classical statistic is M·∫|φ_emp-φ|² w dt with M the sample count, and
    M here is every flattened row, B·tokens. Since E|φ_emp-φ|² = (1-φ²)/M for iid samples,
    that cancels the 1/M and leaves ∫(1-φ²) w dt ≈ 1.05 under the null, whatever B, the
    token count, or D. So ~1.05 reads as "indistinguishable from an iid isotropic Gaussian
    sample" in every configuration, and one `weight` transfers across grid sizes and
    bagging statistics. Above 1.05 means the tokens are correlated -- effective sample
    count below nominal, so k identical tokens sit ~k× up. That is the anti-collapse
    signal, not an artifact of counting correlated tokens as independent.

    Note this diverges from le-wm, which scales by B alone and so puts the null at
    1.05/tokens. Divide by the tokens per bag to compare against their numbers.

    Two things ~1.05 is not. It is the null MEAN, not a lower bound: it fluctuates
    (std ~0.05), and a batch whose empirical moments are more exact than a real sample's
    -- an exactly whitened one, say -- scores well below. And it is the value of sampling
    noise, not of non-Gaussianity: a genuinely wrong distribution grows linearly in M, so
    a bigger batch sharpens deviations without moving the reference.
    """

    def __init__(self, knots=17, num_proj=1024, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=dtype)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=dtype)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings):
        x = embeddings.reshape(-1, embeddings.size(-1)).to(self.dtype)
        A = torch.randn(x.size(-1), self.num_proj, device=x.device, dtype=self.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (x @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        return (err @ self.weights).mean() * x.size(0)


class PredictionLoss(nn.Module):
    """MSE between the predictor's output and its regression target, summed over the term's
    streams. Serves both paradigms: `dit_predictor` (velocity vs `x1 - x0`) and `vit_predictor`
    (next latent vs next latent).
    """

    def __init__(self, streams, weight, label="prediction"):
        super().__init__()
        self.streams = list(streams)
        self.weight = weight
        # a single-stream term logs under its stream name; a term spanning several
        # logs under the loss type's label
        self.label = self.streams[0] if len(self.streams) == 1 else label

    def forward(self, ctx):
        return sum(F.mse_loss(ctx["pred"][n], ctx["target"][n]) for n in self.streams)


class AEReconLoss(nn.Module):
    """Autoencoder reconstruction over one predict stream (A2A's L_AE): decode the
    stream's ENCODER latent -- the same x1 the flow term regresses toward -- back through
    `decoder` and penalize the error against the raw window (`ctx["raw"]`, the stream's
    normalized batch rows). Grounds a learned flow-latent space (`chunk_mlp` codec) so
    the encoder cannot collapse: a latent the decoder must reconstruct 36 actions from
    has to keep them apart.

    `decoder` is a shared reference to an `encoders:` entry (usually the same module the
    stream names as its `decoder:`), so this term trains the exact codec inference uses.
    `norm` picks l1 (the A2A paper's choice, default) or mse.
    """

    def __init__(self, stream, decoder, weight, norm="l1"):
        super().__init__()
        assert norm in ("l1", "mse"), f"unknown ae_recon norm '{norm}'"
        self.streams = [stream]
        self.stream = stream
        self.decoder = decoder
        self.weight = weight
        self.norm = norm
        self.label = f"ae_{stream}"

    def forward(self, ctx):
        recon = self.decoder(ctx["latent"][self.stream])
        raw = ctx["raw"][self.stream]
        return F.l1_loss(recon, raw) if self.norm == "l1" else F.mse_loss(recon, raw)


class IntegrationLoss(nn.Module):
    """Inference-consistency over one flow stream: score the endpoint the SAMPLER produces,
    not the velocity at a random t.

    The flow term supervises v(x_t, t) at independently drawn t, which is only an unbiased
    surrogate in the infinite-step limit; a 4-step Euler rollout accumulates the per-step
    error, and nothing in training ever saw the states its own trajectory visits. This term
    closes that loop: `ctx["integrated"]` is the endpoint of a differentiable N-step solve
    started from the same x0 the stream would use at eval (its A2A `source:` seed, else
    noise), and the error is taken against the same x1 the flow term regresses toward.

    `decoder` (optional, a shared `encoders:` entry) scores in RAW space instead: decode the
    endpoint and compare against `ctx["raw"]`, which is what the robot actually executes and
    the only place a codec-latent stream's error is in real units. Without it the comparison
    is in flow space against the encoded target.

    Nothing here is stop-gradded: the term's error propagates into the predictor through the
    whole solve, and into a jointly trained encoder through both the seed it starts from and
    the target it aims at. `num_steps` configures the solve itself and is read by the
    algorithm, not here (see `PredictiveModel._integration`).
    """

    def __init__(self, stream, weight, decoder=None, norm="mse", units="normalized"):
        super().__init__()
        assert norm in ("l1", "mse"), f"unknown integration norm '{norm}'"
        assert units in ("normalized", "raw"), (
            f"unknown integration units '{units}'; use 'normalized' or 'raw'"
        )
        self.streams = [stream]
        self.stream = stream
        self.decoder = decoder
        self.weight = weight
        self.norm = norm
        self.units = units
        self.label = f"int_{stream}" + ("_raw" if units == "raw" else "")

    def forward(self, ctx):
        z = ctx["integrated"][self.stream]
        if self.decoder is not None:
            pred, target = self.decoder(z), ctx["raw"][self.stream]
        else:
            pred, target = z, ctx["latent"][self.stream]
        if self.units == "raw":
            # `ctx["raw"]` means UN-ENCODED, not un-normalized: both branches are in
            # normalized units; rescaling by the normalizer's per-dim span is what
            # reaches the robot's units
            scale = (ctx.get("unit_scale") or {}).get(self.stream)
            if scale is None:
                raise ValueError(
                    f"integration term on '{self.stream}' asks for `units: raw`, but the "
                    f"model carries no norm_stats for the field it reads -- there is no "
                    f"scale to convert by. Train with a `normalize:` dataset, or drop to "
                    f"`units: normalized`"
                )
            scale = scale.to(pred.dtype).to(pred.device)
            pred, target = pred * scale, target * scale
        return F.l1_loss(pred, target) if self.norm == "l1" else F.mse_loss(pred, target)


class SIGRegLoss(nn.Module):
    """SIGReg over the encoder latents of the term's streams.

    `statistic` picks the sample-bag structure. Pooling an axis asserts the tokens along
    it are independent draws; separating an axis asserts each slice along it matches the
    target on its own. So a scheme prices dependence along axes it pools, and marginal
    drift along axes it separates:

        pooled (default)  one bag, (B, N, D) -> (B*N, D). Prices dependence on every
                          axis at once; blind to drift on any of them.
        per_timestep      le-wm's structure: one bag per frame position, over that
                          frame's tokens across the batch, averaged over the T positions.
                          Prices redundancy between a frame's tokens and drift between
                          frames; blind to temporal correlation, so a stream that is
                          constant over time passes.
        per_spatial       the transpose: one bag per grid/query position, over that
                          position's whole time window, averaged over the S positions.
                          Prices temporal dependence -- the only statistic that catches a
                          stream constant over time -- and drift between grid positions.

    Note what no statistic separates: collapse along an axis and legitimate smoothness
    along it are both dependence, so a term that prices one prices the other. Use the
    var/<stream>/{temporal,spatial} diagnostics to tell them apart.

    `concat` says how the term's streams relate, joining their (B, T, S, D) latents into
    one before the statistic runs. Without it each stream is scored separately and the
    scores are summed, so the streams are never compared to each other and each is
    normalized by its own axis lengths:

        null (default)  independent terms, summed. Two streams that are identical to each
                        other cost nothing.
        temporal        one timestream: context and future frames of one camera joined
                        end to end, each distinct time kept once. Reproduces le-wm, which
                        encodes the whole window at once and calls SIGReg once on it, so
                        every frame carries equal weight -- summing self-normalized terms
                        instead weights a 2-frame context 1.5x a 3-frame future, and
                        double-counts whatever the windows share (a shift-by-one target
                        overlaps its own context at 2 of 4 frames).
        spatial         extra grid/query positions of one frame: joins two cameras at the
                        same timesteps, so `per_timestep` prices them collapsing onto
                        each other.
        batch           extra samples of one distribution: same axes, more rows.

    Joining requires the other two axes to match (temporal needs equal S, spatial equal T,
    batch both), and every stream needs a frame count.

    Every term carries one bias-free linear head, shared by all of its streams — streams
    grouped into one term are assumed to come from the same encoder/source, so they share
    the projection; give a stream its own term to give it its own head. The head receives
    gradients only from this term and is unused at prediction time, so the backbone keeps
    the anti-collapse constraint (a deterministic head cannot manufacture variability) but
    is freed from making its raw features isotropic-Gaussian themselves.

    `proj_dim` sets its output width, defaulting to the stream dim. A square head is
    initialized to the identity, so the default measures the raw latents at init and only
    departs from them as it learns; a narrower `proj_dim` gets the usual random init.
    """

    CONCAT_DIMS = {"batch": 0, "temporal": 1, "spatial": 2}

    def __init__(self, streams, weight, knots=17, num_proj=1024, proj_dim=None,
                 stream_dims=None, statistic="pooled", stream_frames=None, concat=None,
                 stream_times=None):
        super().__init__()
        assert statistic in ("pooled", "per_timestep", "per_spatial"), (
            f"unknown statistic '{statistic}'"
        )
        assert concat is None or concat in self.CONCAT_DIMS, (
            f"unknown concat '{concat}'; use {sorted(self.CONCAT_DIMS)} or omit it"
        )
        self.streams = list(streams)
        self.weight = weight
        # statistic and concat qualify the label, so several terms over one set of
        # streams do not overwrite each other's logs
        self.label = "sigreg_" + "_".join(self.streams)
        if statistic != "pooled":
            self.label += f"_{statistic}"
        if concat is not None:
            self.label += f"_{concat}"
        self.sigreg = SIGReg(knots=knots, num_proj=num_proj)
        self.statistic = statistic
        self.concat = concat
        self.stream_frames = dict(stream_frames or {})
        self.stream_times = {
            n: t for n, t in (stream_times or {}).items() if n in self.streams
        }
        if statistic in ("per_timestep", "per_spatial") or concat is not None:
            missing = [n for n in self.streams if not self.stream_frames.get(n)]
            assert not missing, (
                f"{concat or statistic} sigreg needs the frame count of {missing} "
                f"(streams without an `index` have no time axis)"
            )
        dims = {n: (stream_dims or {}).get(n) for n in self.streams}
        if any(d is None for d in dims.values()):
            raise ValueError(
                f"sigreg needs the latent dim of every stream it reads to size its head, "
                f"got {dims}; a stream must declare `dim`"
            )
        if len(set(dims.values())) != 1:
            raise ValueError(
                f"a sigreg term shares one head across its streams, which requires equal "
                f"stream dims, got {dims}; split into separate terms"
            )
        in_dim = dims[self.streams[0]]
        out_dim = in_dim if proj_dim is None else int(proj_dim)
        self.head = nn.Linear(in_dim, out_dim, bias=False)
        if out_dim == in_dim:
            # a square head starts as the identity, so omitting proj_dim costs nothing at
            # init and the term measures the raw latents until the head moves
            nn.init.eye_(self.head.weight)

    def _score(self, grid):
        """(B, T, S, D) -> the statistic, averaged over however many bags it splits into."""
        if self.statistic == "pooled":
            bags = [grid.flatten(1, 2)]
        elif self.statistic == "per_timestep":
            bags = [grid[:, i] for i in range(grid.shape[1])]
        else:
            bags = [grid[:, :, j] for j in range(grid.shape[2])]
        return sum(self.sigreg(bag) for bag in bags) / len(bags)

    def _grid(self, ctx, name):
        z = ctx["latent"][name]
        if self.head is not None:
            z = self.head(z)
        # no concat with the pooled statistic never splits the token axis, so it does
        # not need a real frame count
        t = self.stream_frames.get(name) or 1
        return z.unflatten(1, (t, -1))

    def _join_temporal(self, grids):
        """Frame slices of every stream, each distinct time kept once.

        Windows overlap whenever a context and a future stream share frames (le-wm's
        shift-by-one target does, at 2 of 4 frames), and those frames are one moment of
        one timestream, not two samples of it. Without times to match on, nothing is
        dropped. The first stream listed wins a tie, which matters only for an encoder
        whose latent depends on the surrounding window.
        """
        if not self.stream_times:
            return torch.cat(grids, dim=1)
        seen, keep = set(), []
        for name, grid in zip(self.streams, grids):
            for i, t in enumerate(self.stream_times[name]):
                if t in seen:
                    continue
                seen.add(t)
                keep.append(grid[:, i : i + 1])
        return torch.cat(keep, dim=1)

    def forward(self, ctx):
        grids = [self._grid(ctx, n) for n in self.streams]
        if self.concat is None:
            return sum(self._score(g) for g in grids)

        dim = self.CONCAT_DIMS[self.concat]
        kept = [i for i in range(4) if i != dim]
        shapes = {n: tuple(g.shape) for n, g in zip(self.streams, grids)}
        if len({tuple(s[i] for i in kept) for s in shapes.values()}) != 1:
            axes = "(B, T, S, D)"
            raise ValueError(
                f"concat '{self.concat}' joins streams along {axes} axis {dim}, so their "
                f"other axes must match; got {shapes}"
            )
        if self.concat == "temporal":
            return self._score(self._join_temporal(grids))
        return self._score(torch.cat(grids, dim=dim))


def build_loss_term(term, default_streams, stream_dims=None, stream_frames=None,
                    stream_times=None, modules=None):
    """One `losses:` entry -> loss module. `default_streams` is the default set a prediction
    term covers when it names no `stream`/`streams` (all predict streams marked
    `type: flow`/`prediction`). `stream_dims` maps stream name -> latent dim, which every
    sigreg term needs to size its head; `stream_frames` maps stream name -> frame count
    (its `index` length), needed when a sigreg term sets a non-pooled `statistic` or any
    `concat`; `stream_times` maps stream name -> per-frame times, letting `concat:
    temporal` tell which frames of two streams are the same moment. `modules` is the
    algorithm's `encoders:` ModuleDict, so an `ae_recon` (or raw-space `integration`) term
    can share the codec's decoder rather than build a private copy.
    """
    kind = term["type"]
    weight = float(term.get("weight", 1.0))
    if kind in ("flow", "prediction"):
        streams = term.get("streams")
        if streams is None:
            streams = [term["stream"]] if term.get("stream") is not None else list(default_streams)
        return PredictionLoss(streams, weight, label=kind)
    if kind in ("ae_recon", "integration"):
        decoder = term.get("decoder")
        if kind == "ae_recon" and decoder is None:
            raise ValueError("ae_recon needs a `decoder` naming the codec's decoder")
        if decoder is not None and (modules is None or decoder not in modules):
            raise ValueError(
                f"{kind} names decoder '{decoder}', which is not an `encoders:` entry"
            )
        module = modules[decoder] if decoder is not None else None
        if kind == "ae_recon":
            return AEReconLoss(term["stream"], module, weight, norm=term.get("norm", "l1"))
        return IntegrationLoss(term["stream"], weight, decoder=module,
                               norm=term.get("norm", "mse"),
                               units=term.get("units", "normalized"))
    if kind == "sigreg":
        return SIGRegLoss(
            list(term["streams"]), weight,
            knots=int(term.get("knots", 17)), num_proj=int(term.get("num_proj", 1024)),
            proj_dim=term.get("proj_dim"), stream_dims=stream_dims,
            statistic=term.get("statistic", "pooled"), stream_frames=stream_frames,
            concat=term.get("concat"), stream_times=stream_times,
        )
    raise ValueError(f"unknown loss type '{kind}'")
