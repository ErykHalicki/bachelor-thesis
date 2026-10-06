"""Pixel decoders on a frozen world model: what its latents keep, and what it gets right.

The config is a finished WAM's algorithm config -- same `model:`, `encoders:`, spec, stream
names -- with the decoders added. A probe config does not restate any of that: it composes
the WAM's yaml, and `experiment.encoder_init=<run id>` then supersedes it with the config
that run stored and refills the whole model (encoders AND trunk) by parameter name, so the
probe reads exactly what that run learned. A `decode:` section says which streams get a
decoder:

    decode:
      scene: { stream: scene_future, decoder: decode_scene }
    losses:
      - {type: prediction, stream: scene, weight: 1.0}

One decoder answers two questions, which is the point of decoding the predictor rather than
just the encoder. At eval each decoded stream renders three columns side by side:

    frame               the future frame the arm actually reached
    decoded latent      that frame ENCODED, then decoded -- the representation's own ceiling
    decoded prediction  the trunk's predicted future latent, decoded

The gap between columns 1 and 2 is what the encoder threw away; the gap between 2 and 3 is
what the predictor got wrong. Reading a prediction against the real frame alone confuses
the two, which is why the middle column exists.

The decoder trains on the ENCODED latent, never on the prediction: it has to be a fixed
readout of the latent space, or it would learn to paper over the predictor's errors and
column 3 would flatter the model. Training therefore never runs the trunk at all -- the
whole world model is frozen, and only the eval pass integrates it.
"""

import torch

from ..utils.spec import parse_index, raw_index, spec_fields
from .generic_vit_predictor import GenericViTPredictor
from .latent_decoder import PixelProbe, to_uint8

__all__ = ["WAMPixelDecoder"]


class WAMPixelDecoder(PixelProbe, GenericViTPredictor):
    """A GenericViTPredictor kept frozen, plus one pixel decoder per `decode:` entry."""

    _frozen_trunk = False

    def __init__(self, cfg):
        super().__init__(cfg)
        decode = cfg.get("decode")
        if not decode:
            raise ValueError(
                "a wam_decoder run needs a `decode:` section naming the streams to decode "
                "({name: {stream: <spec stream>, decoder: <encoders entry>}})"
            )
        self.decode_spec = decode
        streams = {**cfg.conditioning, **cfg.predict}
        # held by name, not as a second module reference, so this run's weights line
        # up key for key with the WAM run it was seeded from
        self._decoder_of = {}
        for name, entry in decode.items():
            stream, dec = entry.get("stream"), entry.get("decoder")
            if stream not in streams:
                raise ValueError(
                    f"decode entry '{name}' reads stream '{stream}', which is in neither "
                    f"`conditioning:` nor `predict:`; available: {sorted(streams)}"
                )
            if dec not in self.encoders:
                raise ValueError(f"decode entry '{name}' names unknown decoder '{dec}'")
            self._check_decoder(name, dec, stream, streams[stream])
            self._decoder_of[name] = dec

        # the loss terms were built from the predict spec, but a probe optimizes
        # decoders: a term naming a WAM stream would train the model this run holds still
        for term in self.loss_terms:
            unknown = [s for s in getattr(term, "streams", []) if s not in decode]
            if unknown:
                raise ValueError(
                    f"loss term '{term.label}' names {unknown}, which are not `decode:` "
                    f"entries; a wam_decoder run's `losses:` may only name decoders "
                    f"(the world model is frozen)"
                )

        # freeze the world model, decoders excepted, overriding the encoders' own
        # `frozen:` -- the spec here is the probed run's, where they were training.
        # Dropout is on in a WAM config, so the trunk must stay in EVAL mode too.
        if cfg.get("freeze_world_model", True):
            decoders = set(self._decoder_of.values())
            keep = {id(p) for name in decoders for p in self.encoders[name].parameters()}
            for param in self.parameters():
                if id(param) not in keep:
                    param.requires_grad_(False)
            for name, encoder in self.encoders.items():
                if name not in decoders:
                    encoder.frozen = True
                    encoder.eval()
            self.predictor.eval()
            self._frozen_trunk = True

    def train(self, mode=True):
        super().train(mode)
        if self._frozen_trunk:
            self.predictor.eval()
        return self

    def raw_pixel_fields(self):
        """The base rule (fields some non-cacheable entry reads) misses this probe's
        targets: `decode:` streams are themselves cacheable conditioning/predict entries,
        but their LOSS compares against the raw frames (`_frames`), so their fields must
        keep arriving as pixels alongside the cache keys (keys+pixels mode).
        """
        needed = super().raw_pixel_fields()
        streams = {**dict(self.conditioning), **dict(self.predict_spec)}
        for entry in self.decode_spec.values():
            stream = entry["stream"]
            needed.update(spec_fields(stream, streams[stream]))
        return needed

    def _decode(self, name, latent):
        return self.encoders[self._decoder_of[name]](latent)

    def _build_context(self, batch):
        """Encoder latents, decoded frames, and the frames themselves. The trunk is not run:
        the decoders train on encoded latents, so a training step is encode + decode.
        """
        clean, sources, cond, hidden = self._condition(batch)
        latent = {**clean, **sources, **cond, **hidden}
        latent.update({
            name: self._encode(name, spec, batch) for name, spec in self.predict_spec.items()
        })
        pred, target = {}, {}
        for name, entry in self.decode_spec.items():
            stream = entry["stream"]
            # float32 on both sides: under bf16 autocast an MSE over pixel-scale values
            # loses most of its resolution
            out = self._decode(name, latent[stream]).float()
            frames = self._frames(stream, batch, out.shape[-2:])
            self._check_shape(name, out, frames)
            pred[name], target[name] = out, frames
        return {"latent": latent, "pred": pred, "target": target}

    @torch.no_grad()
    def reconstruct(self, batch):
        """{decode entry: {frame, decoded latent, decoded prediction}}, uint8 (B, steps, 3, H, W).

        The prediction column comes from one full rollout of the world model on this batch
        (`GenericViTPredictor.predict`), so it is the same integration eval and the robot
        run on -- noise draw included, which is why the eval seeds it.
        """
        was_training = self.training
        self.eval()
        try:
            clean, sources, cond, hidden = self._condition(batch)
            latent = {**clean, **sources, **cond, **hidden}
            latent.update({
                name: self._encode(name, spec, batch)
                for name, spec in self.predict_spec.items()
            })
            # A2A streams integrate from their configured seeds: probing from Gaussian
            # noise would test a start the model never trained from
            seeds = self.build_flow_seeds(batch, latent, training=False)
            rolled = self.predictor.rollout(
                clean, sources, cond=cond, num_steps=self.num_flow_steps,
                cfg_scale=self.cfg_scale, cfg_drop=self.cfg_drop, x0=seeds,
            )
            out = {}
            for name, entry in self.decode_spec.items():
                stream = entry["stream"]
                decoded = self._decode(name, latent[stream])
                columns = {
                    "frame": to_uint8(self._frames(stream, batch, decoded.shape[-2:])),
                    "decoded latent": to_uint8(decoded),
                }
                if stream in rolled:
                    columns["decoded prediction"] = to_uint8(self._decode(name, rolled[stream]))
                out[name] = columns
        finally:
            self.train(was_training)
        return out

    def _ar_context_map(self):
        """predict stream -> the conditioning stream its predictions can refill.

        Video streams pair by shared source field (scene_future -> scene_context, last
        two predicted steps become the new 2-step context); the dense state stream pairs
        with the state context, whose encoder re-embeds the last predicted state row.
        """
        pairs = {}
        for pname, pspec in self.predict_spec.items():
            if pname == self.action_stream:
                continue
            field = pspec.get("from", pname)
            for cname, cspec in self.conditioning.items():
                if cspec.get("role") == "action" or cspec.get("via", "tokens") != "tokens":
                    continue
                if cspec.get("from", cname) == field:
                    pairs[pname] = cname
        return pairs

    @torch.no_grad()
    def reconstruct_ar(self, batches):
        """`reconstruct`, autoregressively over consecutive windows: window k>0 gets its
        video/state context from window k-1's PREDICTED latents (state re-embedded through
        its own encoder), only actions and the goal-free conditioning come from the real
        batch. Returns {decode entry: {frame, decoded latent, decoded prediction}} with
        len(batches) * steps columns per row -- the compounding-error picture.

        `batches` are aligned windows one chunk apart (the eval samples them); the frame
        and decoded-latent columns come from each window's real pixels, so the ceiling
        stays honest while the prediction column drifts.
        """
        was_training = self.training
        self.eval()
        pairs = self._ar_context_map()
        unfed = sorted(
            c for c, cs in self.conditioning.items()
            if cs.get("via", "tokens") == "tokens" and cs.get("role") != "action"
            and c not in pairs.values() and c != "action_clean"
        )
        if unfed:
            raise ValueError(
                f"AR rollout cannot refill conditioning {unfed} from predictions; "
                f"windows after the first would silently read real data there"
            )
        try:
            carried = None                      # {context stream: latent} from the last window
            frames, ceilings, preds = {}, {}, {}
            for batch in batches:
                clean, sources, cond, hidden = self._condition(batch)
                if carried:
                    clean = {**clean, **carried}
                latent = {**clean, **sources, **cond, **hidden}
                latent.update({
                    name: self._encode(name, spec, batch)
                    for name, spec in self.predict_spec.items()
                })
                seeds = self.build_flow_seeds(batch, latent, training=False)
                rolled = self.predictor.rollout(
                    clean, sources, cond=cond, num_steps=self.num_flow_steps,
                    cfg_scale=self.cfg_scale, cfg_drop=self.cfg_drop, x0=seeds,
                )
                carried = {}
                for pname, cname in pairs.items():
                    cspec = self.conditioning[cname]
                    steps = len(parse_index(cspec["index"]))
                    z = rolled[pname]
                    if cspec.get("encoder") and self.predict_spec[pname].get("encoder") is None:
                        # dense raw-valued prediction (state): re-embed its tail rows
                        carried[cname] = self.run_encoder(cspec["encoder"], z[:, -steps:])
                    else:
                        carried[cname] = z[:, -steps:]
                for name, entry in self.decode_spec.items():
                    stream = entry["stream"]
                    decoded = self._decode(name, latent[stream])
                    frames.setdefault(name, []).append(
                        self._frames(stream, batch, decoded.shape[-2:]))
                    ceilings.setdefault(name, []).append(decoded)
                    if stream in rolled:
                        preds.setdefault(name, []).append(self._decode(name, rolled[stream]))
            out = {}
            for name in frames:
                columns = {
                    "frame": to_uint8(torch.cat(frames[name], dim=1)),
                    "decoded latent": to_uint8(torch.cat(ceilings[name], dim=1)),
                }
                if name in preds:
                    columns["decoded prediction"] = to_uint8(torch.cat(preds[name], dim=1))
                out[name] = columns
        finally:
            self.train(was_training)
        return out

    def ar_window_stride(self):
        """Raw rows between consecutive AR windows: the latent-future horizon."""
        strides = []
        for entry in self.decode_spec.values():
            spec = self.predict_spec[entry["stream"]]
            strides.append(max(parse_index(raw_index(spec))))
        assert strides and len(set(strides)) == 1, (
            f"decoded streams disagree on the future horizon ({strides}); AR windows "
            f"need one stride"
        )
        return strides[0]

    @torch.no_grad()
    def prediction_errors(self, batch):
        """Pixel MSE of each decoded rollout against the future frames it predicts, in the
        same [0, 1] units as the decoder's own loss terms -- the quantitative form of the
        panel's third column. Seeding is the caller's job, as with `reconstruct`."""
        was_training = self.training
        self.eval()
        try:
            clean, sources, cond, hidden = self._condition(batch)
            latent = {**clean, **sources, **cond, **hidden}
            latent.update({
                name: self._encode(name, spec, batch)
                for name, spec in self.predict_spec.items()
            })
            seeds = self.build_flow_seeds(batch, latent, training=False)
            rolled = self.predictor.rollout(
                clean, sources, cond=cond, num_steps=self.num_flow_steps,
                cfg_scale=self.cfg_scale, cfg_drop=self.cfg_drop, x0=seeds,
            )
            out = {}
            for name, entry in self.decode_spec.items():
                stream = entry["stream"]
                if stream not in rolled:
                    continue
                pred = self._decode(name, rolled[stream]).float()
                frames = self._frames(stream, batch, pred.shape[-2:]).float()
                out[name] = torch.mean((pred - frames) ** 2)
        finally:
            self.train(was_training)
        return out

    def summary(self):
        return {
            **super().summary(),
            "decode/streams": ", ".join(
                f"{n}<-{e['stream']}" for n, e in self.decode_spec.items()
            ),
        }
