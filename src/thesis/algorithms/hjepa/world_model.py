"""H-JEPA as an algorithm of this repo: trains on any windowed dataset backend, plans through
`thesis.planning`, and serves on the robot through `PlanDriver` like any world model.

The `level1..levelN` sections use the reference's schema unchanged (configs carry over from
external/h-jepa/h_jepa/config/train), and the module tree matches the reference, so its
released checkpoints load (`load_reference_checkpoint`). `num_levels: 1` is flat LeWM.

The data contract is the usual spec. One level-1 step is `frameskip` dataset rows, and a
training sample is the level-1 stream long enough for the top level:

    conditioning.<obs>   the current frame, index "0" -- all that planning observes
    predict.<future>     the rest of the stream, raw_index "<fs>..<(L1-1)*fs>:<fs>"
    predict.<action>     every raw action under it, index "0..<L1*fs - 1>"

`L1` is fixed by the level geometry and checked at build; the error names the exact
windows to write. The model reads each field's union window as the level-1 stream.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ...utils.spec import parse_index, raw_index, spec_fields
from ..base import BaseAlgorithm
from .jepa import HJEPA, JEPA, FusedEncoder, FusionEncoder, ProjectedEncoder, ProjectedPredictor
from .modules import (
    ARPredictor,
    Embedder,
    SequenceEncoder,
    SequenceMLPEncoder,
    SIGReg,
    build_encoder,
    build_projector,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _get(cfg, key, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else default


def build_hjepa(cfg, frameskip):
    """Reference `create_world_model`: (HJEPA, {name: SIGReg})."""
    jepas = []
    for level in range(1, int(cfg.num_levels) + 1):
        lc = cfg[f"level{level}"]
        wm = lc.wm
        target_length = int(wm.history_size) + int(_get(wm, "rollout_n", 1))
        enc_cfg = dict(lc.encoder)
        enc_proj = enc_cfg.pop("projector", {"type": "identity"})
        if _get(wm, "use_proprio", False):
            pix_spec, prop_spec = enc_cfg.pop("pixel_encoder"), enc_cfg.pop("proprio_encoder")
            pix_base, hidden = build_encoder(dict(pix_spec["encoder"]),
                                             default_patch_size=cfg.patch_size,
                                             default_image_size=cfg.img_size)
            pix_out = int(pix_spec["projector"].get("output_dim", hidden))
            pixel_encoder = ProjectedEncoder(pix_base, build_projector(
                pix_spec["projector"], input_dim=hidden, output_dim=pix_out))
            prop_cfg = dict(prop_spec["encoder"])
            prop_cfg.setdefault("input_dim", int(wm.proprio_dim))
            prop_cfg.setdefault("output_dim", int(wm.proprio_emb_dim))
            prop_base, prop_hidden = build_encoder(prop_cfg)
            prop_out = int(prop_spec["projector"].get("output_dim", prop_cfg["output_dim"]))
            proprio_encoder = ProjectedEncoder(prop_base, build_projector(
                prop_spec["projector"], input_dim=prop_hidden, output_dim=prop_out))
            embed_dim = int(_get(wm, "embed_dim", hidden))
            encoder = FusionEncoder(
                pixel_encoder, proprio_encoder,
                build_projector(enc_proj, input_dim=pix_out + prop_out, output_dim=embed_dim),
                proprio_key="proprio" if level == 1 else "proprio_input",
            )
        else:
            base, hidden = build_encoder(enc_cfg, default_patch_size=cfg.patch_size,
                                         default_image_size=cfg.img_size)
            embed_dim = int(_get(wm, "embed_dim", hidden))
            projector = build_projector(enc_proj, input_dim=hidden, output_dim=embed_dim)
            if str(enc_cfg.get("type")) == "fused_vit":
                views = [f"view{i}" for i in range(int(enc_cfg.get("num_views", 1)) - 1)]
                encoder = FusedEncoder(base, projector, view_keys=views,
                                       proprio_key="proprio" if enc_cfg.get("proprio_dim") else None)
            else:
                encoder = ProjectedEncoder(base, projector)

        pred_cfg = dict(lc.predictor)
        if pred_cfg.pop("type", "ar") != "ar":
            raise ValueError("only the ARPredictor (the paper's) is ported")
        pred_proj = pred_cfg.pop("projector", {"type": "mlp"})
        predictor = ProjectedPredictor(
            ARPredictor(num_frames=int(wm.history_size), input_dim=embed_dim,
                        output_dim=hidden, **pred_cfg),
            build_projector(pred_proj, input_dim=hidden, output_dim=embed_dim),
        )

        action_encoder, queue_size = None, 0
        if _get(lc, "action_encoder") is not None:
            kw = dict(lc.action_encoder)
            queue_size = int(kw.pop("queue_size", 0))
            if kw.pop("type", "sequence") == "mlp":
                action_encoder = SequenceMLPEncoder(temporal_stride=int(_get(lc, "stride", 1)), **kw)
            else:
                action_encoder = SequenceEncoder(**kw)

        emb_kw = dict(_get(lc, "action_embed") or {})
        if emb_kw.pop("type", None) == "identity":
            action_embed = nn.Identity()
        else:
            if level == 1:
                emb_kw["input_dim"] = int(frameskip) * int(cfg.level1.wm.action_dim)
            emb_kw["emb_dim"] = embed_dim
            action_embed = Embedder(**emb_kw)

        if float(_get(lc, "idm_coeff", 0.0)) > 0 or float(_get(lc, "action_sigreg_coeff", 0.0)) > 0:
            raise ValueError("idm_coeff / action_sigreg_coeff are not ported (no paper config uses them)")

        jepas.append(JEPA(encoder=encoder, predictor=predictor, action_embed=action_embed,
                          action_encoder=action_encoder, level=level,
                          action_queue_size=queue_size,
                          temporal_stride=int(_get(lc, "stride", 1)),
                          temporal_window_size=int(_get(lc, "window_size", 1)),
                          target_length=target_length))

    sigregs = {}
    for level in range(1, int(cfg.num_levels) + 1):
        for comp, comp_cfg in _loss_components(cfg[f"level{level}"].loss).items():
            if _enabled(comp_cfg, "sigreg"):
                kw = {k: v for k, v in dict(comp_cfg["sigreg"]).items() if k not in ("enabled", "weight")}
                sigregs[_sigreg_name(comp, level)] = SIGReg(**kw)
    return HJEPA(jepas), sigregs


def _loss_components(loss_cfg):
    comps = {n: loss_cfg[n] for n in ("embed", "pixel", "proprio") if n in loss_cfg}
    return comps or {"embed": loss_cfg}


def _enabled(comp_cfg, name):
    if name not in comp_cfg:
        return name == "pred"
    return bool(comp_cfg[name].get("enabled", False))


def _weight(comp_cfg, name):
    if name not in comp_cfg:
        return 1.0
    return float(comp_cfg[name].get("weight", 1.0))


def _sigreg_name(comp, level):
    return f"sigreg_level{level}" if comp == "embed" else f"sigreg_{comp}_level{level}"


def _component_key(comp, level):
    return f"embed_{level}" if comp == "embed" else f"{comp}_embed_{level}"


def _pred_component(pred, output, level, comp):
    if comp == "embed":
        return pred
    pixel = output[f"pixel_embed_{level}"]
    dim = 2 if pixel.ndim >= 5 else -1
    size = pixel.size(dim)
    if comp == "pixel":
        return pred.narrow(dim, 0, size)
    return pred.narrow(dim, size, pred.size(dim) - size)


def required_level1_steps(cfg):
    """Level-1 steps a sample must span for every level to fill its target length."""
    def need(level):
        n = int(cfg[f"level{level}"].wm.history_size) + int(_get(cfg[f"level{level}"].wm, "rollout_n", 1))
        for upper in range(level, 1, -1):
            uc = cfg[f"level{upper}"]
            n = (n - 1) * int(_get(uc, "stride", 1)) + int(_get(uc, "window_size", 1))
        return n
    return max(need(level) for level in range(1, int(cfg.num_levels) + 1))


class HJEPAWorldModel(BaseAlgorithm):
    action_stream = None

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conditioning = cfg.conditioning
        self.predict_spec = cfg.predict
        self.frameskip = int(cfg.frameskip)
        self.action_dim = int(cfg.level1.wm.action_dim)
        self.img_size = int(cfg.img_size)
        self.num_levels = int(cfg.num_levels)
        self.sparse_level1_encode = bool(cfg.get("sparse_level1_encode", True))
        # training-only Gaussian noise on the normalized actions, so the predictor also
        # learns what actions slightly off the demonstrations do (0 = the reference)
        self.action_noise_std = float(cfg.get("action_noise_std", 0.0) or 0.0)
        self.use_proprio = bool(_get(cfg.level1.wm, "use_proprio", False))

        roles = dict(cfg.fields)
        self.field = roles["pixels"]
        self.action_field = roles["action"]
        self.view_fields = [str(f) for f in (roles.get("views") or [])]
        enc1 = dict(cfg.level1.encoder)
        self.fused = str(enc1.get("type")) == "fused_vit"
        reads_state = self.use_proprio or (self.fused and bool(enc1.get("proprio_dim")))
        self.proprio_field = roles.get("proprio") if reads_state else None
        if reads_state and not self.proprio_field:
            raise ValueError("this level-1 encoder reads proprio: set `fields.proprio`")
        if self.fused and int(enc1.get("num_views", 1)) != 1 + len(self.view_fields):
            raise ValueError(f"fused_vit num_views={enc1.get('num_views')} but fields name "
                             f"{1 + len(self.view_fields)} cameras (pixels + views)")

        self.hjepa, sigregs = build_hjepa(cfg, self.frameskip)
        self.sigregs = nn.ModuleDict(sigregs)
        self.level1_steps = required_level1_steps(cfg)
        self._check_windows()

        plan = cfg.get("plan") or {}
        rh = int(plan.get("receding_horizon", 1))
        self.chunk_len = self.execute_len = rh * self.frameskip
        self.obs_len = 1
        obs_fields = ([self.field] + self.view_fields
                      + ([self.proprio_field] if self.proprio_field else []))
        self.field_offsets = {f: [0] for f in obs_fields}

    def _rows(self, field):
        rows = set()
        for name, spec in {**self.conditioning, **self.predict_spec}.items():
            if field in spec_fields(name, spec) and "index" in spec:
                rows.update(parse_index(raw_index(spec)))
        return sorted(rows)

    def _check_windows(self):
        L1, fs = self.level1_steps, self.frameskip
        want = {self.field: list(range(0, L1 * fs, fs)),
                self.action_field: list(range(L1 * fs))}
        if self.proprio_field:
            want[self.proprio_field] = want[self.field]
        for view in self.view_fields:
            want[view] = want[self.field]
        for field, rows in want.items():
            got = self._rows(field)
            if got != rows:
                raise ValueError(
                    f"hjepa needs {L1} level-1 steps of {fs} rows: field '{field}' must load "
                    f"{len(rows)} rows {rows} but the spec loads {len(got)}: {got}. Expected spec: "
                    f"conditioning frame index '0'; future frames raw_index "
                    f"'{fs}..{(L1 - 1) * fs}:{fs}'; actions index '0..{L1 * fs - 1}'."
                )

    def optim_params(self, lr=None):
        groups = []
        for level in range(1, self.num_levels + 1):
            params = [p for p in self.hjepa.get_level(level).parameters() if p.requires_grad]
            groups.append({"params": params, "lr": float(self.cfg[f"level{level}"].lr)})
        return groups

    def preprocess_pixels(self, x):
        """(B, T, C, H, W) uint8 or [0, 1] float -> ImageNet-normalized float at img_size,
        the reference's ToImage / ToDtype / Normalize / Resize."""
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        mean = x.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = x.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        x = (x - mean) / std
        if x.shape[-2:] != (self.img_size, self.img_size):
            b, t = x.shape[:2]
            h, w = x.shape[-2:]
            short = min(h, w)
            size = (self.img_size * h // short, self.img_size * w // short)
            x = F.interpolate(x.flatten(0, 1), size=size, mode="bilinear", antialias=True,
                              align_corners=False).unflatten(0, (b, t))
        return x

    def level1_stream(self, batch):
        """Batch fields -> the reference's `*_level1` dict."""
        pixels = batch[self.field]
        B, L1 = pixels.shape[:2]
        actions = batch[self.action_field].float()
        if self.training and self.action_noise_std > 0:
            actions = actions + self.action_noise_std * torch.randn_like(actions)
        info = {
            "pixels_level1": self.preprocess_pixels(pixels),
            "action_level1": actions.reshape(B, L1, self.frameskip * actions.shape[-1]),
        }
        for i, view in enumerate(self.view_fields):
            info[f"view{i}_level1"] = self.preprocess_pixels(batch[view])
        if self.proprio_field:
            info["proprio_level1"] = batch[self.proprio_field].float()
        return info

    def forward_levels(self, info):
        """Reference `hjepa_forward` on a `*_level1` dict: per-level predictions, losses
        and diagnostics, keyed as the reference logs them."""
        output = self.hjepa.encode_hierarchical(info, sparse_level1_encode=self.sparse_level1_encode)
        total = None
        for level in range(1, self.num_levels + 1):
            sfx = f"_level{level}"
            lc = self.cfg[f"level{level}"]
            hs = int(lc.wm.history_size)
            rollout_n = int(_get(lc.wm, "rollout_n", 1))
            rw = float(_get(lc.wm, "rollout_loss_weight", 1.0))
            jepa = self.hjepa.get_level(level)
            emb, act_emb = output[f"embed_{level}"], output[f"action_{level}"]

            nsteps = _get(lc.wm, "nsteps")
            if nsteps is not None:
                passes = jepa.parallel_unroll(emb, act_emb, int(nsteps))
                pred_emb, tgt_emb = passes.mean(dim=0)[:, 1:], emb[:, 1:]
            elif rollout_n == 1:
                teacher = jepa.predict(emb[:, :hs], act_emb[:, :hs])
                pred_emb, tgt_emb = teacher, emb[:, 1: hs + 1]
            else:
                teacher = jepa.predict(emb[:, :hs], act_emb[:, :hs])
                roll = emb[:, :hs]
                for step in range(rollout_n):
                    nxt = jepa.predict(roll[:, -hs:], act_emb[:, step: step + hs])[:, -1:]
                    roll = torch.cat([roll, nxt], dim=1)
                pred_emb, tgt_emb = roll[:, hs:], emb[:, hs: hs + rollout_n]

            with torch.no_grad():
                output[f"mse_loss{sfx}"] = F.mse_loss(pred_emb, tgt_emb)
                output[f"dim_std_loss{sfx}"] = tgt_emb.std()

            level_loss = None
            for comp, comp_cfg in _loss_components(lc.loss).items():
                comp_emb = output[_component_key(comp, level)]
                tgt = comp_emb.detach() if _get(lc.wm, "detach_pred_target") else comp_emb
                comp_loss = None
                tag = "" if comp == "embed" else f"_{comp}"
                if _enabled(comp_cfg, "pred") and nsteps is not None:
                    pc = _pred_component(passes, output, level, comp)[:, :, 1:]
                    pred_loss = F.mse_loss(pc, tgt[:, 1:].unsqueeze(0).expand_as(pc))
                    output[f"pred_loss{tag}{sfx}"] = pred_loss
                    comp_loss = _weight(comp_cfg, "pred") * pred_loss
                elif _enabled(comp_cfg, "pred"):
                    tf = F.mse_loss(_pred_component(teacher, output, level, comp), tgt[:, 1: hs + 1])
                    output[f"teacher_forcing_loss{tag}{sfx}"] = tf
                    if rollout_n == 1:
                        pred_loss = tf
                    else:
                        rl = F.mse_loss(_pred_component(pred_emb, output, level, comp),
                                        tgt[:, hs: hs + rollout_n])
                        output[f"rollout_loss{tag}{sfx}"] = rl
                        pred_loss = tf + rw * rl if hs > 1 else rw * rl
                    output[f"pred_loss{tag}{sfx}"] = pred_loss
                    comp_loss = _weight(comp_cfg, "pred") * pred_loss
                if _enabled(comp_cfg, "sigreg"):
                    sig = self.sigregs[_sigreg_name(comp, level)](comp_emb.flatten(2).transpose(0, 1))
                    output[f"sigreg_loss{tag}{sfx}"] = sig
                    weighted = _weight(comp_cfg, "sigreg") * sig
                    comp_loss = weighted if comp_loss is None else comp_loss + weighted
                if comp_loss is None:
                    continue
                output[f"{comp}_loss{sfx}"] = comp_loss
                level_loss = comp_loss if level_loss is None else level_loss + comp_loss
            if level_loss is None:
                continue
            output[f"loss{sfx}"] = level_loss
            total = level_loss if total is None else total + level_loss
        output["loss"] = total
        return output

    def loss(self, batch):
        output = self.forward_levels(self.level1_stream(batch))
        logs = {"loss": output["loss"]}
        for k, v in output.items():
            if k != "loss" and "loss" in k and torch.is_tensor(v) and v.ndim == 0:
                logs[f"loss/{k}"] = v.detach()
        for level in range(1, self.num_levels + 1):
            emb = output[f"embed_{level}"].detach().float()
            total = emb.reshape(-1, emb.shape[-1]).var(dim=0).mean().clamp_min(1e-12)
            logs[f"var/embed_{level}/temporal"] = emb.var(dim=1).mean() / total
        return logs

    @torch.no_grad()
    def encode_levels(self, info, key="pixels"):
        """Reference `HierarchicalSolver._encode_all_levels`: one frame (B, 1, ...) or
        (B, S, 1, ...) through every level -> {"embed_<level>": (B, 1, D)}."""
        pixels = info[key]
        if pixels.ndim >= 6:
            pixels = pixels[:, 0]
        device = next(self.parameters()).device
        enc = {"pixels": pixels.to(device)}
        prefix = "goal_" if key == "goal" else ""
        prop_key = f"{prefix}proprio"
        if prop_key in info:
            p = info[prop_key].to(device)
            enc["proprio"] = p[:, 0] if p.ndim >= 4 else p
        for i in range(len(self.view_fields)):
            v = info[f"{prefix}view{i}"].to(device)
            enc[f"view{i}"] = v[:, 0] if v.ndim >= 6 else v
        out = {}
        for level in range(1, self.num_levels + 1):
            jepa = self.hjepa.get_level(level)
            in_key = "pixels" if level == 1 else self.hjepa.level_input_key(level)
            if level > 1:
                x, w = enc[in_key], jepa.temporal_window_size
                if x.shape[1] < w:
                    x = torch.cat([x[:, :1].expand(-1, w - x.shape[1], *x.shape[2:]), x], dim=1)
                enc[in_key] = x[:, -w:]
                if self.hjepa.level_uses_proprio(level):
                    enc[jepa.encoder.proprio_key] = enc[f"proprio_embed_{level - 1}"]
            lo = jepa.encode(enc, key=in_key, chunk_temporal_inputs=level > 1)
            for name in ("embed", "pixel_embed", "proprio_embed"):
                if f"{name}_0" in lo:
                    enc[f"{name}_{level}"] = lo[f"{name}_0"]
            out[f"embed_{level}"] = lo["embed_0"][:, -1:].detach()
        return out

    def plan_config(self, overrides=None):
        """The model's `plan:` section with an eval's overrides merged on top."""
        from omegaconf import OmegaConf

        base = OmegaConf.to_container(self.cfg.get("plan") or OmegaConf.create({}), resolve=True)
        extra = overrides or {}
        if not isinstance(extra, dict):
            extra = OmegaConf.to_container(extra, resolve=True)
        merged = _deep_merge(base, extra)
        for level, spec in (extra.get("solvers") or {}).items():
            old = (base.get("solvers") or {}).get(level) or {}
            if spec and spec.get("name", old.get("name")) != old.get("name"):
                merged["solvers"][level] = dict(spec)
        return merged

    def make_planner(self, overrides=None, n_envs=1):
        """An `MPCPlanner` over this model: one solver per level (any registered kind), the
        hierarchy wiring of the reference, and its horizon schedules."""
        from ...planning import (
            HierarchicalPlanConfig,
            MPCPlanner,
            PlanConfig,
            build_hierarchical_solver,
        )

        plan = self.plan_config(overrides)
        device = str(next(self.parameters()).device)
        levels, specs = [], {}
        for level in range(1, self.num_levels + 1):
            lc = dict((plan.get("levels") or {}).get(f"level{level}") or {})
            if level == 1:
                lc.setdefault("action_block", self.frameskip)
                if int(lc["action_block"]) != self.frameskip:
                    raise ValueError(f"level1 action_block must equal frameskip={self.frameskip}")
            else:
                lc.setdefault("action_dim", _latent_action_dim(self.hjepa.get_level(level)))
            lc.setdefault("receding_horizon", plan.get("receding_horizon", 1))
            lc.setdefault("horizon", 1)
            levels.append(PlanConfig.from_dict(lc))
            spec = dict((plan.get("solvers") or {}).get(f"level{level}") or {"name": "gd", "n_steps": 30})
            spec.setdefault("device", device)
            specs[f"level{level}"] = spec
        config = HierarchicalPlanConfig(receding_horizon=int(plan.get("receding_horizon", 1)),
                                        levels=tuple(levels))
        solver = build_hierarchical_solver([self.hjepa.get_level(i) for i in range(1, self.num_levels + 1)],
                                           specs, self.encode_levels)
        return MPCPlanner(solver, config, action_dim=self.action_dim, n_envs=n_envs,
                          eval_budget=plan.get("eval_budget"))

    def plan_inputs(self, obs, goal):
        """Driver obs (fields, last row = now) + goal ({field: (B, 1, ...)}) -> solver info."""
        info = {"pixels": self.preprocess_pixels(obs[self.field][:, -1:]),
                "goal": self.preprocess_pixels(goal[self.field][:, -1:])}
        for i, view in enumerate(self.view_fields):
            info[f"view{i}"] = self.preprocess_pixels(obs[view][:, -1:])
            info[f"goal_view{i}"] = self.preprocess_pixels(goal[view][:, -1:])
        if self.proprio_field:
            info["proprio"] = obs[self.proprio_field][:, -1:].float()
            info["goal_proprio"] = goal[self.proprio_field][:, -1:].float()
        B = info["pixels"].shape[0]
        for k in [k for k in info if k.startswith("goal")]:
            if info[k].shape[0] != B:
                info[k] = info[k].expand(B, *info[k].shape[1:])
        return info

    def plan(self, planner, obs, goal, steps_taken=None):
        """One MPC step: -> (B, execute_len, action_dim) normalized primitive actions."""
        self.eval()
        info = self.plan_inputs(obs, goal)
        planner.resize(info["pixels"].shape[0])
        return planner.plan(info, steps_taken=steps_taken)

    @torch.no_grad()
    def goal_distance(self, obs, goal, actions):
        """Level-1 rollout of primitive `actions` (B, T * frameskip, A) from `obs`: the MSE
        between the final predicted latent and the goal's. How far a given action sequence
        gets toward the goal in the model's own judgement."""
        self.eval()
        info = self.plan_inputs(obs, goal)
        emb = self.encode_levels(info, "pixels")["embed_1"]
        goal_emb = self.encode_levels(info, "goal")["embed_1"]
        B = actions.shape[0]
        blocks = actions.reshape(B, -1, self.frameskip * self.action_dim).unsqueeze(1)
        out = self.hjepa.get_level(1).rollout({"embed_0": emb.unsqueeze(1)}, blocks)
        final = out["predicted_embed_0"][:, 0, -1]
        return ((final - goal_emb[:, 0]) ** 2).mean(dim=-1)

    def load_reference_checkpoint(self, path, strict=True):
        """Load an external/h-jepa `*_object.ckpt` (a pickled HJEPA) without executing it."""
        from ...utils.safe_ckpt import load_object_state_dict

        state, _ = load_object_state_dict(path)
        return self.hjepa.load_state_dict(state, strict=strict)

    def summary(self):
        return {**super().summary(), "num_levels": self.num_levels,
                "level1_steps": self.level1_steps, "frameskip": self.frameskip}


def _latent_action_dim(jepa):
    enc = jepa.action_encoder
    return int(enc.output_dim)


def _deep_merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
