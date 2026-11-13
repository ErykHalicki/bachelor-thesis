"""One H-JEPA level (`JEPA`) and the level stack (`HJEPA`), ported from external/h-jepa
(h_jepa/models/jepa.py, hjepa.py) with the same module tree and the same numerics.

Every tensor dict here follows the reference's key convention: an `encode` call writes its
outputs as `embed_0` / `pixel_embed_0` / `proprio_embed_0` / `action_0`, and the stack
renames them `embed_<level>` etc. as it lifts a level-1 stream through every level.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


def _run_encoder(encoder, x):
    if x.ndim >= 4:
        return encoder(x, interpolate_pos_encoding=True)
    return encoder(x)


def _split_encoder_output(output, allow_cls_token=True):
    hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
    if hidden.ndim == 2:
        return hidden
    if hidden.ndim == 3 and allow_cls_token:
        return hidden[:, 0]
    raise ValueError(
        f"encoder output must collapse to (B*T, D) here, got shape {tuple(hidden.shape)}"
    )


class ProjectedEncoder(nn.Module):
    def __init__(self, encoder, projector=None):
        super().__init__()
        self.encoder = encoder
        self.projector = projector or nn.Identity()

    def forward(self, x, *, allow_cls_token=True):
        hidden = _split_encoder_output(_run_encoder(self.encoder, x), allow_cls_token)
        return self.projector(hidden)

    def encode_info(self, info, *, key, allow_cls_token=True):
        x = rearrange(info[key].float(), "b t ... -> (b t) ...")
        return {"embed": self(x, allow_cls_token=allow_cls_token)}


class FusedEncoder(nn.Module):
    """A FusedViT over the main camera, `view_keys` cameras and an optional proprio stream,
    projected like ProjectedEncoder. Planning reads only its fused CLS embedding."""

    def __init__(self, encoder, projector=None, view_keys=(), proprio_key=None):
        super().__init__()
        self.encoder = encoder
        self.projector = projector or nn.Identity()
        self.view_keys = list(view_keys)
        self.proprio_key = proprio_key

    def encode_info(self, info, *, key, allow_cls_token=True):
        missing = [k for k in (*self.view_keys, self.proprio_key) if k and k not in info]
        if missing:
            raise KeyError(f"FusedEncoder needs {missing} beside '{key}'")
        views = [rearrange(info[k].float(), "b t ... -> (b t) ...") for k in (key, *self.view_keys)]
        proprio = (rearrange(info[self.proprio_key].float(), "b t ... -> (b t) ...")
                   if self.proprio_key else None)
        return {"embed": self.projector(self.encoder(torch.stack(views, dim=1), proprio))}


class FusionEncoder(nn.Module):
    """Pixel and proprio streams encoded apart and fused by one projector."""

    def __init__(self, pixel_encoder, proprio_encoder, projector=None, proprio_key="proprio"):
        super().__init__()
        self.pixel_encoder = pixel_encoder
        self.proprio_encoder = proprio_encoder
        self.projector = projector or nn.Identity()
        self.proprio_key = proprio_key

    def encode_info(self, info, *, key, allow_cls_token=True):
        if self.proprio_key not in info:
            raise KeyError(f"FusionEncoder needs '{self.proprio_key}' in its input")
        pixel = self.pixel_encoder.encode_info(info, key=key,
                                               allow_cls_token=allow_cls_token)["embed"]
        proprio = self.proprio_encoder.encode_info(info, key=self.proprio_key,
                                                   allow_cls_token=allow_cls_token)["embed"]
        embed = self.projector(torch.cat([pixel, proprio], dim=-1))
        return {"embed": embed, "pixel_embed": pixel, "proprio_embed": proprio}


class ProjectedPredictor(nn.Module):
    def __init__(self, predictor, projector=None):
        super().__init__()
        self.predictor = predictor
        self.projector = projector or nn.Identity()

    def forward(self, emb, act_emb):
        preds = self.predictor(emb, act_emb)
        preds = self.projector(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) ... -> b t ...", b=emb.size(0))


class JEPA(nn.Module):
    """One level: encoder, action encoder (macro-actions, level 2+), action embedding and
    predictor, plus the geometry (`temporal_stride`, `temporal_window_size`) with which it
    chunks the level below."""

    def __init__(self, encoder, predictor, action_embed, action_encoder=None, level=1,
                 action_queue_size=0, temporal_stride=1, temporal_window_size=1,
                 target_length=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_embed = action_embed
        self.action_encoder = action_encoder or nn.Identity()
        self.level = level
        self.temporal_stride = int(temporal_stride)
        self.temporal_window_size = int(temporal_window_size)
        self.target_length = None if target_length is None else int(target_length)
        self.configure_action_queue(action_queue_size)

    def _state_windows(self, x, starts, window_size):
        if window_size == 1:
            return x.index_select(1, starts)
        indices = starts[:, None] + torch.arange(window_size, device=x.device)[None, :]
        chunks = x.index_select(1, indices.reshape(-1))
        return chunks.reshape(x.size(0), starts.numel(), window_size, *x.shape[2:])

    def _action_windows(self, x, starts, state_window_size, stride):
        """The first action of chunk T aligns with the last state of chunk T; the end is
        zero-padded and the returned mask is True there."""
        seq_len = x.size(1)
        indices = (starts + state_window_size - 1)[:, None] + torch.arange(stride, device=x.device)[None, :]
        valid = indices < seq_len
        indices = indices.clamp(max=seq_len - 1)
        chunks = x.index_select(1, indices.reshape(-1))
        chunks = chunks.reshape(x.size(0), starts.numel(), stride, *x.shape[2:])
        valid_b = valid.unsqueeze(0)
        while valid_b.ndim < chunks.ndim:
            valid_b = valid_b.unsqueeze(-1)
        mask = (~valid).unsqueeze(0).expand(x.size(0), -1, -1)
        return chunks * valid_b.to(dtype=chunks.dtype), mask

    def _chunk_temporal_info(self, info, key, stride, window_size):
        stride, window_size = int(stride), int(window_size)
        if stride < 1 or window_size < 1:
            raise ValueError(f"stride and window_size must be >= 1, got {stride}, {window_size}")
        ref = info[key]
        if not torch.is_tensor(ref) or ref.ndim < 3:
            raise ValueError(f"'{key}' must be a (B, T, ...) tensor")
        seq_len = ref.size(1)
        if seq_len < window_size:
            raise ValueError(f"sequence length {seq_len} is shorter than window_size={window_size}")
        starts = torch.arange(0, seq_len - window_size + 1, stride, device=ref.device)
        chunked = dict(info)
        for name, value in info.items():
            if not torch.is_tensor(value) or value.ndim < 3:
                continue
            if value.size(0) != ref.size(0) or value.size(1) != seq_len:
                continue
            if name == "action":
                chunked[name], chunked["action_mask"] = self._action_windows(
                    value, starts, window_size, stride)
            elif not name.startswith("action"):
                chunked[name] = self._state_windows(value, starts, window_size)
        return chunked

    def configure_action_queue(self, queue_size=0):
        """The latent macro-actions seen in training, kept to seed a planner's prior."""
        queue_size = int(queue_size)
        self.action_queue_size = queue_size
        if queue_size == 0:
            self.latent_action_queue = None
            return
        if "latent_action_queue" not in self._buffers:
            self.register_buffer("latent_action_queue", torch.empty(0, 0), persistent=True)
        elif self.latent_action_queue is None:
            self._buffers["latent_action_queue"] = torch.empty(0, 0)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "latent_action_queue"
        if key in state_dict and "latent_action_queue" in self._buffers:
            saved = state_dict[key]
            self._buffers["latent_action_queue"] = saved.new_empty(saved.shape)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _update_latent_action_queue(self, latent_actions):
        if not self.training or self.action_queue_size <= 0:
            return
        samples = latent_actions[:, 0].detach()
        queue = self.latent_action_queue
        if queue.numel() == 0:
            updated = samples
        else:
            updated = torch.cat([queue.to(device=samples.device, dtype=samples.dtype), samples])
        self._buffers["latent_action_queue"] = updated[-self.action_queue_size:]

    def get_latent_action_queue(self):
        queue = getattr(self, "latent_action_queue", None)
        if not torch.is_tensor(queue) or queue.numel() == 0:
            return None
        return queue

    def encode(self, info, key="pixels", temporal_stride=None, chunk_temporal_inputs=False):
        """Encode states (and actions, when `info` has them). Writes `embed_0` (B, T, D) and
        `action_0` (B, T, A) into `info` and returns it."""
        chunked_state_input = False
        if chunk_temporal_inputs:
            if temporal_stride is None:
                temporal_stride = self.temporal_stride
            chunked_state_input = self.temporal_window_size > 1
            info = self._chunk_temporal_info(info, key, temporal_stride, self.temporal_window_size)

        B = info[key].size(0)
        outputs = self.encoder.encode_info(info, key=key, allow_cls_token=not chunked_state_input)
        for name, value in outputs.items():
            info[f"{name}_0"] = rearrange(value, "(b t) ... -> b t ...", b=B)

        if "action" in info:
            mask = info.get("action_mask")
            latent = (self.action_encoder(info["action"]) if mask is None
                      else self.action_encoder(info["action"], mask))
            self._update_latent_action_queue(latent)
            info["action_0"] = self.action_embed(latent)
        return info

    def predict(self, emb, act_emb):
        return self.predictor(emb, act_emb)

    def parallel_unroll(self, emb, act_emb, nsteps):
        T = emb.size(1) - 1
        pred_input = emb[:, :T]
        passes = []
        for _ in range(nsteps):
            predicted = torch.cat([emb[:, :1], self.predict(pred_input, act_emb[:, :T])], dim=1)
            passes.append(predicted)
            pred_input = torch.cat([emb[:, :1], predicted[:, 1:T].detach()], dim=1)
        return torch.stack(passes)

    def rollout(self, info, action_sequence, history_size=1):
        """Roll the predictor from `info` (B, S, ...) over candidate actions (B, S, T, A);
        writes `predicted_embed_0` (B, S, T + 1, D): the context, then one prediction per
        action."""
        H = info["pixels"].size(2) if "pixels" in info else 1
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        if "embed_0" not in _init:
            with torch.no_grad():
                _init = self.encode(_init)
        emb = info["embed_0"] = _init["embed_0"].unsqueeze(1).expand(B, S, -1, -1)

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_embed(act)
            pred = self.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
            emb = torch.cat([emb, pred], dim=1)
            act = torch.cat([act, act_future[:, t: t + 1, :]], dim=1)
        act_emb = self.action_embed(act)
        pred = self.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)

        info["predicted_embed_0"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def criterion(self, info_dict):
        """MSE between the last `cost_last_n` predictions and the goal, (B, S)."""
        pred = info_dict["predicted_embed_0"][:, :, 1:, :]
        cost_last_n = max(1, int(info_dict.get("cost_last_n", 1)))
        pred = pred[:, :, -min(cost_last_n, pred.shape[-2]):, :]
        goal = info_dict["goal_embed_0"].expand_as(pred).detach()
        cost = F.mse_loss(pred, goal, reduction="none")
        weight = float(info_dict.get("intermediate_cost_weight", 1.0))
        if weight != 1.0 and cost.shape[2] > 1:
            weights = cost.new_ones(cost.shape[2])
            weights[:-1] = weight
            cost = cost * weights.view(1, 1, -1, 1)
        return cost.sum(dim=tuple(range(2, cost.ndim)))

    def get_cost(self, info_dict, action_candidates):
        device = next(self.parameters()).device
        for k in list(info_dict):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)
        if "goal_embed_0" not in info_dict:
            goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            goal["pixels"] = goal["goal"]
            for k in info_dict:
                if k.startswith("goal_"):
                    goal[k[len("goal_"):]] = goal.pop(k)
            goal.pop("action", None)
            with torch.no_grad():
                goal = self.encode(goal)
            info_dict["goal_embed_0"] = goal["embed_0"].unsqueeze(1)
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)


class HJEPA(nn.Module):
    """The level stack; `jepas[0]` is level 1."""

    def __init__(self, jepas: Sequence[nn.Module]):
        super().__init__()
        if len(jepas) == 0:
            raise ValueError("jepas must contain at least one JEPA")
        self.jepas = nn.ModuleList(jepas)

    @property
    def num_levels(self):
        return len(self.jepas)

    def get_level(self, level):
        if not 1 <= level <= self.num_levels:
            raise IndexError(f"level must be in [1, {self.num_levels}], got {level}")
        return self.jepas[level - 1]

    def __getitem__(self, idx):
        return self.jepas[idx]

    def __len__(self):
        return self.num_levels

    def level_uses_proprio(self, level):
        encoder = getattr(self.jepas[level - 1], "encoder", None)
        return hasattr(encoder, "pixel_encoder") and hasattr(encoder, "proprio_encoder")

    def level_input_key(self, level):
        return f"pixel_embed_{level - 1}" if self.level_uses_proprio(level) else f"embed_{level - 1}"

    @staticmethod
    def _collect_level_info(info, level):
        suffix = f"_level{level}"
        return {k[: -len(suffix)]: v for k, v in info.items() if k.endswith(suffix)}

    @staticmethod
    def _add_level_target_aliases(info, level):
        for k, v in list(info.items()):
            if k.startswith("action") or not torch.is_tensor(v) or v.ndim < 3:
                continue
            info[f"{k}_level{level}"] = v

    @staticmethod
    def _draw_crop_start(seq_len, target_len, device):
        return torch.randint(0, seq_len - target_len + 1, (), device=device).item()

    def _random_crop_level_outputs(self, output, level, target_len, extra_keys=None, start=None):
        target_len = int(target_len)
        if target_len <= 0:
            return
        level_keys = {f"embed_{level}", f"pixel_embed_{level}", f"proprio_embed_{level}",
                      f"action_{level}"}
        ref_keys = [f"embed_{level}", f"action_{level}"]
        if extra_keys:
            level_keys.update(extra_keys)
            ref_keys.extend(sorted(extra_keys))
        ref = next((output[k] for k in ref_keys
                    if torch.is_tensor(output.get(k)) and output[k].ndim >= 3), None)
        if ref is None or ref.size(1) <= target_len:
            return
        seq_len = ref.size(1)
        if start is None:
            start = self._draw_crop_start(seq_len, target_len, ref.device)
        indices = torch.arange(start, start + target_len, device=ref.device)
        suffix = f"_level{level}"
        for k, v in list(output.items()):
            if not torch.is_tensor(v) or v.ndim < 3 or v.size(1) != seq_len:
                continue
            if k in level_keys or k.endswith(suffix):
                output[k] = v.index_select(1, indices)

    def level_geometry(self, level):
        jepa = self.jepas[level - 1]
        return int(jepa.temporal_stride), int(jepa.temporal_window_size)

    def _draw_crops_and_level1_frames(self, seq_len, max_level, device):
        lengths = {1: int(seq_len)}
        for level in range(2, max_level + 1):
            stride, window = self.level_geometry(level)
            lengths[level] = (lengths[level - 1] - window) // stride + 1
        crop_starts, kept = {}, {}
        for level in range(1, max_level + 1):
            target = self.jepas[level - 1].target_length
            if target is None or target <= 0 or lengths[level] <= target:
                crop_starts[level] = None
                kept[level] = set(range(lengths[level]))
                continue
            start = self._draw_crop_start(lengths[level], target, device)
            crop_starts[level] = start
            kept[level] = set(range(start, start + target))
        needed = set(kept[max_level])
        for level in range(max_level, 1, -1):
            stride, window = self.level_geometry(level)
            needed = kept[level - 1] | {t * stride + k for t in needed for k in range(window)}
        return crop_starts, sorted(needed)

    def _encode_level1(self, info, key, frame_indices):
        ref = info[key]
        idx = torch.as_tensor(frame_indices, device=ref.device, dtype=torch.long)
        sparse_info = dict(info)
        for k, v in info.items():
            if (torch.is_tensor(v) and v.ndim >= 3 and v.size(1) == ref.size(1)
                    and not k.startswith("action")):
                sparse_info[k] = v.index_select(1, idx)
        level_out = self.jepas[0].encode(sparse_info, key=key)
        seq_len = ref.size(1)
        for name in ("embed", "pixel_embed", "proprio_embed"):
            sparse = level_out.get(f"{name}_0")
            if sparse is None:
                continue
            full = sparse.new_zeros(sparse.size(0), seq_len, *sparse.shape[2:])
            full[:, idx] = sparse
            info[f"{name}_1"] = full
        if "action_0" in level_out:
            info["action_1"] = level_out["action_0"]

    def _store_chunked_state_outputs(self, info, level_out, level):
        ref = level_out.get("embed_0")
        if not torch.is_tensor(ref) or ref.ndim < 3:
            return
        current_len = ref.size(1)
        prev = f"_level{level - 1}"
        for k, v in level_out.items():
            if k.startswith(("action", "embed_", "pixel_embed_", "proprio_embed_", "proprio_input")):
                continue
            if not torch.is_tensor(v) or v.ndim < 3 or not k.endswith(prev):
                continue
            if v.size(1) != current_len:
                raise ValueError(f"'{k}' has time length {v.size(1)}, level {level} has {current_len}")
            info[f"{k[: -len(prev)]}_level{level}"] = v

    def encode_hierarchical(self, info, sparse_level1_encode=True):
        """Lift the level-1 stream (`pixels_level1`, `action_level1`, ...) through every level
        and randomly crop each level to its `target_length`."""
        level_info = self._collect_level_info(info, 1)
        self._add_level_target_aliases(level_info, 1)
        level1_keys = set(level_info)

        seq_len = level_info["pixels"].size(1)
        if sparse_level1_encode:
            crop_starts, frames = self._draw_crops_and_level1_frames(
                seq_len, self.num_levels, level_info["pixels"].device)
        else:
            crop_starts, frames = {}, range(seq_len)
        self._encode_level1(level_info, "pixels", frames)

        for level in range(2, self.num_levels + 1):
            jepa = self.jepas[level - 1]
            level_info["action"] = level_info[f"action_{level - 1}"]
            if self.level_uses_proprio(level):
                level_info[jepa.encoder.proprio_key] = level_info[f"proprio_embed_{level - 1}"]
            level_out = jepa.encode(level_info, key=self.level_input_key(level),
                                    chunk_temporal_inputs=True)
            self._store_chunked_state_outputs(level_info, level_out, level)
            for name in ("embed", "pixel_embed", "proprio_embed"):
                if f"{name}_0" in level_out:
                    level_info[f"{name}_{level}"] = level_out[f"{name}_0"]
            level_info[f"action_{level}"] = level_out["action_0"]

        output = dict(info)
        output.update({k: v for k, v in level_info.items()
                       if k not in level1_keys and k != "proprio_input"})
        for level in range(1, self.num_levels + 1):
            target = self.jepas[level - 1].target_length
            if target is None:
                continue
            self._random_crop_level_outputs(output, level, target,
                                            extra_keys=level1_keys if level == 1 else None,
                                            start=crop_starts.get(level))
        return output
