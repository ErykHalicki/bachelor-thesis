"""The batch a mixed-embodiment stream hands the model.

Views are the unit for cameras and embodiments are the unit for actions and state; neither
is padded. Every kept view of every sample is one row of `views`, so the encoder sees a
flat batch and never learns how many cameras a sample had. Actions and state stay in their
embodiment's native width, grouped per embodiment with the batch rows they belong to.
"""

from dataclasses import dataclass, fields

import torch


@dataclass
class Batch:
    views: torch.Tensor          # uint8 (N, T, 3, H, W), N = views summed over samples
    view_sample: torch.Tensor    # (N,) batch row of each view
    view_slot: torch.Tensor      # (N,) position of the view within its sample
    view_emb: torch.Tensor       # (N,) embodiment id
    view_cam: torch.Tensor       # (N,) camera index within the embodiment
    emb_id: torch.Tensor         # (B,)
    src_fps: torch.Tensor        # (B,)
    groups: dict                 # emb_id -> {idx (b,), action (b, T-1, K, D), state (b, T, S)}

    @property
    def size(self):
        return len(self.emb_id)

    def _map(self, fn):
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        for name, value in out.items():
            if name == "groups":
                out[name] = {e: {k: fn(v) for k, v in g.items()} for e, g in value.items()}
            else:
                out[name] = fn(value)
        return Batch(**out)

    def to(self, device):
        return self._map(lambda t: t.to(device, non_blocking=True))

    def pin_memory(self):
        return self._map(lambda t: t.pin_memory())


def collate(samples):
    """`samples` are the dicts `Member.window` returns."""
    counts = torch.tensor([len(s["cams"]) for s in samples])
    rows = torch.arange(len(samples))
    emb_id = torch.tensor([s["emb"] for s in samples])
    groups = {}
    for e in emb_id.unique().tolist():
        idx = (emb_id == e).nonzero().squeeze(1)
        groups[e] = {
            "idx": idx,
            "action": torch.stack([torch.from_numpy(samples[i]["action"]) for i in idx]),
            "state": torch.stack([torch.from_numpy(samples[i]["state"]) for i in idx]),
        }
    return Batch(
        views=torch.cat([s["frames"] for s in samples]),
        view_sample=rows.repeat_interleave(counts),
        view_slot=torch.cat([torch.arange(c) for c in counts.tolist()]),
        view_emb=emb_id.repeat_interleave(counts),
        view_cam=torch.cat([torch.as_tensor(s["cams"]) for s in samples]),
        emb_id=emb_id,
        src_fps=torch.tensor([s["fps"] for s in samples], dtype=torch.float32),
        groups=groups,
    )
