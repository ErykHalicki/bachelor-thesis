"""Per-embodiment model parts: action normalization and the projection into the predictor.

Each embodiment keeps its native action width. Its chunk of K sub-steps is normalized with
that embodiment's own statistics and flattened through its own linear map, so nothing in
the data has to share a space. Statistics are buffers, saved with the checkpoint.
"""

import torch
import torch.nn as nn


class ActionAdapters(nn.Module):
    CLIP = 5.0

    def __init__(self, spec, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.ModuleList()
        for i, e in enumerate(spec["embodiments"]):
            lo = torch.tensor(e["stats"]["action"]["q01"], dtype=torch.float32)
            hi = torch.tensor(e["stats"]["action"]["q99"], dtype=torch.float32)
            self.register_buffer(f"mid_{i}", (hi + lo) / 2)
            self.register_buffer(f"half_{i}", ((hi - lo) / 2).clamp_min(1e-6))
            self.proj.append(nn.Linear(e["substeps"] * e["action_dim"], dim))

    def normalize(self, emb_id, action):
        mid, half = getattr(self, f"mid_{emb_id}"), getattr(self, f"half_{emb_id}")
        return ((action - mid) / half).clamp(-self.CLIP, self.CLIP)

    def forward(self, batch):
        """One action token per step, (B, T - 1, dim), from each row's native chunk."""
        some = next(iter(batch.groups.values()))["action"]
        out = some.new_zeros(batch.size, some.shape[1], self.dim)
        for e, group in batch.groups.items():
            tokens = self.proj[e](self.normalize(e, group["action"]).flatten(2))
            out[group["idx"]] = tokens.to(out.dtype)
        return out
