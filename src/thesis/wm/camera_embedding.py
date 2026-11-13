"""How a view's identity becomes a learned vector added to its token.

The data layer always says which (embodiment, camera) a view is. A strategy decides which
table row that pair reads:

    per_embodiment_camera   one vector per (embodiment, camera); nothing is shared
    per_role                one vector per role tag, shared across embodiments
    per_role_plus_offset    per_role plus a zero-initialised per-(embodiment, camera) offset
    per_index               one vector per camera index, ignoring identity
    none                    no embedding; views are an unordered set
"""

import torch
import torch.nn as nn


def _pairs(spec):
    return [(e, c) for e, emb in enumerate(spec["embodiments"]) for c in range(len(emb["cameras"]))]


def _by_pair(spec):
    return {pair: i for i, pair in enumerate(_pairs(spec))}


def _by_role(spec):
    roles = {}
    for e, c in _pairs(spec):
        emb = spec["embodiments"][e]
        role = emb["cameras"][c]["role"]
        if role is None:
            raise ValueError(f"{emb['name']}/{emb['cameras'][c]['name']} has no role, "
                             f"which role-based camera embeddings need")
        roles.setdefault(role, len(roles))
    return {(e, c): roles[spec["embodiments"][e]["cameras"][c]["role"]] for e, c in _pairs(spec)}


def _by_index(spec):
    return {(e, c): c for e, c in _pairs(spec)}


STRATEGIES = {
    "per_embodiment_camera": [(_by_pair, False)],
    "per_role": [(_by_role, False)],
    "per_role_plus_offset": [(_by_role, False), (_by_pair, True)],
    "per_index": [(_by_index, False)],
    "none": [],
}


class CameraEmbedding(nn.Module):
    def __init__(self, spec, dim, strategy="per_embodiment_camera"):
        super().__init__()
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown camera embedding '{strategy}', have {sorted(STRATEGIES)}")
        self.dim = dim
        self.tables = nn.ModuleList()
        max_cams = max(len(e["cameras"]) for e in spec["embodiments"])
        for t, (index_fn, zero_init) in enumerate(STRATEGIES[strategy]):
            index = index_fn(spec)
            lookup = torch.zeros(len(spec["embodiments"]), max_cams, dtype=torch.long)
            for (e, c), i in index.items():
                lookup[e, c] = i
            self.register_buffer(f"lookup_{t}", lookup)
            table = nn.Embedding(max(index.values()) + 1, dim)
            nn.init.zeros_(table.weight) if zero_init else nn.init.normal_(table.weight, std=0.02)
            self.tables.append(table)

    def forward(self, view_emb, view_cam):
        out = torch.zeros(len(view_emb), self.dim, device=view_emb.device)
        for t, table in enumerate(self.tables):
            out = out + table(getattr(self, f"lookup_{t}")[view_emb, view_cam])
        return out
