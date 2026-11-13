"""One held-out clip per mixture member: every camera at steps 0/5/10/15 plus its state-delta
actions, as JSON for a visual inspection page.

    python scripts/oxe/inspect_mixture.py out.json   (run from the repo root)
"""
import base64, io, json, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from pathlib import Path
from hydra import compose, initialize_config_dir
from thesis.datasets.mixture.clips import MixtureSource, state_delta

with initialize_config_dir(str(Path("configs").resolve()), version_base=None):
    cfg = compose("base", overrides=["run=robojepa_oxe"])
src = MixtureSource(cfg.dataset, split="val")
T = src.frames
SHOW = [0, 5, 10, 15]

def jpg(x):
    x = F.interpolate(x[None].float(), size=(180, 240), mode="bilinear", antialias=True)[0]
    im = Image.fromarray(x.round().clamp(0, 255).byte().permute(1, 2, 0).numpy())
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()

def labels(pr):
    out = [f"pos {a}" for a in "xyz"[:len(pr.position)]]
    out += [f"rot {a}" for a in "xyz"[:len(pr.rotation)]]
    out += [f"other {i}" for i in range(len(pr.other))]
    out += ["gripper"] * len(pr.gripper)
    return out

rng = np.random.default_rng(0)
data = []
for m in src.members:
    pr = m.embodiment.proprio
    L = m.source.lengths
    eps = [e for e in m.episodes if L[e] >= (T + 1) * m.stride] or list(m.episodes)
    ep = int(eps[len(eps) // 2])
    start = max(0, (int(L[ep]) - (T + 1) * m.stride) // 2)
    fidx, states = m.window(ep, start, T)
    act = state_delta(states, pr)
    cams = []
    for c in m.cameras:
        raw = m.source.frames(c.column, ep, fidx)
        h, w = raw.shape[-2:]
        cams.append({"name": c.name, "view": c.view, "native": [int(w), int(h)],
                     "frames": [jpg(raw[i]) for i in SHOW if i < len(raw)]})
    deltas = []
    for _ in range(64):
        e = int(rng.choice(m.episodes))
        s0 = int(rng.integers(0, max(int(L[e]) - (T + 1) * m.stride, m.stride)))
        _, st = m.window(e, s0, T)
        if len(st) > 1:
            deltas.append(state_delta(st, pr))
    d = np.abs(np.concatenate(deltas))
    data.append({
        "name": m.embodiment.name, "fps": float(m.fps), "stride": m.stride,
        "step_s": m.stride / m.fps, "episode": ep, "start": start, "n": len(fidx),
        "labels": labels(pr), "state_cols": [int(i) for i in pr.dims], "column": pr.column,
        "actions": act.round(5).tolist(), "states": states.round(4).tolist(),
        "p50": np.percentile(d, 50, 0).round(5).tolist(),
        "p99": np.percentile(d, 99, 0).round(5).tolist(),
        "cams": cams, "show": [i for i in SHOW if i < len(fidx)],
    })
    print(m.embodiment.name, len(m.cameras), act.shape, file=sys.stderr)
Path(sys.argv[1]).write_text(json.dumps(data))
