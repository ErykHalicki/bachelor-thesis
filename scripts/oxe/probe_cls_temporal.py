"""How much LeVJEPA [cls] tokens change within a clip versus across clips, on held-out
mixture clips (layer-normed, as the world model sees them).

    python scripts/oxe/probe_cls_temporal.py   (run from the repo root)
"""
import torch, torch.nn.functional as F
from hydra import compose, initialize_config_dir
from pathlib import Path
from thesis.datasets import build_dataset
from thesis.algorithms.encoders import build_encoder
from omegaconf import OmegaConf

with initialize_config_dir(str(Path("configs").resolve()), version_base=None):
    cfg = compose("base", overrides=["run=robojepa_oxe"])
ds = build_dataset(cfg.dataset, augment=False, split="val")
enc_spec = OmegaConf.to_container(cfg.algorithm.encoders.levjepa, resolve=True)
enc_spec["frozen"] = True
enc = build_encoder(enc_spec).cuda().half().eval()

rows = []
step = max(1, len(ds) // 48)
for i in range(0, len(ds), step)[:48]:
    item = ds[i]
    imgs, sv = item["observation.images"], item["step_valid"].bool()
    x = imgs[sv, 0]
    with torch.no_grad():
        z = enc(x.cuda().unsqueeze(0))[0]
    z = z.reshape(x.shape[0], -1).float()
    z = F.layer_norm(z, z.shape[-1:])
    rows.append((int(item["embodiment_id"]), z.cpu()))

cos1 = torch.cat([F.cosine_similarity(z[1:], z[:-1]) for _, z in rows]).mean()
cosend = torch.stack([F.cosine_similarity(z[0], z[-1], dim=0) for _, z in rows]).mean()
allz = torch.cat([z for _, z in rows])
total = allz.var(0).mean()
within = torch.stack([z.var(0).mean() for _, z in rows]).mean()
means = torch.stack([z.mean(0) for _, z in rows])
cross = F.cosine_similarity(means.unsqueeze(0), means.unsqueeze(1), dim=-1)
off = cross[~torch.eye(len(rows), dtype=bool)].mean()
copy = torch.cat([(z[1:] - z[:-1]).abs().mean(1) for _, z in rows]).mean()
meanpred = torch.cat([(z[1:] - z.mean(0)).abs().mean(1) for _, z in rows]).mean()
print(f"clips={len(rows)} frames/clip={sum(len(z) for _, z in rows)/len(rows):.1f}")
print(f"cos(consecutive frames)={cos1:.4f}  cos(first,last)={cosend:.4f}  cos(clip means, different clips)={off:.4f}")
print(f"within-clip temporal var / total var over valid frames = {within/total:.4f}")
print(f"copy-last-frame L1={copy:.4f}  clip-mean L1={meanpred:.4f}  (LN'd targets, per-dim)")
for e in sorted({e for e, _ in rows}):
    zs = [z for ee, z in rows if ee == e]
    c = torch.cat([F.cosine_similarity(z[1:], z[:-1]) for z in zs]).mean()
    w = torch.stack([z.var(0).mean() for z in zs]).mean() / total
    print(f"  emb {e}: n={len(zs)} cos_consec={c:.4f} temporal_share={w:.4f}")
