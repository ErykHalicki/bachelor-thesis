"""Audit Open-X RLDS datasets at the source (feature spec plus two decoded episodes each).

    python scripts/oxe/audit.py scripts/oxe/oxe_sheet.json audit.json

`oxe_sheet.json` is the Open-X dataset spreadsheet (Dataset Overview tab) as JSON rows.
"""
import base64, io, json, struct, sys, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dryrun import GCS, BUCKET, get, listing, fields, varint

def pick_version(name):
    _, versions = listing(f"robotics/{name}/", "/")
    key = lambda p: [int(x) if x.isdigit() else 0 for x in p.rstrip("/").split("/")[-1].split(".")]
    for ver in sorted(versions, key=key, reverse=True):
        items, _ = listing(ver)
        shards = [it for it in items if ".tfrecord-" in it["name"]]
        if shards:
            return ver, items, shards
    return None, [], []

def spec_leaves(node, path=""):
    """(path, class, dtype, shape, description) for every leaf of features.json."""
    if not isinstance(node, dict):
        return
    cls = node.get("pythonClassName", "").split(".")[-1]
    desc = node.get("description", "")
    for k in ("image", "tensor", "text"):
        if k in node and isinstance(node[k], dict):
            t = node[k]
            yield path, cls, t.get("dtype"), (t.get("shape") or {}).get("dimensions"), desc
            return
    if cls in ("Text", "Scalar") or "dtype" in node:
        yield path, cls, node.get("dtype"), None, desc
    for k, v in node.items():
        if k in ("featuresDict", "features", "feature", "sequence"):
            yield from spec_leaves(v, path)
        elif isinstance(v, dict) and k not in ("image", "tensor", "text"):
            yield from spec_leaves(v, path + "/" + k)

def parse_record(rec):
    out = {}
    for f, _, feats in fields(rec):
        if f != 1:
            continue
        for _, _, entry in fields(feats):
            key = feat = None
            for k, _, v in fields(entry):
                if k == 1: key = v.decode()
                elif k == 2: feat = v
            kind, vals, nbytes = None, None, len(feat)
            for ff, _, v in fields(feat):
                if ff == 1:
                    kind = "bytes"; vals = [x for fff, _, x in fields(v) if fff == 1]
                elif ff == 2:
                    kind = "float"; buf = b"".join(x if w == 2 else x for fff, w, x in fields(v) if fff == 1)
                    vals = np.frombuffer(buf, "<f4")
                elif ff == 3:
                    kind = "int"; ints = []
                    for fff, w, x in fields(v):
                        if fff != 1: continue
                        if w == 2:
                            j = 0
                            while j < len(x):
                                val, j = varint(x, j); ints.append(val if val < 1 << 63 else val - (1 << 64))
                        else:
                            ints.append(x if x < 1 << 63 else x - (1 << 64))
                    vals = np.array(ints, np.int64)
            out[key] = (kind, vals, nbytes)
    return out

def thumb(img):
    im = img.convert("RGB"); im.thumbnail((200, 150))
    b = io.BytesIO(); im.save(b, "JPEG", quality=75)
    return base64.b64encode(b.getvalue()).decode()

def analyse(rec):
    p = parse_record(rec)
    T = len(p.get("steps/is_first", (None, []))[1]) or None
    keys = {}
    for key, (kind, vals, nbytes) in p.items():
        info = {"kind": kind, "bytes": nbytes, "n": len(vals) if vals is not None else 0}
        if T and key.startswith("steps/") and info["n"] and info["n"] % T == 0:
            per = info["n"] // T
            info["per_step"] = per
            if kind in ("float", "int") and per <= 4096:
                a = np.asarray(vals, np.float64).reshape(T, per)
                fin = np.isfinite(a)
                std = np.where(fin.all(0), a.std(0), np.nan)
                info.update(nan=int((~fin).sum()), const_dims=int((std < 1e-9).sum()),
                            dims=per, absmax=float(np.nanmax(np.abs(a))) if fin.any() else None,
                            first=[round(float(x), 4) for x in a[0, :12]],
                            std=[round(float(x), 5) for x in std[:12]])
            elif kind == "bytes":
                v0 = vals[0]
                if v0[:2] == b"\xff\xd8" or v0[:4] == b"\x89PNG":
                    mid = vals[(len(vals) // 2)]
                    im = Image.open(io.BytesIO(mid)); im.load()
                    arr = np.asarray(im)
                    info.update(image=True, fmt=im.format, mode=im.mode, wh=list(im.size),
                                pix_std=float(arr.std()), avg_item=nbytes / len(vals))
                    if im.mode in ("RGB", "RGBA", "L") and arr.dtype == np.uint8:
                        info["thumb"] = thumb(im)
                elif len(v0) < 400:
                    try: info["text"] = v0.decode()[:200]
                    except Exception: pass
        elif kind == "bytes" and vals and len(vals[0]) < 400:
            try: info["text"] = vals[0].decode()[:200]
            except Exception: pass
        keys[key] = info
    return {"T": T, "bytes": len(rec), "keys": keys}

def read_records(url, size, n, cap):
    out, off = [], 0
    while off + 12 < size and len(out) < n:
        ln = struct.unpack("<Q", get(url, (off, off + 7)))[0]
        if ln > cap:
            return out, ln
        out.append(get(url, (off + 12, off + 12 + ln - 1), timeout=1800))
        off += 12 + ln + 4
    return out, None

def audit(row):
    name = row["Registered Dataset Name"].strip()
    try:
        ver, items, shards = pick_version(name)
        if not ver:
            return {"name": name, "sheet": row, "error": "not on gs://gresearch/robotics"}
        base = f"{GCS}/{BUCKET}/{ver}"
        feats = json.loads(get(base + "features.json"))
        info = json.loads(get(base + "dataset_info.json"))
        splits = {s["name"]: sum(int(x) for x in s.get("shardLengths", [])) for s in info.get("splits", [])}
        train = sorted((it for it in shards if "-train." in it["name"]), key=lambda it: it["name"]) or shards
        eps, too_big = [], None
        for it in [train[0], train[len(train) // 2]] if len(train) > 1 else train:
            recs, big = read_records(f"{GCS}/{BUCKET}/{urllib.parse.quote(it['name'])}", int(it["size"]), 1, 1.6e9)
            too_big = too_big or big
            eps += [analyse(r) for r in recs]
        return {"name": name, "sheet": row, "version": ver.rstrip("/").split("/")[-1],
                "bytes": sum(int(it["size"]) for it in shards), "splits": splits,
                "episodes": sum(splits.values()), "description": info.get("description", "")[:1500],
                "spec": [list(x) for x in spec_leaves(feats)], "samples": eps, "too_big": too_big}
    except Exception as e:
        import traceback
        return {"name": name, "sheet": row, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}

if __name__ == "__main__":
    rows = json.load(open(sys.argv[1]))
    out = []
    with ThreadPoolExecutor(4) as pool:
        for r in pool.map(audit, rows):
            s = r.get("samples") or []
            print(r["name"], r.get("error") or (r["version"], [x["T"] for x in s], r.get("too_big")), flush=True)
            out.append(r)
    json.dump(out, open(sys.argv[2], "w"))
