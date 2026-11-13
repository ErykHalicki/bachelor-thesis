"""Dry run for the Open-X RLDS download: sizes, cameras, step counts, bandwidth. Stdlib only."""
import json, struct, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor

GCS = "https://storage.googleapis.com"
BUCKET = "gresearch"


def get(url, rng=None, timeout=120):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def listing(prefix, delimiter=None):
    items, prefixes, token = [], [], None
    while True:
        q = {"prefix": prefix, "fields": "items(name,size),prefixes,nextPageToken", "maxResults": 1000}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["pageToken"] = token
        d = json.loads(get(f"{GCS}/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(q)))
        items += d.get("items", [])
        prefixes += d.get("prefixes", [])
        token = d.get("nextPageToken")
        if not token:
            return items, prefixes


def varint(b, i):
    v = s = 0
    while True:
        c = b[i]; i += 1
        v |= (c & 0x7F) << s; s += 7
        if c < 0x80:
            return v, i


def fields(b):
    i, n = 0, len(b)
    while i < n:
        key, i = varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(b, i); yield f, wt, v
        elif wt == 2:
            ln, i = varint(b, i); yield f, wt, b[i:i + ln]; i += ln
        elif wt == 5:
            yield f, wt, b[i:i + 4]; i += 4
        elif wt == 1:
            yield f, wt, b[i:i + 8]; i += 8
        else:
            raise ValueError(wt)


def feature_len(feat):
    for f, wt, v in fields(feat):
        if f == 1:      # bytes list
            return sum(1 for ff, _, _ in fields(v) if ff == 1)
        if f == 2:      # float list (packed or not)
            return sum(len(x) // 4 if w == 2 else 1 for ff, w, x in fields(v) if ff == 1)
        if f == 3:      # int64 list
            c = 0
            for ff, w, x in fields(v):
                if ff != 1:
                    continue
                if w == 2:
                    j = 0
                    while j < len(x):
                        _, j = varint(x, j); c += 1
                else:
                    c += 1
            return c
    return 0


def episode_steps(record):
    for f, _, feats in fields(record):          # Example.features
        if f != 1:
            continue
        for ff, _, entry in fields(feats):      # map entries
            key = feat = None
            for k, _, v in fields(entry):
                if k == 1: key = v.decode()
                elif k == 2: feat = v
            if key in ("steps/is_first", "steps/is_last", "steps/reward"):
                return feature_len(feat)
    return None


def sample_steps(url, size, budget):
    """Steps of the first episodes of one shard, reading only those records."""
    out, off, spent = [], 0, 0
    big = []
    while off + 12 < size and spent < budget and len(out) < 3:
        ln = struct.unpack("<Q", get(url, (off, off + 7)))[0]
        if ln > budget:
            big.append(ln)
            break
        if spent + ln > budget and out:
            break
        rec = get(url, (off + 12, off + 12 + ln - 1), timeout=600)
        spent += ln
        s = episode_steps(rec)
        if s is not None:
            out.append(s)
        off += 12 + ln + 4
    return out, spent, big


def walk_images(node, path=""):
    if isinstance(node, dict):
        cls = node.get("pythonClassName", "")
        if cls.endswith("Image") or cls.endswith("Tensor") and "image" in path.lower():
            shape = node.get("image", node.get("tensor", {})).get("shape", {}).get("dimensions")
            dtype = node.get("image", node.get("tensor", {})).get("dtype")
            if shape and len(shape) == 3:
                yield path, [int(x) for x in shape], dtype
        for k, v in node.items():
            yield from walk_images(v, path + "/" + k if k not in ("featuresDict", "features", "feature", "sequence") else path)


def inspect(cand):
    name = cand["reg"]
    try:
        _, versions = listing(f"robotics/{name}/", "/")
        if not versions:
            return {**cand, "error": "not in gs://gresearch/robotics"}
        ver = sorted(versions, key=lambda p: [int(x) if x.isdigit() else 0 for x in p.rstrip("/").split("/")[-1].split(".")])[-1]
        items, _ = listing(ver)
        info = json.loads(get(f"{GCS}/{BUCKET}/{ver}dataset_info.json"))
        feats = json.loads(get(f"{GCS}/{BUCKET}/{ver}features.json"))
        shards = [it for it in items if ".tfrecord-" in it["name"]]
        total = sum(int(it["size"]) for it in shards)
        splits = {s["name"]: sum(int(x) for x in s.get("shardLengths", [])) for s in info.get("splits", [])}
        cams = []
        for path, shape, dtype in walk_images(feats):
            p = path.lower()
            depth = "depth" in p or shape[-1] == 1 or dtype in ("uint16", "float32")
            cams.append({"key": path.split("observation/")[-1], "hw": shape[:2], "c": shape[2],
                         "dtype": dtype, "depth": depth})
        train = sorted((it for it in shards if "-train." in it["name"]), key=lambda it: it["name"]) or shards
        picks = [train[0], train[len(train) // 2], train[-1]] if len(train) >= 3 else train
        steps, spent, big = [], 0, []
        for it in picks:
            s, b, g = sample_steps(f"{GCS}/{BUCKET}/{urllib.parse.quote(it['name'])}",
                                   int(it["size"]), budget=40e6)
            steps += s; spent += b; big += g
        eps = sum(splits.values())
        mean = sum(steps) / len(steps) if steps else None
        return {**cand, "version": ver.rstrip("/").split("/")[-1], "bytes": total, "shards": len(shards),
                "splits": splits, "episodes": eps, "cams": cams, "sample_steps": steps,
                "mean_steps": mean, "est_steps": round(mean * eps) if mean else None,
                "sampled_bytes": spent, "skipped_episode_bytes": big}
    except Exception as e:
        return {**cand, "error": f"{type(e).__name__}: {e}"}


def bandwidth():
    _, versions = listing("robotics/bc_z/", "/")
    items, _ = listing(sorted(versions)[-1])
    shards = sorted((it for it in items if ".tfrecord-" in it["name"]), key=lambda it: it["name"])

    def pull(it):
        url = f"{GCS}/{BUCKET}/{urllib.parse.quote(it['name'])}"
        n = 0
        with urllib.request.urlopen(url, timeout=600) as r:
            while True:
                b = r.read(1 << 20)
                if not b:
                    return n
                n += len(b)

    res = {}
    for streams, picks in ((1, shards[:1]), (8, shards[1:9])):
        t = time.time()
        with ThreadPoolExecutor(streams) as pool:
            n = sum(pool.map(pull, picks))
        dt = time.time() - t
        res[streams] = {"bytes": n, "seconds": round(dt, 1), "MBps": round(n / dt / 1e6, 1)}
        print("bandwidth", streams, res[streams], file=sys.stderr, flush=True)
    return res


if __name__ == "__main__":
    cands = json.load(open(sys.argv[1]))
    bw = bandwidth()
    with ThreadPoolExecutor(8) as pool:
        out = []
        for r in pool.map(inspect, cands):
            print(r["reg"], r.get("error") or (r["version"], round(r["bytes"] / 1e9, 1), r["episodes"],
                                              r["mean_steps"]), file=sys.stderr, flush=True)
            out.append(r)
    json.dump({"bandwidth": bw, "datasets": out}, open(sys.argv[2], "w"), indent=1)
