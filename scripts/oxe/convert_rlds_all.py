"""Convert a whole list of Open-X RLDS datasets into LeRobot v3, a few at a time.

    python scripts/oxe/convert_rlds_all.py --list configs/mixture/rlds_datasets.yaml \
        --out ~/data/oxe/rlds --tmp ~/data/oxe/_rlds_tmp --logs outputs/oxe_rlds

Each dataset runs as its own `convert_rlds.py` process, logging to
<logs>/<name>.log, `--lanes` of them at once, in list order. Finished datasets are skipped,
and a failed one is retried `--retries` times; a retry resumes from the dataset's last
checkpoint, so re-running this command after any interruption continues where it stopped.
No new dataset starts while the output disk has less than `--min-free-gb` free.
<logs>/summary.json records every dataset's status, episodes, frames and size.
"""

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


def short(name):
    return name.removesuffix("_converted_externally_to_rlds")


def finished(out):
    return (out / "meta/info.json").exists() and not (out / ".progress.json").exists()


def size_gb(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tmp", type=Path, required=True)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--decode-threads", type=int, default=4)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--only", nargs="*", default=None, help="names to run, default all")
    a = ap.parse_args()
    out_root, tmp_root = a.out.expanduser(), a.tmp.expanduser()
    a.logs.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    entries = yaml.safe_load(a.list.read_text())["datasets"]
    if a.only:
        entries = [e for e in entries if e["name"] in a.only or short(e["name"]) in a.only]
    status = {short(e["name"]): {"status": "pending"} for e in entries}
    lock = threading.Lock()
    summary = a.logs / "summary.json"

    def save():
        with lock:
            summary.write_text(json.dumps(status, indent=1))

    def convert(entry):
        name = entry["name"]
        out = out_root / short(name)
        rec = status[short(name)]
        if finished(out):
            rec.update(status="done (earlier)")
        else:
            while shutil.disk_usage(out_root).free / 1e9 < a.min_free_gb:
                rec.update(status="waiting for disk space")
                save()
                time.sleep(300)
            cmd = [sys.executable, "-u", str(Path(__file__).with_name("convert_rlds.py")), name,
                   "--fps", str(entry["fps"]), "--out", str(out_root), "--tmp", str(tmp_root),
                   "--decode-threads", str(a.decode_threads)]
            for attempt in range(a.retries + 1):
                rec.update(status=f"running (attempt {attempt + 1})", started=time.ctime())
                save()
                with open(a.logs / f"{short(name)}.log", "a") as log:
                    log.write(f"\n=== attempt {attempt + 1}, {time.ctime()}\n")
                    log.flush()
                    code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
                if code == 0 and finished(out):
                    rec.update(status="done")
                    break
                rec.update(status=f"failed (exit {code})")
                time.sleep(30)
        if finished(out):
            info = json.loads((out / "meta/info.json").read_text())
            rec.update(episodes=info["total_episodes"], frames=info["total_frames"],
                       gb=round(size_gb(out), 2), finished=time.ctime())
        save()
        print(f"{short(name)}: {rec['status']}", flush=True)

    save()
    with ThreadPoolExecutor(a.lanes) as pool:
        list(pool.map(convert, entries))
    failed = [n for n, r in status.items() if not r["status"].startswith("done")]
    print(f"{len(entries) - len(failed)}/{len(entries)} datasets converted"
          + (f"; failed: {', '.join(failed)}" if failed else ""), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
