"""
Re-encode all videos in a LeRobot dataset to h264 short-GOP in place, optionally
rescaling to a target resolution.

AV1 decode is slow and long-GOP codecs punish random-seek training dataloaders.
This script swaps every mp4 for an h264 short-GOP version (default --gop 1,
all-intra) and optionally downscales to --scale (default 256) so training workers
never have to resize on the fly. meta/info.json is updated in place.

Usage:
    python -m thesis.scripts.reencode_dataset <repo_id>
    python -m thesis.scripts.reencode_dataset ehalicki/so101_cube
    python -m thesis.scripts.reencode_dataset ehalicki/so101_cube --gop 4 --crf 18 --scale 0
    python -m thesis.scripts.reencode_dataset ehalicki/so101_cube --scale 256
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_HOME


def reencode_one(src: Path, gop: int, crf: int, preset: str, scale: int) -> None:
    tmp = src.with_suffix(".reencode.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
    ]
    if scale > 0:
        cmd += ["-vf", f"scale={scale}:{scale}"]
    cmd += [
        "-c:v", "libx264",
        "-g", str(gop),
        "-keyint_min", str(gop),
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-an",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    shutil.move(str(tmp), str(src))


def update_info_json(info_path: Path, scale: int) -> int:
    if not info_path.exists():
        return 0
    with open(info_path) as f:
        info = json.load(f)
    changed = 0
    features = info.get("features", info)
    for val in features.values():
        if not isinstance(val, dict):
            continue
        if val.get("dtype") != "video":
            continue
        vinfo = val.get("info", {})
        if vinfo.get("video.codec") and vinfo["video.codec"] != "h264":
            vinfo["video.codec"] = "h264"
            changed += 1
        if scale > 0:
            if vinfo.get("video.height") != scale:
                vinfo["video.height"] = scale
                changed += 1
            if vinfo.get("video.width") != scale:
                vinfo["video.width"] = scale
                changed += 1
            shape = val.get("shape")
            if isinstance(shape, list) and len(shape) == 3:
                if shape[0] != scale or shape[1] != scale:
                    val["shape"] = [scale, scale, shape[2]]
                    changed += 1
    if changed:
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
    return changed


def main():
    parser = argparse.ArgumentParser(description="Re-encode LeRobot dataset videos to h264 short-GOP in place, optionally rescaling")
    parser.add_argument("repo_id", help="LeRobot repo id (e.g. ehalicki/so101_cube)")
    parser.add_argument("--gop", type=int, default=1, help="GOP size (1 = all-intra, best for random seek)")
    parser.add_argument("--crf", type=int, default=18, help="libx264 CRF (lower = higher quality)")
    parser.add_argument("--preset", default="fast", help="libx264 preset")
    parser.add_argument("--scale", type=int, default=256, help="Rescale to NxN (0 = keep original resolution)")
    parser.add_argument("--cache-root", default=None, help="Override cache root (default: HF_LEROBOT_HOME). Used with repo_id to build the dataset path.")
    parser.add_argument("--root", default=None, help="Path directly to the dataset dir. Bypasses --cache-root/repo_id joining. Use this for slurm-style pipelines where datasets live outside HF_LEROBOT_HOME.")
    args = parser.parse_args()

    if args.root:
        repo_dir = Path(args.root)
    else:
        cache_root = Path(args.cache_root) if args.cache_root else HF_LEROBOT_HOME
        repo_dir = cache_root / args.repo_id
    if not repo_dir.exists():
        print(f"ERROR: repo not found at {repo_dir}", file=sys.stderr)
        sys.exit(1)

    mp4s = sorted(repo_dir.rglob("*.mp4"))
    if not mp4s:
        print(f"No .mp4 files found under {repo_dir}")
        sys.exit(0)

    total_before = sum(p.stat().st_size for p in mp4s)
    print(f"Found {len(mp4s)} mp4 files under {repo_dir}")
    scale_desc = f"{args.scale}x{args.scale}" if args.scale > 0 else "original"
    print(f"Re-encoding: libx264 -g {args.gop} -crf {args.crf} -preset {args.preset}  scale={scale_desc}")
    print(f"Total size before: {total_before / 1e9:.2f} GB")
    print()

    for i, src in enumerate(mp4s, 1):
        rel = src.relative_to(repo_dir)
        before = src.stat().st_size
        try:
            reencode_one(src, args.gop, args.crf, args.preset, args.scale)
        except subprocess.CalledProcessError as e:
            print(f"  [{i}/{len(mp4s)}] FAILED {rel}: {e}", file=sys.stderr)
            continue
        after = src.stat().st_size
        ratio = after / max(before, 1)
        print(f"  [{i}/{len(mp4s)}] {rel}  {before/1e6:.1f}MB -> {after/1e6:.1f}MB  ({ratio:.2f}x)")

    total_after = sum(p.stat().st_size for p in mp4s)
    print()
    print(f"Total size after:  {total_after / 1e9:.2f} GB  ({total_after / max(total_before, 1):.2f}x)")

    info_updates = 0
    for info_path in repo_dir.rglob("meta/info.json"):
        n = update_info_json(info_path, args.scale)
        if n:
            info_updates += n
            print(f"Updated {info_path.relative_to(repo_dir)} ({n} field(s))")
    if info_updates == 0:
        print("No meta/info.json updates needed.")


if __name__ == "__main__":
    main()
