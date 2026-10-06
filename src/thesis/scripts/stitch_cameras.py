"""Stitch a LeRobot dataset's camera videos into one side-by-side mp4.

Point it at a dataset root (a training set, or an eval rollout recording under
outputs/rollouts/) and it joins each camera's chunk files in order, scales every view to
one height, and hstacks them left-to-right into a single real-time video -- the whole
session watchable in one player, cameras aligned frame-for-frame.

    python -m thesis.scripts.stitch_cameras outputs/rollouts/20260818_143000
    python -m thesis.scripts.stitch_cameras <root> --cameras observation.images.zed_left,observation.images.wrist_cam
    python -m thesis.scripts.stitch_cameras <root> --height 480 --out /tmp/session.mp4

Needs ffmpeg on PATH. Camera order is left-to-right: sorted by name unless --cameras
gives an explicit order.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def _camera_video(root, camera, tmpdir):
    """One mp4 spanning all of `camera`'s chunk files, joined by stream copy."""
    cam_dir = root / "videos" / camera
    files = sorted(f for f in cam_dir.rglob("*.mp4") if "stitched" not in f.name)
    if not files:
        raise SystemExit(f"no mp4 files under {cam_dir}")
    if len(files) == 1:
        return files[0]
    listing = Path(tmpdir) / f"{camera.replace('/', '_')}.txt"
    listing.write_text("".join(f"file '{f.resolve()}'\n" for f in files))
    joined = Path(tmpdir) / f"{camera.replace('/', '_')}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(joined)],
        check=True,
    )
    return joined


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="LeRobot dataset root (contains videos/)")
    p.add_argument("--cameras", default=None,
                   help="comma-separated camera keys, left-to-right; default: all, sorted")
    p.add_argument("--height", type=int, default=360,
                   help="common height every view is scaled to (aspect kept)")
    p.add_argument("--out", default=None, help="output path; default <root>/stitched.mp4")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    videos_dir = root / "videos"
    if not videos_dir.is_dir():
        raise SystemExit(f"{root} has no videos/ directory -- not a video dataset root")
    cameras = (args.cameras.split(",") if args.cameras
               else sorted(d.name for d in videos_dir.iterdir() if d.is_dir()))
    if not cameras:
        raise SystemExit(f"no camera directories under {videos_dir}")
    out = Path(args.out) if args.out else root / "stitched.mp4"

    with tempfile.TemporaryDirectory() as tmpdir:
        inputs = [_camera_video(root, cam, tmpdir) for cam in cameras]
        if len(inputs) == 1:
            print(f"only one camera ({cameras[0]}); nothing to stitch -- its video is "
                  f"{inputs[0]}")
            return
        # scale each view to the common height (-2 keeps aspect at even width), then
        # hstack left-to-right; filtering re-encodes, so pick the widely playable codec
        scaled = "".join(f"[{i}:v]scale=-2:{args.height}[v{i}];" for i in range(len(inputs)))
        stack = "".join(f"[v{i}]" for i in range(len(inputs)))
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        for f in inputs:
            cmd += ["-i", str(f)]
        cmd += ["-filter_complex", f"{scaled}{stack}hstack=inputs={len(inputs)}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", str(out)]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            raise SystemExit("ffmpeg not found on PATH") from None

    print(f"wrote {out}  ({' | '.join(c.rsplit('.', 1)[-1] for c in cameras)}, "
          f"height {args.height})")


if __name__ == "__main__":
    main()
