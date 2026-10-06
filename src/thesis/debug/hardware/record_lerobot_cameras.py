#!/usr/bin/env python3
"""Records the generic webcam and the ZED/ZED-Mini side by side into a single
MP4, going through lerobot's actual camera backends (`OpenCVCamera` and
`ZedCamera`) rather than the hand-rolled cv2/py-zed-open-capture path in
cameras.py -- useful for sanity-checking the lerobot ZED integration end to
end. By default the ZED is recorded in "stereo" mode, i.e. its full undivided
left+right frame, so the output is [webcam | zed-left | zed-right] with no
extra camera object needed for the second eye. Run inside the project venv:
    source .venv/bin/activate
    python debug/hardware/record_lerobot_cameras.py
    python debug/hardware/record_lerobot_cameras.py --opencv-index-or-path /dev/video0 --zed-side left --seconds 10
Ctrl+C stops the recording early and still finalizes the file.

See lerobot_cameras.py for camera detection/setup.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from lerobot_cameras import open_cameras
from thesis.utils.stream import merge
from thesis.utils.stderr_filter import CORRUPT_JPEG, suppress_stderr_lines

DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
RECORDINGS_DIR = Path(__file__).parent / "recordings"


def default_output():
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return str(RECORDINGS_DIR / f"lerobot_cameras_{datetime.now():%Y%m%d_%H%M%S}.mp4")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--opencv-index-or-path",
        default=None,
        help="OpenCV camera index or /dev/video path for the generic webcam "
        "(default: auto-detect the first non-ZED capture node)",
    )
    parser.add_argument(
        "--zed-serial", default=None, help="ZED serial number (default: auto-detect the first ZED found)"
    )
    parser.add_argument(
        "--zed-side",
        default="stereo",
        choices=["left", "right", "stereo"],
        help="Which ZED eye(s) to record: 'left'/'right' for one cropped eye, "
        "'stereo' for the full undivided left+right frame (default: stereo)",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="target capture/output frame rate")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="per-camera height before stacking")
    parser.add_argument(
        "--seconds", type=float, default=0, help="stop after this many seconds (default: record until Ctrl+C)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output .mp4 path (default: debug/hardware/recordings/lerobot_cameras_<timestamp>.mp4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or default_output()
    suppress_stderr_lines([CORRUPT_JPEG])

    webcam, zed = open_cameras(
        args.fps,
        zed_side=args.zed_side,
        opencv_index_or_path=args.opencv_index_or_path,
        zed_serial=args.zed_serial,
    )

    frames = [None, None]
    writer = None
    period = 1 / args.fps
    n = 0
    t_start = time.time()
    try:
        while not args.seconds or time.time() - t_start < args.seconds:
            t0 = time.time()
            for i, cam in enumerate((webcam, zed)):
                try:
                    frame = cam.async_read()
                except TimeoutError as e:
                    if frames[i] is None:
                        raise RuntimeError(f"{cam} produced no frame before timing out.") from e
                    print(f"Warning: {cam} read timed out, reusing last frame.", file=sys.stderr)
                    continue
                # ZedCamera returns RGB; this preview writes through cv2, which expects BGR
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if cam is zed else frame
            merged = merge(frames, args.height)

            if writer is None:
                h, w = merged.shape[:2]
                writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                if not writer.isOpened():
                    sys.exit(f"Could not open {output} for writing.")
                print(f"Recording to {output} ({w}x{h} @ {args.fps}fps)")
                print("Press Ctrl+C to stop." if not args.seconds else f"Recording for {args.seconds:.0f}s.")

            writer.write(merged)
            n += 1
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        dt = time.time() - t_start
        if writer is not None:
            writer.release()
        webcam.disconnect()
        zed.disconnect()
        actual_fps = n / dt if dt else 0.0
        print(f"\nWrote {n} frames ({actual_fps:.1f} fps) over {dt:.1f}s to {output}")


if __name__ == "__main__":
    main()
