#!/usr/bin/env python3
"""Streams the generic webcam and the ZED/ZED-Mini, stitched side by side, as an
MJPEG stream over HTTP for viewing in VLC or a browser:
    http://<pi-ip>:8090/stream.mjpg
Goes through lerobot's actual camera backends (`OpenCVCamera` and `ZedCamera`)
rather than the hand-rolled cv2/py-zed-open-capture path in cameras.py --
useful for sanity-checking the lerobot ZED integration end to end. By default
the ZED is streamed in "stereo" mode, i.e. its full undivided left+right
frame, so the output is [webcam | zed-left | zed-right] with no extra camera
object needed for the second eye. Run inside the project venv:
    source .venv/bin/activate
    python debug/hardware/stream_lerobot_cameras.py
Ctrl+C stops the stream.

See lerobot_cameras.py for camera detection/setup.

Both cameras are polled sequentially by one background thread (read webcam,
read ZED, merge, encode, repeat) rather than one thread per camera -- see the
original stream_cameras.py's module docstring for why: threading each camera
independently measurably starves the higher-bandwidth one under GIL
contention, even though the cameras sit on entirely separate USB controllers
with bandwidth to spare individually.
"""

import argparse
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from lerobot_cameras import open_cameras
from thesis.utils.stream import merge
from thesis.utils.stderr_filter import CORRUPT_JPEG, suppress_stderr_lines

PORT = 8090
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30

STREAM = None


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


class MergedStream:
    """Sequentially polls both cameras in one background thread and keeps the
    latest merged+encoded JPEG ready to serve (see module docstring for why
    this is sequential in one thread rather than one thread per camera)."""

    def __init__(self, webcam, zed, height, fps):
        self.cams = (webcam, zed)
        self.height = height
        self.period = 1 / fps
        self.frames = [None, None]
        self.jpg = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            t0 = time.time()
            for i, cam in enumerate(self.cams):
                try:
                    frame = cam.async_read()
                except TimeoutError:
                    continue
                # ZedCamera returns RGB; this preview writes through cv2, which expects BGR
                self.frames[i] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if cam is self.cams[1] else frame
            if all(f is not None for f in self.frames):
                merged = merge(self.frames, self.height)
                ok, jpg = cv2.imencode(".jpg", merged, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self.lock:
                        self.jpg = jpg.tobytes()
            time.sleep(max(0.0, self.period - (time.time() - t0)))

    def latest_jpg(self):
        with self.lock:
            return self.jpg

    def close(self):
        self.running = False
        self.thread.join(timeout=1)
        for cam in self.cams:
            cam.disconnect()


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/stream.mjpg"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()
        try:
            while True:
                jpg = STREAM.latest_jpg()
                if jpg is not None:
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                time.sleep(STREAM.period)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        pass


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
        help="Which ZED eye(s) to stream: 'left'/'right' for one cropped eye, "
        "'stereo' for the full undivided left+right frame (default: stereo)",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="target capture/stream frame rate")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="per-camera height before stacking")
    parser.add_argument("--port", type=int, default=PORT, help="HTTP port to serve the stream on")
    return parser.parse_args()


def main():
    args = parse_args()
    suppress_stderr_lines([CORRUPT_JPEG])

    webcam, zed = open_cameras(
        args.fps,
        zed_side=args.zed_side,
        opencv_index_or_path=args.opencv_index_or_path,
        zed_serial=args.zed_serial,
    )

    global STREAM
    STREAM = MergedStream(webcam, zed, args.height, args.fps)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), StreamHandler)
    print(f"Merged stream: http://{local_ip()}:{args.port}/stream.mjpg")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping stream...")
        server.shutdown()
        STREAM.close()


if __name__ == "__main__":
    main()
