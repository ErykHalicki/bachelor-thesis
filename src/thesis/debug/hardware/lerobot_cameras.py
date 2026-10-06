"""Shared camera setup for record_lerobot_cameras.py and stream_lerobot_cameras.py:
opens the generic webcam and the ZED/ZED-Mini through lerobot's own camera
backends (`OpenCVCamera`, `ZedCamera`) rather than the hand-rolled
cv2/py-zed-open-capture path in cameras.py.

Device detection lives in thesis.utils.cameras (not here) so it stays a real,
importable dependency for non-debug code (e.g. scripts/b601/record.py) -- this
file only owns the debug-preview-specific bit: actually connecting the cameras.
Frames are merged for the hconcat preview by thesis.utils.stream.merge,
shared with the preview record.py serves during a recording session.
"""

from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.opencv.configuration_opencv import ColorMode
from lerobot.cameras.zed import ZedCamera, ZedCameraConfig
from thesis.utils.cameras import resolve_devices


def open_cameras(
    fps,
    zed_side="stereo",
    opencv_index_or_path=None,
    zed_serial=None,
    webcam_width=1280,
    webcam_height=720,
    webcam_fps=30,
):
    """Resolves and connects both cameras. The ZED is connected before the webcam
    as extra insurance against the SIGABRT race described in resolve_devices().

    webcam_fps is independent of `fps` (which only controls the ZED): most
    webcams only support one discrete fps per resolution in MJPG mode, so
    forwarding an arbitrary `fps` to the webcam can hard-fail connect() with
    an "OpenCVCamera failed to set fps=..." RuntimeError.

    Returns (webcam, zed), both already connected.
    """
    zed_serial, opencv_index_or_path = resolve_devices(opencv_index_or_path, zed_serial)

    # MJPG rather than cv2's default: uncompressed modes often cannot sustain 720p at
    # the requested fps. warmup_s is raised because some webcams return blank buffers
    # for seconds after a mode change. The backend is pinned because cv2's ANY can pick
    # FFMPEG, which ignores fourcc, size and fps and then hard-fails connect().
    webcam = OpenCVCamera(
        OpenCVCameraConfig(
            index_or_path=opencv_index_or_path,
            fps=webcam_fps,
            width=webcam_width,
            height=webcam_height,
            fourcc="MJPG",
            color_mode=ColorMode.BGR,
            warmup_s=3,
            backend=Cv2Backends.V4L2,
        )
    )
    zed = ZedCamera(ZedCameraConfig(side=zed_side, serial_number=zed_serial, fps=fps))

    print(f"Connecting to {zed} and {webcam}...")
    zed.connect()
    webcam.connect()
    return webcam, zed


