"""Wrist camera capture (OpenCV, V4L2). Frames are BGR, CAM_W x CAM_H."""
import time

import cv2
import numpy as np

from chess_player import config as C


def is_device(src):
    """A local camera: an integer index (Windows) or a /dev path (Linux). Anything else is a stream URL."""
    return isinstance(src, int) or str(src).startswith("/dev")


def open_capture(src):
    """cv2.VideoCapture for a local camera (DirectShow on Windows, V4L2 on Linux) or a stream URL."""
    import sys

    if isinstance(src, int) or str(src).isdigit():
        return cv2.VideoCapture(int(src), cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2)
    if str(src).startswith("/dev"):
        return cv2.VideoCapture(str(src), cv2.CAP_V4L2)
    return cv2.VideoCapture(str(src))


class Camera:
    """The gripper camera. If wrist_stream.py is running it owns the device, so frames come from its local
    stream instead (same 640x480 frames)."""

    def __init__(self, path=C.CAMERA):
        self.cap = open_capture(path)
        ok = self.cap.isOpened() and self.cap.read()[0]
        if ok:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAM_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_H)
            self.cap.set(cv2.CAP_PROP_FPS, C.CAM_FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(10):  # auto exposure settles
                self.cap.read()
            return
        self.cap.release()
        from chess_player.vision.ext_camera import ExtCamera

        try:
            self.stream = ExtCamera(C.WRIST_STREAM_URL, white_balance=False)
        except RuntimeError:
            raise RuntimeError(f"can't open camera {path}, and no wrist_stream.py at {C.WRIST_STREAM_URL}")
        self.cap = None

    def grab(self, settle_s=0.3, flush=5) -> np.ndarray:
        """A fresh frame: wait for the arm to stop shaking, then drop buffered frames."""
        time.sleep(settle_s)
        if self.cap is None:
            time.sleep(0.15)  # let a frame captured after the settle time arrive
            return self.stream.grab()
        for _ in range(flush):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return frame

    def grab_median(self, n=3) -> np.ndarray:
        """Median of a few frames: less sensor noise for change detection."""
        frames = [self.grab(settle_s=0.05 if i else 0.3) for i in range(n)]
        return np.median(np.stack(frames), axis=0).astype(np.uint8)

    def close(self):
        if self.cap is None:
            self.stream.close()
        else:
            self.cap.release()
