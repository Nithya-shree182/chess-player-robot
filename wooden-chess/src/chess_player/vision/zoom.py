"""'Zoom in' on unclear pieces with the phone camera.

The live stream is 1280x720: a piece is only ~50 px across there. When a square's reading is unsure (low YOLO
confidence), the phone's IP Webcam app is asked for a FULL-RESOLUTION still (4000x2256, ~3x more detail per square),
the board is read again on that still, and the more confident reading wins for the unsure squares. This is better than
the app's own zoom, which only zooms around the centre of the picture and would change the calibrated view.

  python -m chess_player.vision.zoom            # fetch a hi-res still and print what is on the board

The hi-res still has its own corner calibration (data/ext_corners_hires.json: its field of view differs slightly from
the stream's); refit both whenever the phone moves.
"""
import json
import threading
import time
import urllib.request

import cv2
import numpy as np

from chess_player import config as C


def phone_base():
    """http://ip:port of the phone's IP Webcam app, from EXT_CAMERA_URL; None if the camera is not a phone stream."""
    u = str(C.EXT_CAMERA_URL)
    if not u.startswith("http"):
        return None
    return u.split("/video")[0].split("/videofeed")[0].rstrip("/")


def available():
    return C.ZOOM_ENABLED and phone_base() is not None and C.HIRES_CORNERS.exists()


class HiResStill:
    def __init__(self, base=None):
        self.base = base or phone_base()
        if self.base is None:
            raise RuntimeError("the camera is not a phone stream")
        self._wb = None
        self._set_size()

    def _get(self, path, timeout=10):
        return urllib.request.urlopen(self.base + path, timeout=timeout).read()

    def _set_size(self):
        try:  # photo size is a phone-app setting; the video stream keeps its own size
            self._get(f"/settings/photo_size?set={C.HIRES_PHOTO_SIZE}", 6)
        except Exception:
            pass  # the app keeps its last value; grab() still works

    def grab(self):
        """Full-resolution BGR still, white-balanced like the stream frames."""
        data = self._get("/photo.jpg", 15)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("the phone sent no picture")
        if C.EXT_WHITE_BALANCE:
            from chess_player.vision.ext_camera import WhiteBalance

            self._wb = self._wb or WhiteBalance(every=1, smooth=1.0)
            img = self._wb(img)
        return img


def merge_zoom(dets, hi, trigger=None):
    """dets/hi: {square: (symbol, conf)}. For the unsure squares of `dets`, take the hi-res reading when it is more
    confident (or when the stream saw nothing there but the still sees a piece). Returns (merged, [changes])."""
    trigger = C.ZOOM_TRIGGER_CONF if trigger is None else trigger
    out, changes = dict(dets), []
    for sq, (sym, conf) in dets.items():
        if conf >= trigger:
            continue
        h = hi.get(sq)
        if h is not None and h[1] > conf:
            if h[0] != sym:
                changes.append((sq, sym, conf, h[0], h[1]))
            out[sq] = h
    return out, changes


class Zoom:
    """Background hi-res re-reader. refine(dets) is cheap when nothing is unsure."""

    def __init__(self, log=print):
        self.log = log
        self.still = HiResStill()
        self.lock = threading.Lock()
        self.result = (0.0, {})  # (time, {square: (symbol, conf)}) from the last hi-res still
        self._busy = False
        self._t_last = 0.0

    def _read_once(self):
        from chess_player.vision import detect_board as D
        from chess_player.vision.aruco_homography import corners_view

        t0 = time.time()
        img = self.still.grab()
        view = corners_view(img, C.HIRES_CORNERS)
        hi = D.detect_pieces(img, view, conf=0.25, imgsz=C.ZOOM_IMGSZ)
        with self.lock:
            self.result = (time.time(), hi)
        return time.time() - t0

    def _bg(self):
        try:
            self._read_once()
        except Exception as e:
            self.log(f"zoom: hi-res read failed ({type(e).__name__}: {e})")
        finally:
            self._busy = False

    def request(self):
        """Start a hi-res read in the background (at most one at a time, at most every ZOOM_MIN_GAP_S seconds)."""
        if self._busy or time.time() - self._t_last < C.ZOOM_MIN_GAP_S:
            return
        self._busy, self._t_last = True, time.time()
        threading.Thread(target=self._bg, daemon=True).start()

    def refine(self, dets, wait=False, max_age=None):
        low = [sq for sq, (_, c) in dets.items() if c < C.ZOOM_TRIGGER_CONF]
        if not low:
            return dets
        max_age = C.ZOOM_MAX_AGE_S if max_age is None else max_age
        if wait:  # a decision depends on it: take a fresh still now
            if time.time() - self.result[0] > max_age:
                try:
                    secs = self._read_once()
                    self._t_last = time.time()
                except Exception as e:
                    self.log(f"zoom: hi-res read failed ({type(e).__name__}: {e})")
                    return dets
        else:
            if time.time() - self.result[0] > max_age:
                self.request()
        with self.lock:
            t, hi = self.result
        if time.time() - t > max_age * 2:
            return dets
        out, changes = merge_zoom(dets, hi)
        for sq, s0, c0, s1, c1 in changes:
            import chess

            self.log(f"zoom: {chess.square_name(sq)} {s0} ({c0:.2f}) -> {s1} ({c1:.2f}) at full resolution")
        return out


def main():
    from chess_player.vision import detect_board as D
    from chess_player.vision.aruco_homography import corners_view

    still = HiResStill()
    t0 = time.time()
    img = still.grab()
    t1 = time.time()
    view = corners_view(img, C.HIRES_CORNERS)
    dets = D.detect_pieces(img, view, conf=0.25, imgsz=C.ZOOM_IMGSZ)
    print(f"still {img.shape[1]}x{img.shape[0]} fetched in {t1 - t0:.1f} s, read in {time.time() - t1:.1f} s")
    print(D.render(dets))
    unsure = {__import__('chess').square_name(sq): round(c, 2) for sq, (s, c) in dets.items() if c < C.ZOOM_TRIGGER_CONF}
    print("unsure squares:", unsure or "none")


if __name__ == "__main__":
    main()
