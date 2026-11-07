"""Persistent camera window for the fixed board camera. Draws the LOCKED grid (data/ext_corners.locked-wooden.json) on the live
picture, shows how far the scene has drifted from the reference frame (ECC, same check as capture_safe) and any ArUco markers it
sees. Also re-serves the stream at http://127.0.0.1:8771/video (and /shot) so the tools can read the camera while it is open.

  source scripts/env.sh && python scripts/livecam.py        (leave it running; close the window to stop)
"""
import json, pathlib, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2, numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from chess_player import config as C
from chess_player.vision import capture_safe as CS

DEV = "/dev/v4l/by-id/usb-Ausha_Webcam_Ausha_Webcam-video-index0"
CORNERS = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else C.DATA / "ext_corners.locked-wooden.json"
WARN_PX = 8.0  # the grid FOLLOWS slow camera creep (see below); only a jump bigger than this needs a refit
LIVE = C.EXT_CORNERS  # what every tool reads: the locked grid moved by the measured shift
latest = {"jpg": None, "raw": None}
lock = threading.Lock()


def grabber():
    CS.set_brightness(95)
    ema = None
    while True:
        cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(3, 1280); cap.set(4, 720)  # (1080p gave corrupt-frame errors on this USB link)
        fails = 0
        while fails < 30:
            ok, f = cap.read()
            if not ok:
                fails += 1; time.sleep(0.05); continue
            fails = 0
            if f.shape[1] != 1280:
                f = cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA)
            # temporal denoise: a dark room makes the sensor noisy. Still pixels are averaged over frames, pixels that changed
            # a lot (the arm, a hand, a piece being moved) are taken as they are.
            f32 = f.astype(np.float32)
            if ema is None or ema.shape != f32.shape:
                ema = f32
            else:
                a = np.where(np.abs(f32 - ema).max(axis=2, keepdims=True) > 28, 1.0, 0.35)
                ema = ema * (1 - a) + f32 * a
            f = ema.astype(np.uint8)
            ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with lock:
                latest["raw"] = f; latest["jpg"] = jpg.tobytes()
            time.sleep(0.02)
        cap.release(); time.sleep(1)  # camera unplugged: keep retrying


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/shot"):
            with lock: j = latest["jpg"]
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.end_headers(); self.wfile.write(j or b""); return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
        try:
            while True:
                with lock: j = latest["jpg"]
                if j:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j) + j + b"\r\n")
                time.sleep(0.05)
        except Exception:
            pass


threading.Thread(target=grabber, daemon=True).start()
threading.Thread(target=lambda: ThreadingHTTPServer(("127.0.0.1", 8771), H).serve_forever(), daemon=True).start()

fig, ax = plt.subplots(num="board camera (live, kept open)")
fig.subplots_adjust(0, 0, 1, 1); ax.axis("off")
im = None
drift = (0.0, 1.0, 0.0)  # shift px, correlation, time measured
dxy = (0.0, 0.0)
locked = json.loads(CORNERS.read_text())
aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.ARUCO_DICT)))
B = 8
while plt.fignum_exists(fig.number):
    with lock: f = None if latest["raw"] is None else latest["raw"].copy()
    if f is None:
        plt.pause(0.1); continue
    if time.time() - drift[2] > 1.0:  # the check is cheap but not needed every frame
        try:
            dx, dy, cc = CS.grid_shift(f)
            drift = (float(np.hypot(dx, dy)), cc, time.time())
            if cc > 0.9 and drift[0] < WARN_PX:  # trustworthy: the grid follows the camera, and tools get the moved grid
                dxy = (dx, dy)
                moved_px = [[x + dx, y + dy] for x, y in locked["px"]]
                tmp = LIVE.with_suffix(".tmp")
                tmp.write_text(json.dumps({"px": moved_px, "auto": {**locked.get("auto", {}), "tracked_shift_px": [dx, dy]}}, indent=2))
                tmp.replace(LIVE)
        except Exception:
            pass
    moved = drift[0] >= WARN_PX or drift[1] < 0.7
    col = (0, 0, 255) if moved else (0, 255, 0)
    try:
        px = np.float32(locked["px"]) + np.float32(dxy)
        Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [B, 0], [B, B], [0, B]]), px)
        for i in range(B + 1):
            for a, b in (((i, 0), (i, B)), ((0, i), (B, i))):
                p = cv2.perspectiveTransform(np.float32([[a, b]]), Hm)[0].astype(int)
                cv2.line(f, tuple(p[0]), tuple(p[1]), col, 1)
        for fi in range(B):  # square names (board frame: x = files a..h, y = ranks 1..8)
            for ri in range(B):
                c = cv2.perspectiveTransform(np.float32([[[fi + 0.5, ri + 0.5]]]), Hm)[0, 0]
                name = "abcdefgh"[fi] + str(ri + 1)
                (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                org = (int(c[0] - tw / 2), int(c[1] + th / 2))
                cv2.putText(f, name, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(f, name, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    except Exception:
        pass
    try:
        mc, ids, _ = aruco.detectMarkers(f)
        if ids is not None: cv2.aruco.drawDetectedMarkers(f, mc, ids)
    except Exception:
        pass
    txt = (f"BOARD/CAMERA MOVED? refit needed (shift {drift[0]:.1f}px, match {drift[1]:.2f})" if moved else f"grid tracking camera: shift {drift[0]:.1f}px, match {drift[1]:.2f}")
    cv2.putText(f, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    if im is None: im = ax.imshow(rgb)
    else: im.set_data(rgb)
    plt.pause(0.05)
