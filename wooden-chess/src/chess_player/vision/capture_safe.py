"""Capture dataset shots from the live camera window with a countdown and a drift check.

  python -m chess_player.vision.capture_safe --name pos02 --fen "<FEN>" --shots 5 --gap 4 --countdown 6

Each shot is compared to the reference frame; if the scene moved more than --max-shift px the shot is rejected, so the labels (projected from the locked corners) stay valid. Wave your hands away during the countdown.
"""
import argparse
import json
import random
import subprocess
import time

import chess
import cv2
import numpy as np

from chess_player import config as C
from chess_player.vision.capture_dataset import DS, write_sample
from chess_player.vision.ext_camera import ExtCamera

REF = C.DATA / "ext_ref_frame.jpg"
DEV = "/dev/v4l/by-id/usb-Ausha_Webcam_Ausha_Webcam-video-index0"
BASE_BRIGHTNESS = 95  # the room light is yellow and strong: the camera default (128) blows out the white pieces


def _edges(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


_MASK = np.zeros((720, 1280), np.uint8)
# regions pieces never cover and the arm does not enter when parked: the wall panel's bottom edge, its side edges and the robot's base
# (they move only if the CAMERA moves; the board is taped)
_MASK[225:300, 70:450] = 1
_MASK[225:300, 660:1180] = 1
_MASK[60:300, 1160:1230] = 1
_MASK[180:330, 470:630] = 1


def grid_shift(frame):
    """Translation (dx, dy px) of the scene against the reference frame (data/ext_ref_frame.jpg, taken when the corners were
    fitted), measured on the piece-free board edges by ECC, and the correlation (under 0.7 = too different to trust).
    The grid in this frame is the fitted grid moved by (dx, dy). Returns (dx, dy, correlation)."""
    ref = cv2.imread(str(REF))
    w = np.eye(2, 3, dtype=np.float32)
    try:
        cc, w = cv2.findTransformECC(_edges(ref), _edges(frame), w, cv2.MOTION_TRANSLATION,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4), _MASK, 5)
    except cv2.error:
        return 0.0, 0.0, 0.0
    return float(w[0, 2]), float(w[1, 2]), float(cc)


def grid_drift(frame):
    """How far the scene moved from the reference frame: (shift in px, correlation). See grid_shift."""
    dx, dy, cc = grid_shift(frame)
    return (float(np.hypot(dx, dy)) if cc > 0 else 99.0), cc


def set_brightness(v):
    subprocess.run(["v4l2-ctl", "-d", DEV, "--set-ctrl", f"brightness={int(v)}"], check=False)


def jitter(frame, rng):
    """Software lighting variety (geometry untouched, so the labels stay exact): contrast, gamma, colour tint."""
    f = frame.astype(np.float32) / 255.0
    f = np.clip((f - 0.5) * rng.uniform(0.75, 1.3) + 0.5, 0, 1)
    f = f ** rng.uniform(0.75, 1.35)
    f = np.clip(f * np.array([rng.uniform(0.88, 1.12), rng.uniform(0.92, 1.08), rng.uniform(0.88, 1.12)], np.float32), 0, 1)
    return (f * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fen", default="start")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--gap", type=float, default=2.5)
    ap.add_argument("--countdown", type=int, default=6)
    ap.add_argument("--brightness", help="comma list of camera brightness values to cycle (default 128 only), e.g. 60,100,128,170,210")
    ap.add_argument("--jitter", action="store_true", help="add random contrast/gamma/tint to every shot")
    ap.add_argument("--max-shift", type=float, default=2.5)
    a = ap.parse_args()
    board = chess.Board() if a.fen == "start" else chess.Board(a.fen)
    levels = [int(v) for v in a.brightness.split(",")] if a.brightness else [BASE_BRIGHTNESS]
    rng = np.random.default_rng()
    cam = ExtCamera()
    try:
        for s in range(a.countdown, 0, -1):
            print(f"  hands away... {s}", flush=True)
            time.sleep(1)
        kept = 0
        for i in range(a.shots):
            b = levels[i % len(levels)]
            set_brightness(b)
            time.sleep(max(a.gap, 1.5) if i else 1.5)
            frame = cam.grab()
            drift, cc = grid_drift(frame)
            if drift > a.max_shift or cc < 0.7:
                print(f"  shot {i}: REJECTED (scene moved {drift:.1f} px, match {cc:.2f})", flush=True)
                continue
            if a.jitter:
                frame = jitter(frame, rng)
            stem = f"{a.name}_{kept:02d}"
            n, _ = write_sample(stem, frame, board, "ext")
            (DS / "meta").mkdir(parents=True, exist_ok=True)
            (DS / "meta" / f"{stem}.json").write_text(json.dumps({"fen": board.fen(), "position": a.name, "camera": "ext", "brightness": b}))
            print(f"  {stem}: {n} boxes, brightness {b} (drift {drift:.1f} px, match {cc:.2f})", flush=True)
            kept += 1
        print(f"kept {kept}/{a.shots}; previews in {DS / 'previews'}")
        if kept == 0:
            raise SystemExit(1)
    finally:
        set_brightness(BASE_BRIGHTNESS)
        cam.close()


if __name__ == "__main__":
    main()
