"""Fit the fixed camera's grid on an EMPTY board from rough corners (a1 h1 h8 a8, pixels), snap to the square corners,
save data/ext_corners.json + a locked copy + the reference frame for the drift check, and draw a check picture.

  python -m chess_player.calib.fit_grid --rough "886,258 378,230 250,572 978,615"   [--url http://127.0.0.1:8771/shot]
"""
import argparse, json, urllib.request

import cv2, numpy as np

from chess_player import config as C
from chess_player.vision import ext_camera as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rough", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8771/shot")
    a = ap.parse_args()
    rough = [tuple(map(float, p.split(","))) for p in a.rough.split()]
    import time as _t

    frames = []
    for _ in range(7):  # median of a few frames: a dark room makes single frames noisy
        frames.append(cv2.imdecode(np.frombuffer(urllib.request.urlopen(a.url).read(), np.uint8), 1))
        _t.sleep(0.15)
    f = np.median(np.stack(frames), axis=0).astype(np.uint8)
    px, n = E.refine_corners(f, rough, iters=3)
    print(f"snapped to {n} square corners; moved {np.round(np.array(px) - np.array(rough), 1).tolist()}")
    data = {"px": px, "auto": {"empty_board": True, "refined_with_corners": n}}
    lock = C.DATA / "ext_corners.locked-wooden.json"
    if lock.exists():
        lock.chmod(0o644)
    for p in (C.EXT_CORNERS, lock):
        p.write_text(json.dumps(data, indent=2))
    lock.chmod(0o444)
    cv2.imwrite(str(C.DATA / "ext_ref_frame.jpg"), f)
    B = 8
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [B, 0], [B, B], [0, B]]), np.float32(px))
    for i in range(B + 1):
        for p, q in (((i, 0), (i, B)), ((0, i), (B, i))):
            s = cv2.perspectiveTransform(np.float32([[p, q]]), Hm)[0].astype(int)
            cv2.line(f, tuple(s[0]), tuple(s[1]), (0, 255, 0), 1)
    cv2.imwrite(str(C.DATA / "ext_corners_check.jpg"), f)
    print("saved", C.EXT_CORNERS, "and", lock)


if __name__ == "__main__":
    main()
