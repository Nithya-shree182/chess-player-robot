"""Fit the board's playing-area corners in the phone camera (live stream AND the full-resolution still).

  python -m chess_player.calib.fit_phone_corners --rough "208,80 176,489 598,521 628,120"
  python -m chess_player.calib.fit_phone_corners --rough "..." --stream shot.jpg --still photo.jpg --dry

Run it with the board EMPTY (no pieces) and the phone mounted for good: whenever the phone or the board moves, redo it.
--rough = four points BY EYE in the 1280x720 stream picture, in this order: a1 corner, h1 corner, h8 corner, a8 corner (the
outer corner of each corner square of the playing area). They only need to be within ~10 px: the fit then snaps to the 35
inner square corners (sub-pixel) and the corner squares' outer corners follow. Writes data/ext_corners.json (stream) and
data/ext_corners_hires.json (still); nothing is saved unless BOTH fits find >= 30 inner corners and pass the colour check
(a1 must be a DARK square), so a bad frame cannot overwrite a good calibration. --dry fits and checks but saves nothing.
Overlays for a quick look: data/ext_corners_check_stream.jpg / _hires.jpg.
"""
import argparse
import json
import shutil
import urllib.request

import cv2
import numpy as np

from chess_player import config as C
from chess_player.vision import ext_camera as E
from chess_player.vision.aruco_homography import corners_view, draw_overlay
from chess_player.vision.zoom import HiResStill, phone_base


def fit(img, rough):
    px = [list(map(float, p)) for p in rough]
    used = 0
    for _ in range(5):
        px, used = E.refine_corners(img, px, iters=3)
    return px, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rough", required=True, help='4 points "x,y x,y x,y x,y" in the 1280x720 stream picture: a1 h1 h8 a8')
    ap.add_argument("--stream", help="stream-size image file (default: fetch /shot.jpg from the phone)")
    ap.add_argument("--still", help="full-resolution image file (default: fetch /photo.jpg from the phone)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    rough = [tuple(float(v) for v in p.split(",")) for p in a.rough.split()]
    if len(rough) != 4:
        raise SystemExit("need exactly four rough points")
    base = phone_base()
    stream = cv2.imread(a.stream) if a.stream else cv2.imdecode(
        np.frombuffer(urllib.request.urlopen(base + "/shot.jpg", timeout=10).read(), np.uint8), cv2.IMREAD_COLOR)
    still = cv2.imread(a.still) if a.still else HiResStill().grab()
    sc = still.shape[1] / stream.shape[1]

    px_s, used_s = fit(stream, rough)
    px_h, used_h = fit(still, [(x * sc, y * sc) for x, y in px_s])
    print(f"stream: {used_s}/35 inner corners {np.round(px_s, 1).tolist()}")
    print(f"still : {used_h}/35 inner corners {np.round(px_h, 1).tolist()}")
    ok_all = True
    for name, img, px, used in (("stream", stream, px_s, used_s), ("hires", still, px_h, used_h)):
        tmp = C.DATA / f"_tmp_corners_{name}.json"
        tmp.write_text(json.dumps({"px": px}))
        v = corners_view(img, tmp)
        ok, msg = E.colour_check(img, v)
        tmp.unlink()
        print(f"{name}: {msg}")
        out = draw_overlay(img, v)
        if name == "hires":
            out = cv2.resize(out, (1280, int(1280 * out.shape[0] / out.shape[1])))
        cv2.imwrite(str(C.DATA / f"ext_corners_check_{name}.jpg"), out)
        ok_all &= bool(ok) and used >= 30
    if not ok_all:
        raise SystemExit("NOT saved: a check failed (pieces on the board? wrong rough corners? a1 not dark?)")
    if a.dry:
        print("dry run: checks passed, nothing saved")
        return
    for f in (C.EXT_CORNERS, C.HIRES_CORNERS):
        if f.exists():
            shutil.copy(f, f.with_name(f.stem + ".prev.json"))
    C.EXT_CORNERS.write_text(json.dumps({"px": px_s, "url": str(C.EXT_CAMERA_URL),
                                         "auto": {"empty_board": True, "refined_with_corners": used_s}}, indent=2))
    C.HIRES_CORNERS.write_text(json.dumps({"px": px_h, "size": [still.shape[1], still.shape[0]]}, indent=2))
    print("saved", C.EXT_CORNERS, "and", C.HIRES_CORNERS, "(previous ones kept as *.prev.json)")


if __name__ == "__main__":
    main()
