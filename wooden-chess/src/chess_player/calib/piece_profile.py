"""Width profile of each piece type (width in mm at every height above the base), from the side-view silhouette in the
product listing photo (pawn 25 mm and king 50 mm are printed there and calibrate the scale).

  python -m chess_player.calib.piece_profile <listing image>      writes data/piece_profiles.json

The grasp planner uses it twice: (1) the jaws close on the widest part inside the pad band (~12 mm tall) at the grasp
height, and (2) a neighbour is as wide as its widest part between the fingertip height and its top (the fingers are
~70 mm long). Replace the numbers by measurements of the real pieces whenever you have calipers.
"""
import json
import sys

import cv2
import numpy as np

from chess_player import config as C

OUT = C.DATA / "piece_profiles.json"
# x ranges of the six pieces (pawn, rook, knight, bishop, queen, king) in the listing photo, found by column analysis
SEGS = [(104, 152), (179, 239), (271, 332), (362, 425), (453, 517), (557, 620)]
NAMES = ["p", "r", "n", "b", "q", "k"]
PAD_BAND_MM = 12.0  # vertical size of the jaw pads that touch the piece


def measure(path):
    img = cv2.imread(str(path))
    roi = img[340:540, 60:680]
    dark = (cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) < 90).astype(np.uint8)
    sc = None
    prof = {}
    heights = {}
    for n, (a, b) in zip(NAMES, SEGS):
        sub = dark[:, a - 60 : b - 60]
        ys = np.where(sub.sum(1) > 0)[0]
        heights[n] = (ys.max() - ys.min() + 1, ys.max())
    sc = (heights["k"][0] - heights["p"][0]) / (50.0 - 25.0)  # px per mm from the two printed heights
    for n, (a, b) in zip(NAMES, SEGS):
        sub = dark[:, a - 60 : b - 60]
        hpx, bot = heights[n]
        h = hpx / sc
        w = []
        for z in range(0, int(h) + 1):
            y = int(round(bot - z * sc))
            w.append(round(float(sub[max(0, min(sub.shape[0] - 1, y))].sum() / sc), 1))
        prof[n] = dict(height=round(h, 1), width_by_mm=w)
    return prof, sc


def best_band(widths, height, band=PAD_BAND_MM):
    """Lowest grip height z (finger bottom) whose pad band [z, z+band] has the smallest widest-part, searched over
    z in [4, height - band] (never right at the base flare)."""
    best = None
    for z in range(4, int(height - band) + 1):
        wmax = max(widths[z : z + int(band) + 1])
        if best is None or wmax < best[1] - 0.25:
            best = (z, wmax)
    return best


def main():
    prof, sc = measure(sys.argv[1])
    print(f"scale {sc:.3f} px/mm")
    for n, d in prof.items():
        z, w = best_band(d["width_by_mm"], d["height"])
        d["grip_z"], d["grip_w"] = z, round(w, 1)
        d["base_w"] = round(max(d["width_by_mm"][:5]), 1)
        print(f"{n}: height {d['height']:4.1f}  base {d['base_w']:4.1f}  best pad band starts at z={z:2d} (widest part in the band {w:.1f} mm)")
    OUT.write_text(json.dumps(prof, indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
