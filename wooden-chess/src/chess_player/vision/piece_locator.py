"""Where does a piece REALLY stand on its square? (fixed camera + YOLO, arm out of view at the look pose)

Hand-placed pieces are 2-5 mm off the square centre and the jaws leave only 2-3 mm of play around a pawn between its neighbours,
so the pick point is corrected by what the camera sees. The camera's base-point reading has a constant bias (the box bottom is the
front edge of the base, YOLO reads 2-5 mm high in y), so only DIFFERENCES are used:

    teaching:  ref  = measured (piece base - square centre) with the piece centred by the human     -> saved as "cam_ref"
    playing:   dev  = measured (piece base - square centre) now  -  ref                              -> pick point += dev

The bias cancels. Readings are medians over several frames and are rejected (returns None) when too few frames saw the piece or
they disagree by more than MAX_SPREAD_MM: then the pick simply runs with the taught numbers, as before.
"""
import time

import numpy as np

from chess_player import config as C

MIN_FRAMES = 4
MAX_SPREAD_MM = 2.5 # frame-to-frame scatter (max - min of the per-frame readings) above this = unreliable
BASE_UP_PX = 16.5  # base point = box bottom-centre moved up this many PIXELS (fitted 2026-11-08: the offset is constant in pixels, so the
# bias is the same on ranks 1, 2, 7 and 8 to ~1.3 mm; the old 0.12 x box height nudge gave a 4.6 mm rank-to-rank difference)
MAX_SEARCH_MM = 18.0  # the detection must be within this of the square centre to count as that piece


class PieceLocator:
    def __init__(self, cam, log=print):
        self.cam = cam  # anything with grab() -> BGR frame (vision.ext_camera.ExtCamera)
        self.log = log

    def _frame_readings(self, frame, centre_xy, sym, conf=0.35):
        from chess_player.vision import detect_board as D
        from chess_player.vision.ext_camera import view

        v = view(frame)
        boxes = []
        D.detect_pieces(frame, v, conf=conf, boxes_out=boxes, kq_height=False)
        best = None
        for (s, conf, x1, y1, x2, y2, _sq, _h, _note) in boxes:
            if sym is not None and s.lower() != sym.lower():
                continue
            base = np.array([(x1 + x2) / 2.0, y2 - BASE_UP_PX])
            mm = np.asarray(v.px_to_mm([base])[0], float)
            d = float(np.linalg.norm(mm - centre_xy))
            if d < MAX_SEARCH_MM and (best is None or d < best[0]):
                best = (d, mm, conf)
        return None if best is None else best[1]

    def measure(self, centre_xy, sym=None, n=8, gap_s=0.3, conf=0.35, max_spread=None):
        """Raw (piece base - centre_xy) in board mm as np.array([dx, dy]), or None when it cannot be trusted."""
        centre_xy = np.asarray(centre_xy, float)
        pts = []
        for i in range(n):
            if i:
                time.sleep(gap_s)
            try:
                p = self._frame_readings(self.cam.grab(max_age_s=2.0), centre_xy, sym, conf)
            except Exception as e:  # stale stream, no model ...
                self.log(f"  locator: {e}")
                return None
            if p is not None:
                pts.append(p - centre_xy)
        if len(pts) < MIN_FRAMES:
            self.log(f"  locator: the piece was found in only {len(pts)} of {n} frames")
            return None
        pts = np.array(pts)
        spread = float(np.max(np.ptp(pts, axis=0)))
        limit = MAX_SPREAD_MM if max_spread is None else max_spread
        if spread > limit:
            self.log(f"  locator: readings scatter {spread:.1f} mm (limit {limit}), not used")
            return None
        return np.median(pts, axis=0)
