"""Fit the fixed camera's grid AUTOMATICALLY on an EMPTY board (no clicking, no rough corners).

  python -m chess_player.calib.auto_grid              # needs scripts/livecam.py running; or use scripts/regrid.sh

How: OpenCV finds the 7x7 inner square corners of the empty board in a median of several frames; the 8 possible
labellings are tried and the one with the file axis pointing left and the rank axis pointing down in the picture
(rank 1 at the top = robot side, h-file on the left) is kept. The outer corners follow from the homography, then the
usual sub-pixel snap (vision.ext_camera.refine_corners) and the same save as calib/fit_grid. Refuses to save when
fewer than 49 corners are found, the fit is poor, or the board is not empty-looking. Draws data/ext_corners_check.jpg.
"""
import argparse
import json
import time
import urllib.request

import cv2
import numpy as np

from chess_player import config as C
from chess_player.vision import ext_camera as E


def grab_median(url, n=9, gap=0.15, wait_s=30):
    """Median of n frames. Skips empty answers: right after a restart the camera window has no picture yet."""
    frames, t_end = [], time.time() + wait_s
    while len(frames) < n and time.time() < t_end:
        try:
            buf = np.frombuffer(urllib.request.urlopen(url, timeout=10).read(), np.uint8)
            img = cv2.imdecode(buf, 1) if buf.size else None
        except OSError:
            img = None
        if img is None:
            time.sleep(0.5)
            continue
        frames.append(img)
        time.sleep(gap)
    if len(frames) < n:
        raise SystemExit(f"the camera window gave only {len(frames)} pictures: is scripts/livecam.py running and the camera plugged in?")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def find_inner(frame):
    """(49,2) inner corners in the order OpenCV returns them (a 7x7 grid), or None."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # wood vs cream squares differ in COLOUR much more than in brightness: Lab b / HSV saturation work (tested 6/6)
    chans = [lab[:, :, 2], cv2.GaussianBlur(lab[:, :, 2], (5, 5), 0), cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 2]),
             hsv[:, :, 1], cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)]
    for g in chans:
        ok, pts = cv2.findChessboardCorners(g, (7, 7), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            pts = cv2.cornerSubPix(g, pts, (7, 7), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
            return pts.reshape(-1, 2)
    return None


def find_middle(frame):
    """(21,2) inner corners of the EMPTY middle (the 3 rank lines between ranks 3|4, 4|5, 5|6 x 7 file lines), or None. Works with the
    start position set up: the pieces stand on ranks 1, 2, 7, 8."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    chans = [lab[:, :, 2], cv2.GaussianBlur(lab[:, :, 2], (5, 5), 0), cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 2]),
             hsv[:, :, 1], cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)]
    for g in chans:
        ok, pts = cv2.findChessboardCorners(g, (7, 3), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            pts = cv2.cornerSubPix(g, pts, (7, 7), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
            return pts.reshape(-1, 2)
    return None


def shift_by_search(frame, old_px, rng=44, rows=(3, 4, 5)):
    """Shift (dx, dy px) of the old grid, found by brute force: for every candidate shift, how checkerboard-like is the picture at the
    predicted inner corners of the EMPTY middle (a corner has light-dark-dark-light quadrants)? Works with the pieces on the board, in
    noise. Returns (dx, dy, score_ratio, n_corners): score_ratio = best score / median score over all shifts (a clear peak is > 2.5)."""
    B, S = C.PLAY_MM, C.SQUARE_MM
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [B, 0], [B, B], [0, B]]), np.float32(old_px))
    pts = np.float32([(i * S, j * S) for i in range(1, 8) for j in rows])
    pred = cv2.perspectiveTransform(pts[None], Hm)[0]
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    g = cv2.GaussianBlur(lab[:, :, 2].astype(np.float32), (0, 0), 1.5)
    g = g - cv2.GaussianBlur(g, (0, 0), 12)  # remove slow lighting gradients
    sq = float(np.median(np.linalg.norm(np.diff(pred.reshape(len(rows), 7, 2)[:, :, :], axis=1), axis=2)))
    d = max(3, int(round(sq * 0.28)))  # quadrant centres at +-d px from the corner
    k = max(2, d // 2)
    ii = cv2.integral(g)

    def box(x, y):  # mean of a (2k+1)^2 box centred at (x, y) via the integral image
        x0, y0, x1, y1 = int(x) - k, int(y) - k, int(x) + k + 1, int(y) + k + 1
        if x0 < 0 or y0 < 0 or x1 >= g.shape[1] or y1 >= g.shape[0]:
            return 0.0
        return (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / float((x1 - x0) * (y1 - y0))

    def score(dx, dy):
        t = 0.0
        for x, y in pred:
            x, y = x + dx, y + dy
            t += abs((box(x - d, y - d) + box(x + d, y + d)) - (box(x + d, y - d) + box(x - d, y + d)))
        return t

    grid = {(dx, dy): score(dx, dy) for dx in range(-rng, rng + 1, 2) for dy in range(-rng, rng + 1, 2)}
    (bx, by), best = max(grid.items(), key=lambda kv: kv[1])
    fine = {(bx + fx, by + fy): score(bx + fx, by + fy) for fx in (-1, 0, 1) for fy in (-1, 0, 1)}
    (bx, by), best = max(fine.items(), key=lambda kv: kv[1])
    return float(bx), float(by), best / (float(np.median(list(grid.values()))) + 1e-6), len(pred)


def orient(pts):
    """Try the 8 labellings of the 7x7 grid; return the outer corners [a1,h1,h8,a8] (px) for the right one."""
    idx = np.arange(49).reshape(7, 7)
    cands = []
    for k in range(4):
        m = np.rot90(idx, k)
        cands += [m, m[:, ::-1]]
    sq = np.float32([(i, j) for i in range(1, 8) for j in range(1, 8)])  # board coords in squares (file, rank)
    outer = np.float32([(0, 0), (8, 0), (8, 8), (0, 8)])
    found = []
    for m in cands:
        H, _ = cv2.findHomography(sq, pts[m.reshape(-1)], 0)
        if H is None:
            continue
        a1, h1, h8, a8 = cv2.perspectiveTransform(outer[None], H)[0]
        fa, ra = h1 - a1, a8 - a1  # file axis (a->h) must run mostly LEFT, rank axis (1->8) mostly DOWN in the picture
        if fa[0] < 0 and abs(fa[0]) > 1.5 * abs(fa[1]) and ra[1] > 0 and abs(ra[1]) > abs(ra[0]):
            found.append((H, [a1, h1, h8, a8]))
    if not found:
        raise RuntimeError("no labelling puts rank 1 at the top and the h-file on the left: is the camera on the human's side?")
    H, c = found[0]
    return [list(map(float, p)) for p in c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8771/shot")
    ap.add_argument("--dry", action="store_true", help="find and draw, do not save")
    ap.add_argument("--pieces", action="store_true",
                    help="the pieces are on the board (start position): measure the shift of the old grid from the empty middle ranks 3-6")
    a = ap.parse_args()
    if a.pieces:
        return main_pieces(a)

    pts = None
    for attempt in range(4):  # the camera window averages frames over time: right after a restart it is noisy
        f = grab_median(a.url)
        pts = find_inner(f)
        if pts is not None:
            break
        time.sleep(2)
    if pts is None:
        cv2.imwrite(str(C.DATA / "ext_corners_check.jpg"), f)
        raise SystemExit("could not find the 7x7 inner corners. Is the board EMPTY, fully in view, and lit? (picture: data/ext_corners_check.jpg)")
    rough = orient(pts)
    px, n = E.refine_corners(f, rough, iters=3)
    px = [list(map(float, p)) for p in px]

    # quality: how far the snapped fit is from the 49 detected corners (px), and the square size in the picture
    B = C.PLAY_MM
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [B, 0], [B, B], [0, B]]), np.float32(px))
    S = C.SQUARE_MM
    grid = np.float32([(i * S, j * S) for i in range(1, 8) for j in range(1, 8)])
    proj = cv2.perspectiveTransform(grid[None], Hm)[0]
    d = np.min(np.linalg.norm(proj[:, None] - pts[None], axis=2), axis=1)
    sq_px = np.linalg.norm(np.float32(px[0]) - np.float32(px[1])) / 8
    print(f"inner corners found: 49; snapped to {n}; fit vs detected corners: median {np.median(d):.2f} px, max {d.max():.2f} px; "
          f"square ~{sq_px:.0f} px")
    out = f.copy()
    for i in range(9):
        for p, q in (((i, 0), (i, B // S)), ((0, i), (B // S, i))):
            s = cv2.perspectiveTransform(np.float32([[(p[0] * S, p[1] * S), (q[0] * S, q[1] * S)]]), Hm)[0].astype(int)
            cv2.line(out, tuple(s[0]), tuple(s[1]), (0, 255, 0), 1)
    for k, nm in enumerate(("a1", "h1", "h8", "a8")):
        cv2.circle(out, tuple(int(v) for v in px[k]), 5, (0, 0, 255), -1)
        cv2.putText(out, nm, (int(px[k][0]) + 6, int(px[k][1]) - 6), 0, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(C.DATA / "ext_corners_check.jpg"), out)
    if np.median(d) > 2.5 or d.max() > 8.0 or n < 25:  # a few px of max error is lens distortion; more means a wrong fit
        raise SystemExit(f"NOT saved: the fit is poor (max {d.max():.1f} px, {n} corners). Look at data/ext_corners_check.jpg; "
                         "empty the board / fix the light / re-run.")
    if a.dry:
        print("dry run: not saved (picture: data/ext_corners_check.jpg)")
        return
    data = {"px": px, "auto": {"empty_board": True, "refined_with_corners": n, "method": "auto_grid"}}
    lock = C.DATA / "ext_corners.locked-wooden.json"
    if lock.exists():
        lock.chmod(0o644)
    for p in (C.EXT_CORNERS, lock):
        p.write_text(json.dumps(data, indent=2))
    lock.chmod(0o444)
    cv2.imwrite(str(C.DATA / "ext_ref_frame.jpg"), f)
    print(f"saved {C.EXT_CORNERS} and the locked copy + reference frame. Check picture: {C.DATA / 'ext_corners_check.jpg'}")


def main_pieces(a):
    lock = C.DATA / "ext_corners.locked-wooden.json"
    old = json.loads(lock.read_text())["px"]
    f = grab_median(a.url)
    dx, dy, ratio, n = shift_by_search(f, old)
    print(f"the old grid is off by dx {dx:+.0f} px, dy {dy:+.0f} px (checkerboard peak {ratio:.1f}x the average over all shifts, {n} corners)")
    px = [[x + dx, y + dy] for x, y in old]
    B, S = C.PLAY_MM, C.SQUARE_MM
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [B, 0], [B, B], [0, B]]), np.float32(px))
    out = f.copy()
    for i in range(9):
        for p, q in (((i, 0), (i, B // S)), ((0, i), (B // S, i))):
            s_ = cv2.perspectiveTransform(np.float32([[(p[0] * S, p[1] * S), (q[0] * S, q[1] * S)]]), Hm)[0].astype(int)
            cv2.line(out, tuple(s_[0]), tuple(s_[1]), (0, 255, 0), 1)
    cv2.imwrite(str(C.DATA / "ext_corners_check.jpg"), out)
    if ratio < 2.5:
        raise SystemExit("NOT saved: no clear checkerboard peak (too dark / ranks 3-6 not empty / the camera also turned?). Picture: data/ext_corners_check.jpg")
    if a.dry:
        print("dry run: not saved (picture: data/ext_corners_check.jpg)")
        return
    data = {"px": px, "auto": {"empty_board": False, "method": "auto_grid --pieces", "shifted_px": [dx, dy]}}
    if lock.exists():
        lock.chmod(0o644)
    for p in (C.EXT_CORNERS, lock):
        p.write_text(json.dumps(data, indent=2))
    lock.chmod(0o444)
    cv2.imwrite(str(C.DATA / "ext_ref_frame.jpg"), f)
    print(f"saved {C.EXT_CORNERS} (moved by the measured shift), the locked copy and the reference frame")


if __name__ == "__main__":
    main()
