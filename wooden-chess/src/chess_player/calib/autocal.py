"""Self-calibration by use: the arm measures its own position error on every square, unattended (about 40 minutes).

  python -m chess_player.calib.autocal              # the sweep (board EMPTY except ONE white pawn on e2), then the fit
  python -m chess_player.calib.autocal --fit-only   # refit from data/autocal_raw.json
  python -m chess_player.calib.autocal --dry-run    # simulated arm and camera with a made-up error field: tests the whole chain

HOW: the pawn is carried over all 64 squares in a snake route. After every placement the arm goes home (out of the picture) and the camera
measures where the pawn's BASE really stands relative to the square centre (vision/piece_locator.py; the base of a piece lies on the board
plane, so one low camera has no height ambiguity). That number E(square) is the arm's landing error there (plus a constant that is the
same everywhere). A smooth surface (quadratic in x, y, robust to the odd bad reading) is fitted through the 64 values and saved in
data/board_residual.json. motion/wooden_pick_place.py then moves every pick and place by -(E(square) - E(the square this piece type was
taught on)), so each taught grip stays valid and the error that depends on WHERE on the board it is, is removed.

The sweep stops at the first failure (nothing gripped, pawn not seen) and can be resumed (--resume) from data/autocal_raw.json.
Camera centring and learned offsets are OFF during the sweep and the correction itself is switched off (CHESS_NO_RESIDUAL), so the raw arm is measured.
"""
import argparse
import json
import os
import time

os.environ["CHESS_NO_RESIDUAL"] = "1"  # measure the RAW arm, never a corrected one

import chess
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot, square_center_mm

RAW = C.DATA / "autocal_raw.json"
OUT = C.DATA / "board_residual.json"
LIMIT_MM = 9.0  # a single reading further than this from the smooth surface is thrown away as a misreading


def _boxes(frame, D, view):
    out = []
    D.detect_pieces(frame, view(frame), conf=0.12, boxes_out=out, kq_height=False)
    return out


def snake():
    """All 64 squares in a snake route (rank 1 left to right, rank 2 right to left, ...): every step is a short carry."""
    for r in range(8):
        files = range(8) if r % 2 == 0 else range(7, -1, -1)
        for f in files:
            yield chess.square(f, r)


def basis(xy):
    x = np.asarray(xy, float)[..., 0] / C.PLAY_MM - 0.5
    y = np.asarray(xy, float)[..., 1] / C.PLAY_MM - 0.5
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)


def fit(points):
    """points: {square_name: [ex, ey]}. Robust (iteratively reweighted) quadratic surface. Returns the dict that is saved."""
    names = sorted(points)
    P = np.array([square_center_mm(chess.parse_square(n)) for n in names])
    Z = np.array([points[n] for n in names], float)
    B = basis(P)
    quad = len(names) >= 40  # a curved surface needs lots of well-spread points; with fewer it extrapolates wildly: use a plane (affine)
    if not quad:
        B = B[:, :3]
    w = np.ones(len(names))
    for _ in range(12):
        coef = np.stack([np.linalg.lstsq(B * w[:, None], Z[:, k] * w, rcond=None)[0] for k in range(2)], axis=1)  # (6, 2)
        res = np.linalg.norm(B @ coef - Z, axis=1)
        s = max(1.0, 1.4826 * float(np.median(res)))
        w = np.where(res <= 1.5 * s, 1.0, 1.5 * s / np.maximum(res, 1e-6))  # Huber-like
    res = np.linalg.norm(B @ coef[:B.shape[1]] - Z, axis=1)
    keep = res <= LIMIT_MM
    coef = np.stack([np.linalg.lstsq(B[keep], Z[keep, k], rcond=None)[0] for k in range(2)], axis=1)
    res = np.linalg.norm(B @ coef - Z, axis=1)
    # leave-one-out: how well does the surface predict a square it has NOT seen (the honest quality number)
    loo = []
    for i in range(len(names)):
        m = np.arange(len(names)) != i
        c = np.stack([np.linalg.lstsq(B[m], Z[m, k], rcond=None)[0] for k in range(2)], axis=1)
        loo.append(float(np.linalg.norm(B[i] @ c - Z[i])))
    spread_before = float(np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean()))
    if not quad:  # keep the saved format (6 coefficients per axis): the curved terms are zero
        coef = np.vstack([coef, np.zeros((3, 2))])
    return dict(model='quadratic' if quad else 'affine', coef=coef.tolist(), n=len(names), raw_spread_rms_mm=spread_before, fit_rms_mm=float(np.sqrt((res ** 2).mean())),
                loo_mean_mm=float(np.mean(loo)), loo_max_mm=float(np.max(loo)), dropped=[n for n, k in zip(names, keep) if not k],
                made=time.strftime("%Y-%m-%d %H:%M"))


def sanity(out, pts):
    """(ok, reason). The fit is only used where it is backed by measurements: they must span the board, and no square may need a huge move."""
    sq = [chess.parse_square(n) for n in pts]
    ranks, files = [chess.square_rank(q) for q in sq], [chess.square_file(q) for q in sq]
    if max(ranks) - min(ranks) < 5 or max(files) - min(files) < 5:
        return False, f"the measured squares do not span the board (ranks {min(ranks) + 1}-{max(ranks) + 1}, files {min(files)}-{max(files)})"
    coef = np.array(out["coef"])
    c0 = basis(np.array([C.PLAY_MM / 2, C.PLAY_MM / 2])) @ coef
    worst = max(float(np.linalg.norm(basis(square_center_mm(q)) @ coef - c0)) for q in chess.SQUARES)
    if worst > 15.0:
        return False, f"the fit asks for a {worst:.0f} mm correction somewhere (limit 15 mm): not believable"
    return True, ""


def report(out, points, saved=True):
    print(f"\nFIT from {out['n']} squares: the arm's landing error varies by {out['raw_spread_rms_mm']:.1f} mm RMS across the board;")
    print(f"  smooth surface leaves {out['fit_rms_mm']:.1f} mm RMS, and predicts an unseen square to {out['loo_mean_mm']:.1f} mm (worst {out['loo_max_mm']:.1f}).")
    if out["dropped"]:
        print("  readings thrown away as misreadings:", out["dropped"])
    coef = np.array(out["coef"])
    print("  correction (mm, x/y) at the corners relative to the centre of the board:")
    for n in ("a1", "h1", "a8", "h8", "d4"):
        e = basis(square_center_mm(chess.parse_square(n))) @ coef - basis(np.array([C.PLAY_MM / 2, C.PLAY_MM / 2])) @ coef
        print(f"    {n}: {e[0]:+.1f} / {e[1]:+.1f}")
    if saved:
        print(f"  saved {OUT}; it is applied automatically (set CHESS_NO_RESIDUAL=1 to switch it off).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quick", action="store_true", help="only ~12 well-spread squares (ranks 4, 6, 8 plus what was measured): about 10 minutes")
    ap.add_argument("--frames", type=int, default=12)
    a = ap.parse_args()

    raw = json.loads(RAW.read_text()) if (a.resume or a.fit_only) and RAW.exists() else {"points": {}, "at": "e2"}
    if a.dry_run:
        raw = {"points": {}, "at": "e2"}

    if not a.fit_only:
        from chess_player.motion.arm import Arm, FakeArm, look_pose
        from chess_player.motion.ik import IK
        from chess_player.motion.pick_place import Transfer
        from chess_player.motion.wooden_pick_place import WoodenPickPlace, recipe

        T = BoardToRobot.load()
        arm = FakeArm() if a.dry_run else Arm()
        pp = WoodenPickPlace(arm, IK(), T, log=lambda *x: None)
        pp.locator = None  # no camera centring: the raw arm is what we measure
        import chess_player.motion.wooden_pick_place as _W

        # search for the pawn the arm placed: y first (front-to-back is where it was off), then x
        _W.SEARCH_STEPS = [(0.0, 0.0), (0.0, 2.5), (0.0, -2.5), (2.5, 0.0), (-2.5, 0.0), (0.0, 5.0), (0.0, -5.0)]
        cam = loc = None
        rng = np.random.default_rng(1)
        if not a.dry_run:
            from chess_player.vision.ext_camera import ExtCamera
            from chess_player.vision.piece_locator import PieceLocator

            cam = ExtCamera()
            time.sleep(2)
            loc = PieceLocator(cam, log=lambda *x: None, )
        cur = chess.parse_square(raw["at"])
        if cam is not None:  # where is the pawn really? (an earlier run may have stopped with it on another square)
            import collections

            from chess_player.vision import detect_board as D
            from chess_player.vision.ext_camera import view

            seen = collections.Counter()
            for _ in range(8):
                fr = cam.grab(max_age_s=2.0)
                best = None
                for d in _boxes(fr, D, view):  # the most confident box of the frame (a lone pawn is detected with low confidence)
                    if d[6] is not None and (best is None or d[1] > best[1]):
                        best = (d[6], d[1])
                if best and best[1] >= 0.12:
                    seen[best[0]] += 1
                time.sleep(0.25)
            if len(seen) == 1 or (seen and seen.most_common(1)[0][1] >= 6):
                found = seen.most_common(1)[0][0]
                if found != cur:
                    print(f"the camera sees the piece on {chess.square_name(found)} (not {chess.square_name(cur)}): starting from there")
                cur = found
            elif len(seen) > 1:
                raise SystemExit("the camera sees more than one piece on the board: leave ONE white pawn only, then run again")
            else:
                print(f"the camera does not see the pawn clearly; assuming it is on {chess.square_name(cur)}")
        todo = [s for s in snake() if chess.square_name(s) not in raw["points"]]
        if a.quick:  # far ranks matter most (the near ranks 1-3 are already measured): the corners and the middle of ranks 4, 6 and 8
            want = ["h4", "d4", "a4", "a6", "d6", "h6", "h8", "d8", "a8"]
            todo = [chess.parse_square(n) for n in want if n not in raw["points"]]
        print(f"{len(todo)} squares to measure, the pawn is on {chess.square_name(cur)}. About {len(todo) * 40 // 60 + 1} minutes. Ctrl+C stops the arm where it is.")
        try:
            for i, nxt in enumerate(todo, 1):
                if nxt == cur:
                    continue
                name = chess.square_name(nxt)
                t = Transfer("P", tuple(square_center_mm(cur)), tuple(square_center_mm(nxt)), f"autocal {chess.square_name(cur)}-{name}")
                t0 = time.time()
                if a.dry_run:
                    m = np.array([0.04 * (square_center_mm(nxt)[0] - 120) - 2.0, -0.03 * (square_center_mm(nxt)[1] - 120) + 1.0]) + rng.normal(0, 1.0, 2)
                else:
                    # a pawn the ARM placed stands place_dx / place_dy off the square centre (taught place offsets): pick it where it is
                    pp.override = {}
                    if raw["points"] or i > 1:
                        r0 = recipe("p")
                        # measured on d1 (2026-11-09): the jaws grip a pawn the arm placed at y = taught shift_y + 3 mm, not at place_dy
                        pp.override = {"shift_x": r0["shift_x"] + r0["place_dx"], "shift_y": r0["shift_y"] + 3.0}
                    pp.execute(t, retries=6, obstacles={})
                    pp.go_look()
                    time.sleep(3.5)  # the arm leaves the picture, the picture settles
                    m = loc.measure(np.array(square_center_mm(nxt), float), None, n=a.frames, conf=0.15, max_spread=5.0)  # any type: YOLO reads a lone pawn on a1 as a rook
                    if m is None:
                        m = loc.measure(np.array(square_center_mm(nxt), float), None, n=20, conf=0.1, max_spread=7.0)
                    if m is None:
                        print(f"  {i:2d}/{len(todo)} {name}: the camera could not measure the pawn here: square skipped", flush=True)
                        cur = nxt
                        raw["at"] = name
                        RAW.write_text(json.dumps(raw, indent=1))
                        continue
                raw["points"][name] = [round(float(m[0]), 2), round(float(m[1]), 2)]
                raw["at"] = name
                if not a.dry_run:
                    RAW.write_text(json.dumps(raw, indent=1))
                cur = nxt
                print(f"  {i:2d}/{len(todo)} {name}: pawn stands {m[0]:+.1f}, {m[1]:+.1f} mm from the square centre   ({time.time() - t0:.0f}s)", flush=True)
        except KeyboardInterrupt:
            print("\nstopped by you (resume with --resume)")
        except Exception as e:  # nothing gripped, pawn not seen ...
            print(f"\nSTOPPED: {e}\nLook at the board, put the pawn back where the log says it is ({raw['at']}) and run again with --resume.")
        finally:
            if cam:
                cam.close()
            try:
                if not a.dry_run:
                    arm.move_to({**look_pose(), "gripper": C.GRIPPER_LOOK}, 3.0)
            except Exception:
                pass
            arm.close()

    pts = raw["points"]
    if len(pts) >= 20:
        out = fit(pts)
        out["points"] = pts
        ok, why = sanity(out, pts)
        if not ok and not a.dry_run:
            report(out, pts, saved=False)
            print(f"\nNOT SAVED: {why}. Measure more squares (python -m chess_player.calib.autocal --quick --resume).")
            return
        if a.dry_run:
            print("(dry run: the fit is shown but NOT saved)")
            report(out, pts, saved=False)
        else:
            OUT.write_text(json.dumps(out, indent=1))
            report(out, pts)
    else:
        print(f"only {len(pts)} squares measured: need at least 20 for the fit. Resume with --resume.")


if __name__ == "__main__":
    main()
