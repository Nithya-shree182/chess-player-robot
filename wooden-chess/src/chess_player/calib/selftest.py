"""Start-up self-test for a new place (about 3 minutes, one command). Prints a PASS / WARN / FAIL table and exits non-zero on any FAIL.

  python -m chess_player.calib.selftest [--no-touch] [--url http://127.0.0.1:8771/shot]
  bash scripts/selftest.sh

Checks, in this order (the arm only moves in the last one, and only over EMPTY squares e3 d6 g4: clear them first):
  1. camera     the camera window answers, the picture is not too dark / flat
  2. grid       the saved camera grid still sits on the board (checkerboard search on the empty middle ranks: works with pieces on)
  3. vision     YOLO reads the board; reports the piece count and whether it is the start position
  4. reach      all 64 squares can be reached by the arm (IK at 30 mm with the calibration on file; no motion)
  5. arm        all six servos answer
  6. touch      the arm touches 3 squares that were NOT used for the calibration; the camera measures how far the tip is from each centre
"""
import argparse
import json
import sys

import chess
import cv2
import numpy as np

from chess_player import config as C

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
rows = []


def rec(name, status, msg):
    rows.append((name, status, msg))
    print(f"[{status}] {name}: {msg}", flush=True)


def check_camera(url):
    from chess_player.calib.auto_grid import grab_median
    try:
        f = grab_median(url, n=3)
    except SystemExit as e:
        rec("camera", FAIL, str(e))
        return None
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    bright, contrast = float(g.mean()), float(g.std())
    if bright < 45 or contrast < 25:
        rec("camera", WARN, f"picture is dark/flat (brightness {bright:.0f}, contrast {contrast:.0f}): add a lamp over the board")
    else:
        rec("camera", OK, f"brightness {bright:.0f}, contrast {contrast:.0f}")
    return f


def check_grid(f):
    from chess_player.calib.auto_grid import shift_by_search
    try:
        old = json.loads((C.DATA / "ext_corners.json").read_text())["px"]
    except (OSError, ValueError, KeyError):
        rec("grid", FAIL, "no camera grid: run scripts/regrid.sh")
        return
    dx, dy, ratio, _n = shift_by_search(f, old, rng=20)
    shift = float(np.hypot(dx, dy))
    if ratio < 2.0:
        rec("grid", WARN, f"no clear checkerboard peak (ratio {ratio:.1f}): ranks 3-6 not empty or too dark; cannot judge the grid")
    elif shift <= 3:
        rec("grid", OK, f"grid is on the board (off by {shift:.1f} px)")
    elif shift <= 6:
        rec("grid", WARN, f"grid is {shift:.1f} px off ({dx:+.0f}, {dy:+.0f}): bash scripts/regrid.sh (or auto_grid --pieces)")
    else:
        rec("grid", FAIL, f"grid is {shift:.1f} px off ({dx:+.0f}, {dy:+.0f}): the camera or board moved: bash scripts/regrid.sh")


def check_vision(f):
    try:
        from chess_player.vision import detect_board as D
        from chess_player.vision.ext_camera import view
        from chess_player.game.state import start_mismatch
        dets = D.detect_pieces(f, view(f), kq_height=False)
        why = start_mismatch(dets)
        if not dets:
            rec("vision", OK, "YOLO sees an empty board")
        elif not why:
            rec("vision", OK, f"{len(dets)} pieces seen: the start position")
        else:
            rec("vision", OK, f"{len(dets)} pieces seen (not the start position: {why[0]})")
    except Exception as e:
        rec("vision", FAIL, f"YOLO could not read the board ({e})")


def check_reach():
    from chess_player.geometry import BoardToRobot, square_center_mm
    from chess_player.motion.arm import look_pose
    from chess_player.motion.ik import IK, solve_or_raise
    try:
        T, ik = BoardToRobot.load(), IK()
    except Exception as e:
        rec("reach", FAIL, f"no board->robot calibration ({e}): run python -m chess_player.calib.camcal")
        return
    look = look_pose()
    q = np.array([look[j] for j in C.ARM_JOINTS])
    q[4] = C.WRIST_ROLL_DEG
    bad = []
    for sq in chess.SQUARES:
        try:
            xy = square_center_mm(sq)
            hi = solve_or_raise(ik, T(*xy, C.Z_SAFE), q, what="hi")
            lo = solve_or_raise(ik, T(*xy, 30.0), hi, C.GRASP_MAX_TILT, what="lo")
        except Exception:
            bad.append(chess.square_name(sq))
    if not bad:
        rec("reach", OK, "all 64 squares reachable")
    elif len(bad) <= 4:
        rec("reach", WARN, f"cannot reach {', '.join(bad)}: move the board closer to the robot or accept")
    else:
        rec("reach", FAIL, f"cannot reach {len(bad)} squares ({', '.join(bad[:8])} ...): the board is too far / on the wrong side")


def check_arm():
    try:
        from chess_player.motion.arm import Arm
        arm = Arm()
        arm.present()
        arm.close()
        rec("arm", OK, "all six servos answer")
    except Exception as e:
        rec("arm", FAIL, f"{str(e)[:150]}  (replug the arm's USB cable, then python src/tools/scan_motors.py)")


def check_touch(url):
    from chess_player.calib import camcal as K
    from chess_player.geometry import BoardToRobot
    from chess_player.motion.arm import Arm
    from chess_player.motion.ik import IK
    from chess_player.vision.ext_camera import view
    arm = None
    try:
        v = view(K.grab(url, n=3))
        arm = Arm()
        pts = K.measure_points(arm, IK(), BoardToRobot.load(), v, K.CHECK_SQUARES, url, tag="selftest", log=lambda m: None)
    except Exception as e:
        rec("touch", FAIL, f"could not run ({str(e)[:150]})")
        return
    finally:
        if arm is not None:
            arm.close()
    if len(pts) < 2:
        rec("touch", FAIL, f"only {len(pts)} of 3 squares gave a reading (squares must be EMPTY; see data/camcal/selftest_*.jpg)")
        return
    bias = K.load_bias()
    e = np.linalg.norm(np.array([np.array(p["board_mm"]) - bias - np.array(p["cmd_xy"]) for p in pts]), axis=1)
    worst = float(e.max())
    detail = ", ".join(f"{p['square']} {x:.1f}" for p, x in zip(pts, e))
    if not K.BIAS_FILE.exists():
        rec("touch", WARN, f"errors mm: {detail}; no tip bias learned yet (camcal --learn-bias): numbers include a constant offset")
    elif worst <= K.PASS_MM:
        rec("touch", OK, f"tip lands within {worst:.1f} mm of the square centres ({detail})")
    elif worst <= K.WARN_MM:
        rec("touch", WARN, f"worst {worst:.1f} mm ({detail}): usable; python -m chess_player.calib.camcal if grasps miss")
    else:
        rec("touch", FAIL, f"worst {worst:.1f} mm ({detail}): python -m chess_player.calib.camcal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8771/shot")
    ap.add_argument("--no-touch", action="store_true", help="skip the arm motion (checks 1-5 only)")
    a = ap.parse_args()
    f = check_camera(a.url)
    if f is not None:
        check_grid(f)
        check_vision(f)
    check_reach()
    check_arm()
    if not a.no_touch and not any(s == FAIL for _, s, _ in rows):
        print("\nThe arm will now touch 3 squares: e3, d6, g4 must be EMPTY, hand near the power switch.")
        if input("Go? [Enter = yes, n = skip] ").strip().lower() != "n":
            check_touch(a.url)
    print("\n==== SELF-TEST ====")
    for name, status, msg in rows:
        print(f"{status:5s} {name}")
    fails = [n for n, s, _ in rows if s == FAIL]
    print("READY TO PLAY" if not fails else "NOT READY: " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
