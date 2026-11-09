"""Correct the board -> robot calibration by eye.

  python -m chess_player.calib.nudge_calibrate [--squares a8 h8 a1 h1 d4 ...] [--height 15] [--keys]

The arm MOVES. For each square it goes to where the current calibration says the centre is and hovers there
(--height mm above the board; keep >= 12). Look straight down at the tip and tell it how far to move until it is
over the centre of the square, then press Enter on an empty line. The nudges are measured, and a correction
(a 2-D similarity: shift + small rotation + scale) is fitted and saved to data/board_correction.json; every arm
target then uses it.

TYPED MODE (default; works in any terminal). At the "nudge>" prompt type moves, then Enter:
    d3        3 mm to YOUR RIGHT (toward the a-file)         a3   3 mm to your left (toward the h-file)
    w2        2 mm AWAY from you (toward the robot)          s2   2 mm TOWARD you (toward rank 8)
    d         1 mm (no number = 1 mm)       D / A / W / S   = 5 mm
    d3 w2     several moves in one line
    (empty)   the tip is over the centre: accept this square
    skip      skip this square             done   finish now (needs at least 4 accepted squares)
    stop      freeze the arm and quit without saving
With --keys it reads single key presses instead (a/d/w/s, arrows, Enter, k, q, Esc).

Pieces must be off the board, and the board must be taped down: if the tip ever touches it, press Ctrl+C.
"""
import argparse
import json
import re

import chess
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.arm import Arm, FakeArm, look_pose
from chess_player.motion.ik import IK, solve_or_raise

MAX_NUDGE_MM = 45.0
MIN_HEIGHT_MM = 12.0
# far corners first (the ones that went wrong), then the rest; spread over the board so the fit is well conditioned
DEFAULT_SQUARES = ["a8", "h8", "a4", "h4", "a1", "h1", "d1", "d8", "d4", "e5"]
# Directions as seen from the HUMAN's side (the camera's seat): a-file on your right, h-file on your left, the robot at the far end.
# board x runs a -> h (your left), board y runs rank 1 -> 8 (toward you).
DIRS = {"a": (1, 0), "d": (-1, 0), "w": (0, -1), "s": (0, 1)}


def parse_typed(line):
    """'d3 w2 A' -> (dx, dy) in mm, or None if it isn't a move command."""
    dx = dy = 0.0
    parts = re.findall(r"([adwsADWS])(\d*\.?\d*)", line)
    rest = re.sub(r"([adwsADWS])(\d*\.?\d*)", "", line).strip()
    if not parts or rest:
        return None
    for ch, num in parts:
        n = float(num) if num else (5.0 if ch.isupper() else 1.0)
        ux, uy = DIRS[ch.lower()]
        dx += ux * n
        dy += uy * n
    return dx, dy


def get_command(keys_mode):
    """-> ('move', (dx, dy)) | 'accept' | 'skip' | 'done' | 'stop' | ('bad', text)"""
    if keys_mode:
        try:
            import msvcrt

            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                ch = {"H": "w", "P": "s", "K": "a", "M": "d"}.get(msvcrt.getwch(), "")
        except ImportError:  # Linux / macOS: one key press without Enter; arrow keys arrive as ESC [ A/B/C/D
            import select
            import sys
            import termios
            import tty

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                ch = sys.stdin.read(1)
                if ch == "\x1b" and select.select([sys.stdin], [], [], 0.05)[0]:  # an arrow key, not a lone Esc
                    seq = sys.stdin.read(2)
                    ch = {"[A": "w", "[B": "s", "[D": "a", "[C": "d"}.get(seq, "")
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if ch == "\x1b":
            return "stop"
        if ch in ("\r", "\n"):
            return "accept"
        if ch == "k":
            return "skip"
        if ch == "q":
            return "done"
        mv = parse_typed(ch) if ch else None
        return ("move", mv) if mv else ("bad", repr(ch))
    line = input("nudge> ").strip()
    if line == "":
        return "accept"
    if line in ("skip", "done", "stop"):
        return line
    mv = parse_typed(line)
    return ("move", mv) if mv else ("bad", line)


def fit_similarity(P, Cmd):
    """c = s * Rot(theta) @ p + b  (least squares, closed form). Returns (A 2x2, b, per-point residual mm)."""
    P, Cmd = np.asarray(P, float), np.asarray(Cmd, float)
    pm, cm = P.mean(0), Cmd.mean(0)
    p, c = P - pm, Cmd - cm
    a = (p * c).sum() / (p * p).sum()  # s*cos
    bb = (p[:, 0] * c[:, 1] - p[:, 1] * c[:, 0]).sum() / (p * p).sum()  # s*sin
    A = np.array([[a, -bb], [bb, a]])
    b = cm - A @ pm
    res = np.linalg.norm(Cmd - (P @ A.T + b), axis=1)
    return A, b, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squares", nargs="+", default=DEFAULT_SQUARES)
    ap.add_argument("--height", type=float, default=15.0, help="mm above the board while you judge (>= 12)")
    ap.add_argument("--keys", action="store_true", help="single key presses instead of typed lines")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.height < MIN_HEIGHT_MM:
        raise SystemExit(f"--height {a.height} is too low: the calibration is only good to a few mm, so the tip could "
                         f"hit the board. Use at least {MIN_HEIGHT_MM:.0f}.")

    T, ik = BoardToRobot.load(apply_correction=False), IK()  # nudge relative to the UNcorrected calibration
    arm = FakeArm() if a.dry_run else Arm()
    look = {**look_pose(), "gripper": C.GRIPPER_LOOK}
    true_pts, cmd_pts, names = [], [], []
    saved = False
    try:
        q = np.array([look[j] for j in C.ARM_JOINTS])
        q[4] = C.WRIST_ROLL_DEG
        quit_all = False
        for name in a.squares:
            if quit_all:
                break
            xy0 = square_center_mm(chess.parse_square(name))
            off = np.zeros(2)
            hi = solve_or_raise(ik, T(*xy0, C.Z_SAFE), q, what=f"{name} high")
            arm.move_to({**dict(zip(C.ARM_JOINTS, hi)), "gripper": 0.0})
            arm.settle()
            lo = solve_or_raise(ik, T(*xy0, a.height), hi, C.GRASP_MAX_TILT, what=f"{name} low")
            arm.move_to(dict(zip(C.ARM_JOINTS, lo)), C.DESCEND_S)
            arm.settle()
            print(f"\n{name}: tip is {a.height:.0f} mm above where {name}'s centre should be.")
            print("   look straight down; move it with e.g. 'd3 w2' (right 3, away 2), empty line = accept, "
                  "'skip', 'done', 'stop'")
            q = lo
            while True:
                cmd = get_command(a.keys)
                if cmd == "stop":
                    if hasattr(arm, "freeze"):
                        arm.freeze()
                    print("stopped: arm frozen, nothing saved")
                    return
                if cmd == "accept":
                    if not off.any():
                        print("   no nudge yet, so you're saying the tip is already over the centre. "
                              "Empty line again to confirm, or type a move.")
                        if get_command(a.keys) != "accept":
                            continue
                    true_pts.append(xy0)
                    cmd_pts.append(xy0 + off)
                    names.append(name)
                    print(f"   accepted {name}: nudge ({off[0]:+.0f}, {off[1]:+.0f}) mm")
                    break
                if cmd == "skip":
                    print(f"   skipped {name}")
                    break
                if cmd == "done":
                    quit_all = True
                    break
                kind, val = cmd
                if kind == "bad":
                    print(f"   didn't understand {val!r}: type moves like d3 w2 (a/d = left/right, w/s = away/toward "
                          f"the robot, capital = 5 mm), or empty line / skip / done / stop")
                    continue
                new = off + np.array(val, float)
                if np.linalg.norm(new) > MAX_NUDGE_MM:
                    print("   that's further than I'll go from the calibrated point")
                    continue
                try:
                    q = solve_or_raise(ik, T(*(xy0 + new), a.height), q, C.GRASP_MAX_TILT, what=name)
                except Exception as e:
                    print("   can't reach that:", e)
                    continue
                off = new
                arm.move_to(dict(zip(C.ARM_JOINTS, q)), 0.5)
                arm.settle(iters=2)
                print(f"   moved: nudge is now ({off[0]:+.0f}, {off[1]:+.0f}) mm")
            # back up before the next square
            q = solve_or_raise(ik, T(*(xy0 + off), C.Z_SAFE), q, what=f"{name} lift")
            arm.move_to(dict(zip(C.ARM_JOINTS, q)), C.DESCEND_S)

        if len(true_pts) >= 4:
            A, b, res = fit_similarity(true_pts, cmd_pts)
            scale = float(np.sqrt(abs(np.linalg.det(A))))
            rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
            print(f"\nfitted correction from {len(true_pts)} squares: shift ({b[0]:+.1f}, {b[1]:+.1f}) mm, "
                  f"rotation {rot:+.2f} deg, scale {scale:.4f}")
            print("   leftover per square (mm):", {n: round(float(r), 1) for n, r in zip(names, res)})
            if scale < 0.9 or scale > 1.1 or abs(rot) > 6:
                print("WARNING: that is a big correction: check the squares were judged consistently. Not saved.")
            else:
                C.BOARD_CORRECTION.write_text(json.dumps(
                    {"A": A.tolist(), "b": b.tolist(), "squares": names,
                     "nudges_mm": [(c - t).tolist() for t, c in zip(true_pts, cmd_pts)],
                     "residual_mm": float(np.sqrt((res ** 2).mean())), "height_mm": a.height}, indent=2))
                saved = True
                print("saved", C.BOARD_CORRECTION)
        else:
            print("fewer than 4 squares accepted: nothing saved")
    finally:
        try:  # lift first, then go home (never sweep sideways at board height)
            now = arm.present()
            qn = np.array([now[j] for j in C.ARM_JOINTS])
            sol, err, _ = ik.solve(ik.fk(qn)[:3, 3] + np.array([0, 0, 0.08]), qn)
            if err < 5:
                arm.move_to(dict(zip(C.ARM_JOINTS, sol)), 1.5)
            arm.move_to(look, 3.0)
        except Exception as e:
            print("(could not return to the look pose:", e, ")")
        arm.close()
    if saved:
        print("Now re-run hover_test on the far squares to check.")


if __name__ == "__main__":
    main()
