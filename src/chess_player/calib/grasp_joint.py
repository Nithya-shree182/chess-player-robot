"""Replay a hand-taught grasp EXACTLY: the same joint angles (so the same tool tilt and tip), not just a tip position.

  python -m chess_player.calib.grasp_joint --pose data/handheld_rook_pose.json --src c2 --dst c3 [--dry]

The taught pose (data/handheld_*_pose.json: six joints read while you held the piece) is reproduced by solving the arm for the
SAME tool axis. The piece is assumed to have stood on the centre of the taught square (--taught-square, default c2) when you taught
it; a different piece position moves the whole pose by the difference. Sequence (the arm MOVES without --dry):
  home -> above the piece (taught pose raised 45 mm, jaws open) -> straight down into the exact taught joints -> [Enter] close ->
  lift -> carry -> down -> release -> up -> home.   Gates (Enter): before closing, and after the lift.
"""
import argparse
import json

import chess
import numpy as np
from scipy.optimize import least_squares

from chess_player import config as C
from chess_player.calib.locate_piece import grab, locate
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.arm import Arm, look_pose
from chess_player.motion.ik import IK

J = C.ARM_JOINTS


def solve_axis(ik, tip_m, axis, seed_deg, roll):
    """Joint angles (pan, lift, elbow, flex) giving the jaw tip at tip_m with the tool axis equal to `axis` (robot frame)."""
    def resid(x):
        q = np.r_[x, roll]
        Tm = ik.fk(q)
        return np.r_[(Tm[:3, 3] - tip_m) * 1000.0, (Tm[:3, 2] - axis) * 100.0]
    sol = least_squares(resid, np.asarray(seed_deg[:4], float), x_scale=10.0)
    q = np.r_[sol.x, roll]
    err = np.linalg.norm((ik.fk(q)[:3, 3] - tip_m) * 1000.0)
    return q, float(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="data/handheld_rook_pose.json")
    ap.add_argument("--src", default="c2")
    ap.add_argument("--dst", default="c3")
    ap.add_argument("--taught-square", default="c2", help="the square the piece stood on when you taught the pose")
    ap.add_argument("--xy", help='the piece position "x,y" in board mm (default: the centre of --src)')
    ap.add_argument("--lift", type=float, default=45.0)
    ap.add_argument("--open", type=float, default=C.GRIP_OPEN_MAX_PCT)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    pose = json.loads((C.ROOT / a.pose).read_text() if not a.pose.startswith("/") else open(a.pose).read())
    u = np.array([pose["joints"][j] for j in C.ALL_JOINTS], float)
    T, ik = BoardToRobot.load(), IK()
    Tm = ik.fk(u[:5])
    axis, tip_hold = Tm[:3, 2].copy(), Tm[:3, 3].copy()
    base_xy = np.asarray(square_center_mm(chess.parse_square(a.taught_square)))
    piece = np.array([float(v) for v in a.xy.split(",")]) if a.xy else np.asarray(square_center_mm(chess.parse_square(a.src)))
    dst = np.asarray(square_center_mm(chess.parse_square(a.dst)))
    held_gripper = float(u[5])

    def at(delta_xy_mm, dz_mm, seed):
        tip = tip_hold + T.R @ np.array([delta_xy_mm[0], delta_xy_mm[1], dz_mm]) / 1000.0
        return solve_axis(ik, tip, axis, seed, u[4])

    q_grasp, e0 = at(piece - base_xy, 0.0, u)
    q_pre, e1 = at(piece - base_xy, a.lift, q_grasp)
    q_dst_pre, e2 = at(dst - base_xy, a.lift, q_pre)
    q_dst, e3 = at(dst - base_xy, 2.0, q_dst_pre)
    print(f"taught: tip {np.round(pose['tip_board_mm'], 1).tolist()}, tilt {pose['tilt_deg']} deg, jaws {held_gripper:.1f} %")
    print(f"solved grasp joints {np.round(q_grasp, 2).tolist()} (taught {np.round(u[:5], 2).tolist()}); tip error {e0:.2f} mm; "
          f"pre-grasp err {e1:.2f}, carry err {e2:.2f}, place err {e3:.2f} mm")
    if max(e0, e1, e2, e3) > 2.0:
        raise SystemExit("cannot reproduce the taught tool angle at one of the poses: not moving")
    if a.dry:
        print("(dry run: nothing moved)")
        return

    arm = Arm()
    look = {**look_pose(), "gripper": C.GRIPPER_LOOK}

    def goto(q, grip, dur=None):
        arm.move_to({**dict(zip(J, q)), "gripper": grip}, dur)

    try:
        now = arm.present()
        qn = np.array([now[j] for j in J])
        tn = T.R.T @ (ik.fk(qn)[:3, 3] - T.t) * 1000
        if tn[2] < 90.0:  # low: let go of anything in the jaws, then straight up
            goto(qn, a.open, 1.5)
            arm.dwell()
            qu, _ = solve_axis(ik, ik.fk(qn)[:3, 3] + T.R @ np.array([0, 0, 100.0 - tn[2]]) / 1000.0, ik.fk(qn)[:3, 2], qn, qn[4])
            goto(qu, a.open, C.DESCEND_S)
        goto(q_pre, a.open)
        arm.settle()
        goto(q_grasp, a.open, C.DESCEND_S + 1.5)
        arm.settle()
        got = arm.present()
        tip = T.R.T @ (ik.fk(np.array([got[j] for j in J]))[:3, 3] - T.t) * 1000
        print(f"AT THE TAUGHT POSE, jaws open: tip ({tip[0]:.1f}, {tip[1]:.1f}, {tip[2]:.1f}) mm; taught ({pose['tip_board_mm'][0]}, "
              f"{pose['tip_board_mm'][1]}, {pose['tip_board_mm'][2]})")
        input("Enter = CLOSE, Ctrl+C = back up: ")
        arm.gripper(1.0, 1.8)
        arm.dwell()
        held = arm.present()["gripper"]
        print(f"closed: jaws stopped at {held:.1f} % (taught hold {held_gripper:.1f} %)" + ("  <-- NOTHING IN THE HAND" if held < held_gripper * 0.5 else ""))
        hold = max(0.0, held - C.HOLD_SQUEEZE_PCT)
        goto(q_pre, hold, C.CARRY_UP_S)
        input("Is it held? Enter = carry and place, Ctrl+C = back up and stop: ")
        goto(q_dst_pre, hold, 3.5)
        goto(q_dst, hold, C.CARRY_DOWN_S)
        arm.settle()
        arm.gripper(min(a.open, held + C.RELEASE_OPEN_PCT))
        arm.dwell()
        goto(q_dst_pre, min(a.open, held + C.RELEASE_OPEN_PCT), C.DESCEND_S)
        goto(np.array([look[j] for j in J]), 0.0, 2.5)
        print("done")
    except KeyboardInterrupt:
        print("\ninterrupted: opening and going back up")
        arm.gripper(a.open)
        goto(q_pre, a.open, C.DESCEND_S)
    finally:
        arm.close()


if __name__ == "__main__":
    main()
