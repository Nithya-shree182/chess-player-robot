"""Vision-guided top-down grasp of one piece: YOLO -> board mm -> straight-down tip, gripper open to the taught maximum,
descend to the depth YOU chose, close, lift, carry to the target square, set down, release, go home.

  python -m chess_player.calib.grasp_vision --src c2 --dst c3 --depth 22 [--lift 45] [--dry]

The arm MOVES (without --dry). Gates (Enter): 1) jaws open at the grasp depth around the piece - check they straddle it,
then Enter closes them; 2) after closing and lifting - check the piece is held, then Enter carries and places it.
Ctrl+C at a gate: the arm goes back up and home. Gripper opening never exceeds config.GRIP_OPEN_MAX_PCT.
"""
import argparse

import chess
import numpy as np

from chess_player import config as C
from chess_player.calib.locate_piece import grab, locate
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.arm import Arm, FakeArm, look_pose
from chess_player.motion.ik import IK, solve_or_raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="c2")
    ap.add_argument("--dst", default="c3")
    ap.add_argument("--depth", type=float, required=True, help="jaw-tip height above the board at the grasp (mm)")
    ap.add_argument("--lift", type=float, default=45.0, help="tip height while carrying (mm)")
    ap.add_argument("--url", default=str(C.EXT_CAMERA_URL))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    got = locate(grab(a.url), chess.parse_square(a.src))
    if got is None:
        raise SystemExit(f"YOLO sees no piece on {a.src}")
    sym, conf, xy, *_ = got
    dst = np.asarray(square_center_mm(chess.parse_square(a.dst)))
    T, ik = BoardToRobot.load(), IK()
    look = {**look_pose(), "gripper": C.GRIPPER_LOOK}
    q0 = np.array([look[j] for j in C.ARM_JOINTS])
    q0[4] = C.WRIST_ROLL_DEG
    open_pct = C.GRIP_OPEN_MAX_PCT
    print(f"{sym} on {a.src} at board mm ({xy[0]:.1f}, {xy[1]:.1f}); target {a.dst} ({dst[0]:.1f}, {dst[1]:.1f}); "
          f"grasp depth {a.depth:.0f} mm, carry height {a.lift:.0f} mm, jaws open {open_pct:.1f} % (~33 mm)")

    def at(x, y, z, seed):
        return solve_or_raise(ik, T(x, y, z), seed, C.GRASP_MAX_TILT, what=f"({x:.0f},{y:.0f},{z:.0f})")

    hi_src = solve_or_raise(ik, T(xy[0], xy[1], C.Z_SAFE), q0, what="above source")
    above = at(xy[0], xy[1], a.lift + 10, hi_src)
    down = at(xy[0], xy[1], a.depth, above)
    carry = at(xy[0], xy[1], a.lift, down)
    over_dst = at(dst[0], dst[1], a.lift, carry)
    put = at(dst[0], dst[1], a.depth + 2.0, over_dst)  # 2 mm higher than the grasp so the piece is set down, not pushed
    print("all poses reachable (tilt <= %.0f deg)" % C.GRASP_MAX_TILT)
    if a.dry:
        print("(dry run: nothing moved)")
        return

    arm = Arm()
    J = C.ARM_JOINTS

    def goto(q, grip, dur=None):
        arm.move_to({**dict(zip(J, q)), "gripper": grip}, dur)

    try:
        goto(hi_src, 0.0)
        arm.settle()
        goto(above, open_pct, C.DESCEND_S)
        arm.settle()
        goto(down, open_pct, C.DESCEND_S + 1.5)
        arm.settle()
        now = arm.present()
        tip = T.R.T @ (ik.fk(np.array([now[j] for j in J]))[:3, 3] - T.t) * 1000
        print(f"AT GRASP DEPTH, jaws open ({now['gripper']:.1f} %): tip board mm ({tip[0]:.1f}, {tip[1]:.1f}, {tip[2]:.1f}); "
              f"pawn at ({xy[0]:.1f}, {xy[1]:.1f})")
        input("Check the open jaws straddle the pawn. Enter = CLOSE, Ctrl+C = back up: ")
        arm.gripper(1.0, 1.8)
        arm.dwell()
        held = arm.present()["gripper"]
        gap = float(np.interp(held, C._GAP_PCT, C._GAP_MM))
        print(f"closed: jaws stopped at {held:.1f} % (gap ~{gap:.0f} mm)" + ("  <-- NOTHING IN THE HAND" if held < 2.0 else ""))
        hold = max(0.0, held - C.HOLD_SQUEEZE_PCT)
        if held < 2.0:
            arm.gripper(open_pct)
            goto(above, open_pct, C.DESCEND_S)
            raise SystemExit("closed on nothing: back up, not carrying")
        goto(carry, hold, C.CARRY_UP_S)
        print(f"lifted to {a.lift:.0f} mm holding at {arm.present()['gripper']:.1f} %")
        input("Is the pawn held? Enter = carry to %s and set it down, Ctrl+C = put it back up and stop: " % a.dst)
        goto(over_dst, hold, 3.5)
        goto(put, hold, C.CARRY_DOWN_S)
        arm.settle()
        arm.gripper(min(open_pct, held + C.RELEASE_OPEN_PCT))
        arm.dwell()
        goto(over_dst, min(open_pct, held + C.RELEASE_OPEN_PCT), C.DESCEND_S)
        goto(look_q(look), 0.0, 2.5)
        print("done: pawn placed, arm home")
    except KeyboardInterrupt:
        print("\ninterrupted: opening and going back up")
        arm.gripper(open_pct)
        goto(above, open_pct, C.DESCEND_S)
    finally:
        arm.close()


def look_q(look):
    return np.array([look[j] for j in C.ARM_JOINTS])


if __name__ == "__main__":
    main()
