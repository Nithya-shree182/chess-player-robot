"""Find the true height of the board surface by a slow touch-down with the closed jaws. The arm MOVES (empty squares only).

  python -m chess_player.calib.find_board_z [--squares d4 b2 g2 b7 g7] [--dry-run]

Over each square the arm goes to travel height, comes down to --start mm above where the calibration puts the surface, then lowers in
small steps (2 mm, then 1 mm below --fine). After every step it compares the tip height FK reports from the MEASURED joints with the
commanded one; contact = the tip stops following by more than 1.2 mm (two steps in a row). It then lifts straight up. The result per square
is the model height (mm, board frame) at which the tip met the board: 0 means the calibration's board height is right, negative = the
board is lower than the calibration says. With 3+ squares a plane is fitted. Results go to data/board_z_touch.json (nothing else changes).
"""
import argparse
import json
import time

import chess
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.arm import Arm, FakeArm, look_pose
from chess_player.motion.ik import IK, solve_or_raise

OUT = C.DATA / "board_z_touch.json"
LAG_MM = 1.2


def tip_z(ik, T, now):
    p = ik.fk(np.array([now[j] for j in C.ARM_JOINTS]))[:3, 3]
    return float((T.R.T @ (p - T.t))[2] * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squares", nargs="+", default=["d4", "b2", "g2", "b7", "g7"])
    ap.add_argument("--start", type=float, default=30.0, help="mm above the calibrated surface where the slow descent begins")
    ap.add_argument("--fine", type=float, default=12.0, help="below this height the steps are 1 mm instead of 2")
    ap.add_argument("--floor", type=float, default=-25.0, help="give up (no contact) below this height")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    T, ik = BoardToRobot.load(), IK()
    arm = FakeArm() if a.dry_run else Arm()
    look = {**look_pose(), "gripper": C.GRIPPER_LOOK}
    results = {}
    try:
        q = np.array([look[j] for j in C.ARM_JOINTS])
        q[4] = C.WRIST_ROLL_DEG
        for name in a.squares:
            xy = square_center_mm(chess.parse_square(name))
            hi = solve_or_raise(ik, T(*xy, C.Z_SAFE), q, what=f"{name} high")
            arm.move_to({**dict(zip(C.ARM_JOINTS, hi)), "gripper": 0.0})
            arm.settle()
            z = a.start
            q = solve_or_raise(ik, T(*xy, z), hi, C.GRASP_MAX_TILT, what=f"{name} start")
            arm.move_to(dict(zip(C.ARM_JOINTS, q)), C.DESCEND_S)
            arm.settle()
            base = tip_z(ik, T, arm.present()) - z
            print(f"{name}: at {z:.0f} mm, tip follows to within {base:+.1f} mm; lowering...")
            hits, contact = 0, None
            while z > a.floor:
                z -= 1.0 if z <= a.fine else 2.0
                q = solve_or_raise(ik, T(*xy, z), q, C.GRASP_MAX_TILT, what=f"{name} z={z:.0f}")
                arm.move_to(dict(zip(C.ARM_JOINTS, q)), 0.5)
                time.sleep(0.3)
                lag = tip_z(ik, T, arm.present()) - z - base
                print(f"    z {z:5.1f}  tip follows to within {lag:+.2f} mm", flush=True)
                if lag > LAG_MM:
                    hits += 1
                    if hits >= 2:
                        contact = tip_z(ik, T, arm.present())
                        break
                else:
                    hits = 0
            if contact is None:
                print(f"  {name}: NO contact down to {a.floor:.0f} mm: stopping (the surface is lower than that, or the tip is not touching)")
            else:
                results[name] = {"xy_mm": [float(xy[0]), float(xy[1])], "z_mm": contact}
                print(f"  {name}: contact: the surface is at model z = {contact:+.1f} mm here")
            up = solve_or_raise(ik, T(*xy, C.Z_SAFE), q, what=f"{name} lift")
            arm.move_to(dict(zip(C.ARM_JOINTS, up)), C.DESCEND_S)
            q = up
            if contact is None:
                break
        arm.move_to(look, 2.5)
    finally:
        arm.close()
    if results:
        pts = np.array([[v["xy_mm"][0], v["xy_mm"][1], v["z_mm"]] for v in results.values()])
        out = {"squares": results, "mean_z_mm": float(pts[:, 2].mean())}
        if len(pts) >= 3:
            A = np.c_[pts[:, :2], np.ones(len(pts))]
            sol = np.linalg.lstsq(A, pts[:, 2], rcond=None)[0]
            out["plane"] = {"slope_x": float(sol[0]), "slope_y": float(sol[1]), "z0_mm": float(sol[2]),
                            "residual_mm": float(np.sqrt(((A @ sol - pts[:, 2]) ** 2).mean()))}
        if not a.dry_run:
            OUT.write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
