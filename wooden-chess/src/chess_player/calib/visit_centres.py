"""Move the arm, one square after another, to the hand-taught centre of EVERY square (data/center_teach.json) at a fixed height above the board.

  python -m chess_player.calib.visit_centres [--height 5] [--dwell 1.5] [--from h1] [--only e4 d4 ...]

EMPTY BOARD. Order: h1 g1 .. a1, h2 .. a8. Per square: up to the travel height, across, down slowly to --height mm above the board surface, wait, report where
the arm REALLY is (measured joints -> forward kinematics) against the taught position, up again. Everything is planned on a simulated arm first: nothing
moves if any square is unreachable. Ctrl+C freezes the arm where it is (no swing home). The camera is not used.
Heights are in the board frame (0 = the board surface as the calibration knows it).
"""
import argparse
import json
import time

import chess
import numpy as np

from chess_player import config as C
from chess_player.calib.center_teach import board_from_robot
from chess_player.geometry import BoardToRobot
from chess_player.motion.arm import Arm, FakeArm, look_pose
from chess_player.motion.ik import IK

TRAVEL_MM = 45.0  # height of the sideways moves
SIDE_SPEED = 45.0  # mm/s
DOWN_SPEED = 20.0  # mm/s, the last stretch down
UP_SPEED = 30.0
GRIP_PCT = 2.0  # closed, as when the centres were taught


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, default=5.0, help="tip height above the board surface, mm")
    ap.add_argument("--dwell", type=float, default=1.5)
    ap.add_argument("--from", dest="start")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()

    teach = json.loads((C.DATA / "center_teach.json").read_text())
    order = [chess.square_name(chess.square(f, r)) for r in range(8) for f in range(7, -1, -1)]
    if a.only:
        order = [s for s in order if s in a.only]
    elif a.start:
        order = order[order.index(a.start):]
    missing = [s for s in order if s not in teach]
    if missing:
        raise SystemExit(f"not taught: {missing}")
    T, ik = BoardToRobot.load(), IK()
    goal = {s: board_from_robot(T, np.array(teach[s]["tip_robot_mm"], float) / 1000.0)[:2] for s in order}  # the taught PHYSICAL tip place, in today's board frame

    def ik_at(xy, z, seed):
        q, err, tilt = ik.solve(T(xy[0], xy[1], z), seed, C.GRASP_MAX_TILT)
        if err > C.IK_MAX_ERR_MM or tilt > C.GRASP_MAX_TILT:
            raise RuntimeError(f"unreachable: IK error {err:.1f} mm, tilt {tilt:.0f} deg at ({xy[0]:.0f}, {xy[1]:.0f}, {z:.0f})")
        return q, float(tilt)

    def leg(seed, xy0, z0, xy1, z1, speed):
        """Cartesian waypoints (IK every 6 mm) from (xy0, z0) to (xy1, z1); returns (waypoints, duration, last seed, max tilt)."""
        L = float(np.linalg.norm(np.r_[xy1 - xy0, z1 - z0]))
        n = max(1, int(np.ceil(L / 6.0)))
        wps, tmax = [], 0.0
        for i in range(1, n + 1):
            f = i / n
            q, tilt = ik_at(xy0 + (xy1 - xy0) * f, z0 + (z1 - z0) * f, seed)
            seed = q
            tmax = max(tmax, tilt)
            wps.append({**dict(zip(C.ARM_JOINTS, q)), "gripper": GRIP_PCT})
        return wps, max(0.6, L / speed), seed, tmax

    # ---- 1. plan everything on a simulated arm: nothing moves unless every square is reachable
    look = {**look_pose(), "gripper": GRIP_PCT}
    seed0 = np.array([look[j] for j in C.ARM_JOINTS], float)
    seed0[4] = C.WRIST_ROLL_DEG
    plan = []
    seed, cur, curz = seed0, None, None
    worst_tilt = 0.0
    for s in order:
        xy = goal[s]
        legs = []
        if cur is not None:
            legs.append(leg(seed, cur, a.height, cur, TRAVEL_MM, UP_SPEED)); seed = legs[-1][2]
            legs.append(leg(seed, cur, TRAVEL_MM, xy, TRAVEL_MM, SIDE_SPEED)); seed = legs[-1][2]
        else:
            q, _ = ik_at(xy, TRAVEL_MM, seed)
            seed = q
        legs.append(leg(seed, xy, TRAVEL_MM, xy, a.height, DOWN_SPEED)); seed = legs[-1][2]
        worst_tilt = max([worst_tilt] + [l[3] for l in legs])
        plan.append((s, legs))
        cur = xy
    first_q, _ = ik_at(goal[order[0]], TRAVEL_MM, seed0)
    print(f"plan OK: {len(order)} squares, tool tilt up to {worst_tilt:.0f} deg, tip {a.height:g} mm above the board, travel {TRAVEL_MM:g} mm")

    # ---- 2. go
    arm = Arm(use_sag_model=False)
    results = []
    try:
        print("to the look pose, then above the first square...")
        arm.move_to(look, 4.0)
        arm.move_to({**dict(zip(C.ARM_JOINTS, first_q)), "gripper": GRIP_PCT}, 4.0)
        for k, (s, legs) in enumerate(plan):
            for i, (wps, dur, _, _) in enumerate(legs):
                if k == 0 and i == 0 and len(legs) == 1:
                    pass
                why = arm.move_path([arm.joints()] + wps, dur)
                if why:
                    raise RuntimeError(why)
            time.sleep(a.dwell)
            q = arm.present()
            p = ik.fk(np.array([q[j] for j in C.ARM_JOINTS]))[:3, 3]
            b = board_from_robot(T, p)
            err = b[:2] - goal[s]
            results.append((s, err, b[2]))
            print(f"[{k + 1}/{len(plan)}] {s}: at x {b[0]:.1f} y {b[1]:.1f} height {b[2]:.1f} mm | taught x {goal[s][0]:.1f} y {goal[s][1]:.1f} | arm is {err[0]:+.1f} X, {err[1]:+.1f} Y from the taught point", flush=True)
        up, dur, _, _ = leg(seed, goal[order[-1]], a.height, goal[order[-1]], 60.0, UP_SPEED)
        arm.move_path([arm.joints()] + up, dur)
        arm.move_to(look, 4.0)
    except KeyboardInterrupt:
        print("\nstopped, freezing the arm")
        arm.freeze()
    finally:
        if results:
            e = np.array([r[1] for r in results])
            print(f"\n{len(results)} squares visited: arm minus taught, mean {e.mean(0)[0]:+.2f} X, {e.mean(0)[1]:+.2f} Y mm; worst {np.abs(e).max(0)[0]:.1f} X, {np.abs(e).max(0)[1]:.1f} Y mm; "
                  f"height reached {np.mean([r[2] for r in results]):.1f} mm on average (asked {a.height:g})")
        arm.close()


if __name__ == "__main__":
    main()
