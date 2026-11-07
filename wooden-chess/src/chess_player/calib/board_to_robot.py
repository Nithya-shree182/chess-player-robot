"""Measure where the board is in the robot frame by touching known points with the jaw tip.

  python -m chess_player.calib.board_to_robot            # touch the 4 outer corners a1, h1, h8, a8
  python -m chess_player.calib.board_to_robot --centers  # also touch d4 and e5 centers (better fit)

Torque is off for the whole script: guide the follower by hand (or put the leader in teleop beforehand and
stop it). Close the gripper, point the jaws roughly straight down (as when grasping), rest the tip on the
point, press Enter. Joint angles go through URDF forward kinematics, and a rigid transform is fitted.

The fit checks itself: the measured a1-h1 distance should equal 8 * SQUARE_MM, and the RMS residual should
be a few mm. A big mismatch means SQUARE_MM is wrong or the URDF zero doesn't match the calibration.
"""
import argparse
import json
import time

import numpy as np

from chess_player import config as C
from chess_player.geometry import fit_rigid
from chess_player.motion.ik import IK


def read_joints():
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    cal_path = C.Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower" / f"{C.FOLLOWER_ID}.json"
    cal = {k: MotorCalibration(**v) for k, v in json.loads(cal_path.read_text()).items()}
    motors = {j: Motor(i + 1, "sts3215", MotorNormMode.DEGREES) for i, j in enumerate(C.ARM_JOINTS)}
    bus = FeetechMotorsBus(C.FOLLOWER_PORT, motors, calibration={j: cal[j] for j in C.ARM_JOINTS})
    bus.connect(handshake=True)
    return bus


def report_and_save(names, board_pts, robot_pts, joints, corners_only=True):
    B = C.PLAY_MM
    T, res = fit_rigid(board_pts, robot_pts)
    rp = np.array(robot_pts) * 1000
    print(f"\nresiduals (mm): {np.round(res, 1)}   RMS {T.residual_mm:.1f}")
    if not corners_only:  # grid points: only the fit quality and the board pose are meaningful
        tilt = np.degrees(np.arccos(np.clip(T.R[2, 2], -1, 1)))
        print(f"board surface: {T.t[2] * 1000:.1f} mm above the robot base frame at a1, tilted {tilt:.1f} deg")
        if T.residual_mm > 4:
            print("WARNING: fit above 4 mm. Touch more carefully (jaws vertical, tip exactly on the meeting point).")
        T.save(points={k: v for k, v in zip(names, board_pts)}, joints=joints)
        print("saved", C.BOARD_TO_ROBOT)
        return
    side = [np.linalg.norm(rp[(i + 1) % 4] - rp[i]) for i in range(4)]
    print(f"measured board sides (mm): {np.round(side, 1)}   expected {B:.1f}")
    tilt = np.degrees(np.arccos(np.clip(T.R[2, 2], -1, 1)))
    print(f"board surface: {T.t[2] * 1000:.1f} mm above the robot base frame at a1, tilted {tilt:.1f} deg")
    far = max(np.linalg.norm(p[:2]) for p in rp)
    print(f"farthest corner {far:.0f} mm from the robot base")
    # consistency checks from chess-player's board_robot_calib.py (corners 0..3 = a1, h1, h8, a8)
    if len(rp) >= 4:
        para = np.linalg.norm((rp[0] + rp[2]) / 2 - (rp[1] + rp[3]) / 2)
        spread = rp[:4, 2].max() - rp[:4, 2].min()
        print(f"parallelogram error {para:.1f} mm (want < 4), side mismatch {max(side) - min(side):.1f} mm (want < 8), "
              f"corner height spread {spread:.1f} mm (want < 5)")
    if T.residual_mm > 5 or max(abs(s - B) for s in side) > 10:
        print("WARNING: poor fit. Touch more carefully (jaws straight down), or move the board closer.")
    T.save(points={k: v for k, v in zip(names, board_pts)}, joints=joints)
    print("saved", C.BOARD_TO_ROBOT)


def corner_descriptions():
    """Where each corner is, from the webcam clicks (webcam = human's side, so image top = robot side and
    image left = the human's left)."""
    if not C.EXT_CORNERS.exists():
        return {}
    px = json.loads(C.EXT_CORNERS.read_text())["px"]
    mx = sum(p[0] for p in px) / 4
    my = sum(p[1] for p in px) / 4
    out = {}
    for name, (x, y) in zip(["a1", "h1", "h8", "a8"], px):
        if C.CAMERA_ON_ROBOT_SIDE:  # camera on the gripper: image bottom = nearest the robot
            side = "nearest the robot" if y > my else "far from the robot"
            hand = "left" if x < mx else "right"
            where = f"{side}, {hand} in the camera picture"
        else:
            side = "robot side" if y < my else "your side"
            hand = "your left" if x < mx else "your right"
            where = f"{side}, on {hand}"
        colour = "dark" if name in ("a1", "h8") else "light"
        out[name] = f"{where}, {colour} square"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", action="store_true", help="also touch the centers of d4 and e5")
    ap.add_argument("--refit", action="store_true", help="re-fit from the joints saved last time (no touching)")
    ap.add_argument("--grid", action="store_true",
                    help="touch 9 points where four squares meet (b2/c2/b3/c3 ... f6/g6/f7/g7): easier to hit exactly "
                         "than a board corner, and nearer the arm than the far corners")
    ap.add_argument("--ring", action="store_true",
                    help="with --grid: also touch the 8 outer points (b1/c1.. etc.) so the whole board is covered (17 points)")
    a = ap.parse_args()
    if a.ring and not a.grid:
        ap.error("--ring goes with --grid")
    if a.refit:
        d = json.loads(C.BOARD_TO_ROBOT.read_text())
        ik = IK()
        names = list(d["points"])
        report_and_save(names, [d["points"][n] for n in names], [ik.fk(np.array(q))[:3, 3] for q in d["joints"]],
                        d["joints"])
        return
    B = C.PLAY_MM
    s = C.SQUARE_MM
    if a.grid:
        # Grid intersections: x = file line (between files b|c, d|e, f|g), y = rank line (2|3, 4|5, 6|7).
        files, ranks = "abcdefgh", "12345678"
        lines = {(i, j) for j in (2, 4, 6) for i in (2, 4, 6)}
        if a.ring:  # + the outer ring (lines 1, 4, 7): reaches the far and near ranks, where the fit is checked least
            lines |= {(i, j) for j in (1, 4, 7) for i in (1, 4, 7) if (i, j) != (4, 4)}
        points = {}
        for j, i in sorted((j, i) for i, j in lines):  # nearest the robot first (rank lines), then left to right
            sq = [f"{files[i - 1]}{ranks[j - 1]}", f"{files[i]}{ranks[j - 1]}", f"{files[i - 1]}{ranks[j]}", f"{files[i]}{ranks[j]}"]
            points[f"point where {', '.join(sq)} meet"] = (i * s, j * s)
    else:
        # Corners of the playing area: the corner squares' outer corners, inside the thin light line.
        corners = {"a1": (0, 0), "h1": (B, 0), "h8": (B, B), "a8": (0, B)}
        where = corner_descriptions()
        points = {f"{n} corner ({where.get(n, '')})": xy for n, xy in corners.items()}
        if a.centers:
            points.update({"d4 center": (3.5 * s, 3.5 * s), "e5 center": (4.5 * s, 4.5 * s)})

    ik = IK()
    bus = read_joints()
    input("Hold the arm: torque goes off on pan / lift / elbow / wrist-flex when you press Enter. The wrist ROLL stays "
          "locked at the angle the arm uses when playing, so it can't be turned...")
    free = [j for j in C.ARM_JOINTS if j != "wrist_roll"]
    bus.disable_torque(free)
    # lock the roll where the IK drives it: the touches must be made in the same condition the arm plays in
    now = bus.sync_read("Present_Position", num_retry=2)
    bus.write("Goal_Position", "wrist_roll", float(now["wrist_roll"]))  # hold where it is first (no jump)
    bus.write("Torque_Enable", "wrist_roll", 1)
    bus.write("Goal_Position", "wrist_roll", float(C.WRIST_ROLL_DEG))
    time.sleep(1.2)
    print(f"Torque is OFF on pan / lift / elbow / wrist-flex: guide the tip by hand. The wrist roll is locked at "
          f"{C.WRIST_ROLL_DEG:.1f} deg (jaws side by side): do not force it.\n")
    board_pts, robot_pts, joints = [], [], []
    try:
        for name, xy in points.items():
            input(f"Put the jaw tip on the {name} and press Enter...")
            q = bus.sync_read("Present_Position", num_retry=2)
            qv = np.array([q[j] for j in C.ARM_JOINTS])
            p = ik.fk(qv)[:3, 3]
            tilt = float(np.degrees(np.arccos(np.clip(-ik.fk(qv)[2, 2], -1, 1))))
            print(f"   joints {np.round(qv, 1)} -> tip {np.round(p * 1000, 1)} mm, tool tilt {tilt:.0f} deg"
                  + ("   <-- tilted a lot: keep the jaws closer to vertical if you can" if tilt > 25 else ""))
            board_pts.append([*xy, 0.0])
            robot_pts.append(p)
            joints.append(qv.tolist())
    finally:
        bus.port_handler.closePort()

    report_and_save(list(points), board_pts, robot_pts, joints, corners_only=not a.grid)


if __name__ == "__main__":
    main()
