"""Teach the grasp of one piece type on its real square, in the planner's own frame, with the follower POWERED.

  python -m chess_player.calib.teach_grasp --piece p --square e2

Stand the piece on the square and line the leader arm up with the follower. The follower copies the leader (the wrist
roll is forced to the angle the planner uses). Move the leader so the follower's jaws surround the piece the way they
should grasp it (pads at the height you want, fixed finger on its side, jaws vertical), watch the live readout, and
press Enter. The offsets of the jaw tip from the piece centre (along the jaws, sideways, height) and the jaw reading
are saved to data/taught_grasps.json; the planner then uses them for that piece type.

Taught with the arm under power: a hand-guided arm (torque off) sits on the other side of its joint backlash than a
powered one, so a pose taught that way ends up ~1 cm too low / to the side when replayed.
  --hand   old behaviour: torque off, guide the follower by hand (not recommended)
"""
import argparse
import json
import select
import sys
import time

import chess
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.ik import IK

OUT = C.DATA / "taught_grasps.json"
MAX_DIST_MM = 22.0  # the tip is at most about this far from the piece centre when the jaws surround it
MAX_TILT = 6.0  # the planner grasps with the jaws vertical; teach it the same way


def _readout(ik, T, q):
    qa = np.array([q[j] for j in C.ARM_JOINTS])
    tip = (T.R.T @ (ik.fk(qa)[:3, 3] - T.t)) * 1000.0
    tilt = float(np.degrees(np.arccos(np.clip(-ik.fk(qa)[2, 2], -1, 1))))
    return qa, tip, tilt


def _wait_enter(ik, T, get_q, step=None, need_tilt=True, msg=None):
    print(msg or f"Keep the jaws VERTICAL: tilt must be under {MAX_TILT:.0f} deg. Live readout; press Enter when it looks right.\n")
    while True:
        if step:
            step()
        q = get_q()
        _, tip, tilt = _readout(ik, T, q)
        ok = "OK    " if tilt <= MAX_TILT else "TILTED"
        print(f"\r  tilt {tilt:5.1f} deg {ok}  tip height {tip[2]:5.1f} mm  gripper {q['gripper']:5.2f} %   ", end="", flush=True)
        if select.select([sys.stdin], [], [], 0.03 if step else 0.25)[0]:
            sys.stdin.readline()
            if tilt <= MAX_TILT or not need_tilt:
                print()
                return
            print(f"\n  tilt is {tilt:.0f} deg: bring the jaws closer to vertical, then Enter again")


def teach_powered(ik, T):
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

    from chess_player.motion.arm import Arm
    from chess_player.teach.record_demo import find_leader

    leader = SOLeader(SOLeaderTeleopConfig(port=find_leader(), id=C.LEADER_ID))
    leader.connect(calibrate=False)
    arm = Arm(use_sag_model=False)

    def goal():
        act = leader.get_action()
        g = {j: float(act[f"{j}.pos"]) for j in C.ALL_JOINTS}
        g["wrist_roll"] = C.WRIST_ROLL_DEG
        return g

    try:
        g = goal()
        worst = max(abs(g[j] - arm.joints()[j]) for j in C.ARM_JOINTS if j != "wrist_roll")
        print(f"leader and follower differ by up to {worst:.0f} deg: easing the follower onto the leader over 2 s")
        arm.move_to(g, 2.0)
        _wait_enter(ik, T, arm.present, step=lambda: arm._write(goal()),
                    msg="STEP 1: the jaws OPEN around the piece, in the grasp pose (just before closing), jaws vertical. "
                        f"Tilt under {MAX_TILT:.0f} deg. Enter when it looks right.\n")
        arm._write(goal())
        time.sleep(0.6)  # let it settle under load before reading
        samples = []
        for _ in range(8):
            arm._write(goal())
            samples.append(arm.present())
            time.sleep(0.05)
        _wait_enter(ik, T, arm.present, step=lambda: arm._write(goal()), need_tilt=False,
                    msg="\nSTEP 2: now CLOSE the jaws on the piece without moving the arm, then Enter.\n")
        time.sleep(0.4)
        held = float(np.mean([(arm._write(goal()), arm.present()["gripper"])[1] for _ in range(6)]))
        _wait_enter(ik, T, arm.present, step=lambda: arm._write(goal()), need_tilt=False,
                    msg="\nSTEP 3: now OPEN the jaws just enough to let go of the piece (arm still), then Enter.\n")
        time.sleep(0.4)
        released = float(np.mean([(arm._write(goal()), arm.present()["gripper"])[1] for _ in range(6)]))
        return samples, held, released
    finally:
        try:
            leader.disconnect()
        except Exception:
            pass
        arm.close()


def teach_by_hand(ik, T):
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    cal_path = C.Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower" / f"{C.FOLLOWER_ID}.json"
    cal = {k: MotorCalibration(**v) for k, v in json.loads(cal_path.read_text()).items()}
    motors = {j: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if j == "gripper" else MotorNormMode.DEGREES)
              for i, j in enumerate(C.ALL_JOINTS)}
    bus = FeetechMotorsBus(C.FOLLOWER_PORT, motors, calibration={j: cal[j] for j in C.ALL_JOINTS})
    bus.connect(handshake=True)
    try:
        input("Hold the arm: torque goes off (wrist roll stays locked). Enter... ")
        bus.disable_torque([j for j in C.ALL_JOINTS if j != "wrist_roll"])
        now = bus.sync_read("Present_Position", num_retry=2)
        bus.write("Goal_Position", "wrist_roll", float(now["wrist_roll"]))
        bus.write("Torque_Enable", "wrist_roll", 1)
        bus.write("Goal_Position", "wrist_roll", float(C.WRIST_ROLL_DEG))
        time.sleep(1.2)
        _wait_enter(ik, T, lambda: bus.sync_read("Present_Position", num_retry=2))
        samples = []
        for _ in range(8):
            samples.append(bus.sync_read("Present_Position", num_retry=2))
            time.sleep(0.05)
        return samples, None, None
    finally:
        bus.port_handler.closePort()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece", required=True, choices=list("pnbrqk"), help="piece type (lowercase letter)")
    ap.add_argument("--square", default="e2")
    ap.add_argument("--hand", action="store_true", help="torque off, guide the follower by hand (not recommended)")
    a = ap.parse_args()
    sq = chess.parse_square(a.square)
    centre = np.asarray(square_center_mm(sq), float)
    ik, T = IK(), BoardToRobot.load()

    print(f"Stand the {a.piece.upper()} piece on {a.square}.")
    samples, held, released = teach_by_hand(ik, T) if a.hand else teach_powered(ik, T)

    q = {j: float(np.mean([s[j] for s in samples])) for j in C.ALL_JOINTS}
    spread = max(max(s[j] for s in samples) - min(s[j] for s in samples) for j in C.ARM_JOINTS)
    qa = np.array([q[j] for j in C.ARM_JOINTS])
    tip = (T.R.T @ (ik.fk(qa)[:3, 3] - T.t)) * 1000.0
    u = ik.jaw_open_direction(qa)
    ub = (T.R.T @ np.asarray(u, float))[:2]
    ub /= np.linalg.norm(ub)
    perp = np.array([-ub[1], ub[0]])
    d = tip[:2] - centre
    shift = float(-np.dot(d, ub))  # tip is this far from the piece centre toward the FIXED finger
    lateral = float(np.dot(d, perp))
    gap = float(np.interp(q["gripper"], C._GAP_PCT, C._GAP_MM))
    tilt = float(np.degrees(np.arccos(np.clip(-ik.fk(qa)[2, 2], -1, 1))))
    print(f"\njoints {np.round(qa, 1)}  gripper {q['gripper']:.2f} % (gap ~{gap:.1f} mm)  (moved {spread:.2f} deg while reading)")
    print(f"tip in board mm: x {tip[0]:.1f} y {tip[1]:.1f} z {tip[2]:.1f}   piece centre x {centre[0]:.1f} y {centre[1]:.1f}")
    if held is not None:
        print(f"jaws open {q['gripper']:.2f} %, closed on the piece {held:.2f} %, released at {released:.2f} %")
    print(f"=> shift toward the fixed finger {shift:.1f} mm, sideways {lateral:+.1f} mm, tip height {tip[2]:.1f} mm, "
          f"roll {q['wrist_roll']:.1f}, tool tilt {tilt:.0f} deg")
    dist = float(np.hypot(*d))
    if dist > MAX_DIST_MM or tip[2] > 45.0:
        raise SystemExit(f"NOT saved: the jaw tip is {dist:.0f} mm from the piece centre and {tip[2]:.0f} mm high; the jaws "
                         f"must be around the piece (within {MAX_DIST_MM:.0f} mm) when you press Enter. Run it again.")
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[a.piece] = dict(square=a.square, shift_mm=round(shift, 2), lateral_mm=round(lateral, 2), grip_z=round(float(tip[2]), 1),
                         gripper_pct=round(q["gripper"], 2), held_pct=None if held is None else round(held, 2),
                         release_delta_pct=None if (held is None or released is None) else round(max(0.0, released - held), 2), gap_mm=round(gap, 1), roll=round(q["wrist_roll"], 1),
                         powered=not a.hand, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    OUT.write_text(json.dumps(data, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
