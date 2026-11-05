"""Teleoperate (leader arm -> follower arm) and RECORD the motion, on Windows or Linux. No cameras needed.

  python -m chess_player.teach.record_poses --task "pawn e2 to e4"
  python -m chess_player.teach.record_poses --task "pawn e2 to e4" --episodes 5

Put the piece on the first square of the task. The follower copies the leader at all times (it eases onto the leader's
pose for the first 2 s, so line the arms up roughly first). Then, per episode:
  Enter   start recording            (then do the whole move with the leader: grasp, lift, carry, set down, retreat)
  Enter   stop recording
  n+Enter discard it,  Enter keep it
Saved to data/demos/<time>_<task>/poses.csv (+ info.json) in the format the original planner learned from, so
  python -m chess_player.teach.analyze_demo --last 1
prints the grasp height, jaw offset, opening, holding reading, carry height and where the piece was set down.

At the end the follower keeps HOLDING its pose (torque on). Ctrl+C at any time also leaves it holding;
python src/tools/torque.py all   lets it go limp (support the arm first).
"""
import argparse
import csv
import json
import re
import threading
import time
from pathlib import Path

import chess
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot, mm_to_square

CAL = Path.home() / ".cache/huggingface/lerobot/calibration"
FOLLOWER_CAL = CAL / "robots/so_follower" / f"{C.FOLLOWER_ID}.json"
LEADER_CAL = CAL / "teleoperators/so_leader" / f"{C.LEADER_ID}.json"
FPS = 30
J = C.ALL_JOINTS  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

COLUMNS = (["t", "frame"] + [f"f_{j}" for j in J] + [f"l_{j}" for j in J]
           + ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg", "board_x_mm", "board_y_mm",
              "height_above_board_mm", "tilt_deg", "square", "lift_load_pct", "elbow_load_pct"])


class Cal:
    """Raw tick <-> LeRobot units for one arm, from its calibration file (no bus normalisation involved)."""

    def __init__(self, path):
        d = json.loads(Path(path).read_text())
        self.lo = np.array([d[j]["range_min"] for j in J], float)
        self.hi = np.array([d[j]["range_max"] for j in J], float)

    def units(self, raw):
        """Raw ticks -> degrees (arm joints, 0 = middle of the calibrated range) and 0-100 % (gripper)."""
        raw = np.asarray(raw, float)
        mid = (self.lo + self.hi) / 2
        out = (raw - mid) * 360 / 4095
        out[5] = (raw[5] - self.lo[5]) / (self.hi[5] - self.lo[5]) * 100
        return out

    def frac(self, raw):
        return (np.asarray(raw, float) - self.lo) / (self.hi - self.lo)

    def from_frac(self, f):
        return self.lo + np.clip(f, 0.0, 1.0) * (self.hi - self.lo)


def make_bus(port, cal_file):
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    cal = {k: MotorCalibration(**v) for k, v in json.loads(Path(cal_file).read_text()).items()}
    motors = {j: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if j == "gripper" else MotorNormMode.DEGREES)
              for i, j in enumerate(J)}
    bus = FeetechMotorsBus(port, motors, calibration=cal)
    bus.connect(handshake=True)
    return bus


def read_raw(bus):
    r = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    return np.array([r[j] for j in J], float)


def frame_row(t, k, f_units, l_units, ik, T, loads):
    """One CSV row (dict) from the follower's and leader's joint values in LeRobot units."""
    Tm = ik.fk(f_units[:5])
    tip = Tm[:3, 3] * 1000
    pb = T.R.T @ (Tm[:3, 3] - T.t) * 1000
    tilt = float(np.degrees(np.arccos(np.clip(-Tm[2, 2], -1, 1))))
    sq = mm_to_square(pb[0], pb[1])
    row = {"t": round(t, 3), "frame": k}
    row.update({f"f_{j}": round(float(v), 2) for j, v in zip(J, f_units)})
    row.update({f"l_{j}": round(float(v), 2) for j, v in zip(J, l_units)})
    row.update(x_mm=round(float(tip[0]), 1), y_mm=round(float(tip[1]), 1), z_mm=round(float(tip[2]), 1),
               roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
               board_x_mm=round(float(pb[0]), 1), board_y_mm=round(float(pb[1]), 1),
               height_above_board_mm=round(float(pb[2]), 1), tilt_deg=round(tilt, 1),
               square=chess.square_name(sq) if sq is not None else "")
    row["lift_load_pct"], row["elbow_load_pct"] = ("", "") if loads is None else loads
    return row


def save_episode(rows, task, keep_dir):
    folder = C.DATA / "demos" / f"{time.strftime('%Y%m%d_%H%M%S')}_{re.sub(r'[^a-z0-9]+', '_', task.lower())[:40]}"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "poses.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (folder / "info.json").write_text(json.dumps(
        {"task": task, "fps": FPS, "frames": len(rows), "seconds": rows[-1]["t"] if rows else 0,
         "board_calibrated": True, "recorder": "record_poses (Windows)"}, indent=2))
    return folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help='e.g. "pawn e2 to e4" (the two squares are read from it)')
    ap.add_argument("--episodes", type=int, default=1)
    a = ap.parse_args()

    T, ik = BoardToRobot.load(), None
    from chess_player.motion.ik import IK

    ik = IK()
    fcal, lcal = Cal(FOLLOWER_CAL), Cal(LEADER_CAL)
    follower = make_bus(C.FOLLOWER_PORT, FOLLOWER_CAL)
    leader = make_bus(C.LEADER_PORT, LEADER_CAL)
    try:
        leader.disable_torque()  # the leader is moved by hand
        raw_f = read_raw(follower)
        follower.sync_write("Goal_Position", {j: float(v) for j, v in zip(J, raw_f)}, normalize=False)  # hold: no jump
        for j in J:
            follower.write("Torque_Enable", j, 1)
        events = []  # Enter presses

        def keys():
            while True:
                events.append(input())

        threading.Thread(target=keys, daemon=True).start()
        kept = 0
        while kept < a.episodes:
            print(f"\nEpisode {kept + 1}/{a.episodes}: '{a.task}'. The follower follows the leader. "
                  f"Press Enter to START recording, Enter again to STOP.")
            events.clear()
            rows, recording, k, t_start, ease0 = [], False, 0, None, time.perf_counter()
            raw_start = read_raw(follower)
            while True:
                t0 = time.perf_counter()
                lraw = read_raw(leader)
                target = fcal.from_frac(lcal.frac(lraw))  # same fraction of its own calibrated range
                if t0 - ease0 < 2.0:  # first 2 s: ease from where the follower is onto the leader's pose
                    s = (t0 - ease0) / 2.0
                    s = s * s * (3 - 2 * s)
                    target = raw_start + (target - raw_start) * s
                follower.sync_write("Goal_Position", {j: float(v) for j, v in zip(J, target)}, normalize=False)
                fraw = read_raw(follower)
                loads = None
                if recording and k % 3 == 0:
                    ld = follower.sync_read("Present_Load", ["shoulder_lift", "elbow_flex"], normalize=False, num_retry=1)
                    loads = (round(abs(ld["shoulder_lift"]) / 10, 1), round(abs(ld["elbow_flex"]) / 10, 1))
                if events:
                    events.pop(0)
                    if not recording:
                        recording, t_start, k, rows = True, t0, 0, []
                        print("  RECORDING... do the move, then press Enter")
                    else:
                        break
                if recording:
                    rows.append(frame_row(t0 - t_start, k, fcal.units(fraw), lcal.units(lraw), ik, T, loads))
                    k += 1
                dt = 1.0 / FPS - (time.perf_counter() - t0)
                if dt > 0:
                    time.sleep(dt)
            print(f"  stopped: {len(rows)} frames, {rows[-1]['t']:.1f} s. Keep it? Enter = keep, n = discard")
            ans = input().strip().lower()
            if ans == "n" or len(rows) < 30:
                print("  discarded")
                continue
            folder = save_episode(rows, a.task, None)
            kept += 1
            print("  saved", folder)
        print("\nDone. Analyse with:  python -m chess_player.teach.analyze_demo --last", a.episodes)
    except KeyboardInterrupt:
        print("\ninterrupted: the follower keeps holding its pose")
    finally:
        try:
            now = read_raw(follower)  # hold exactly where it is
            follower.sync_write("Goal_Position", {j: float(v) for j, v in zip(J, now)}, normalize=False)
        except Exception:
            pass
        follower.port_handler.closePort()
        leader.port_handler.closePort()


if __name__ == "__main__":
    main()
