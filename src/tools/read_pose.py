"""Print the follower's joint positions and torque state (read-only; doesn't move or lock the arm). Windows and Linux.

  python src/tools/read_pose.py
  python src/tools/read_pose.py --save data/look_pose.json data/poses/home_pose.json   (also save it as the home/look pose)

Replaces src/tools/read_positions.py (hard-coded Linux port). Port and calibration id come from config.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chess_player import config as C  # noqa: E402
from lerobot.motors import Motor, MotorCalibration, MotorNormMode  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402

CAL = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower" / f"{C.FOLLOWER_ID}.json"
NAMES = list(C.ALL_JOINTS)

ap = argparse.ArgumentParser()
ap.add_argument("--save", nargs="*", default=[], help="JSON file(s) to write the pose to")
ap.add_argument("--desc", default="")
a = ap.parse_args()

cal = {k: MotorCalibration(**v) for k, v in json.loads(CAL.read_text()).items()}
motors = {n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if n == "gripper" else MotorNormMode.DEGREES)
          for i, n in enumerate(NAMES)}
bus = FeetechMotorsBus(C.FOLLOWER_PORT, motors, calibration=cal)
bus.connect(handshake=True)
raw = bus.sync_read("Present_Position", normalize=False)
norm = bus.sync_read("Present_Position")
torque = bus.sync_read("Torque_Enable", normalize=False)
bus.port_handler.closePort()
for n in NAMES:
    print(f"{n:14} raw={raw[n]:5d} value={norm[n]:8.2f} torque={'on' if torque[n] else 'off'}")
for path in a.save:
    out = {"name": Path(path).stem, "description": a.desc, "recorded": time.strftime("%Y-%m-%d"), "robot_id": C.FOLLOWER_ID,
           "raw": raw, "degrees": {n: round(float(norm[n]), 2) for n in NAMES if n != "gripper"},
           "gripper_percent": round(float(norm["gripper"]), 2)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2))
    print("saved", path)
