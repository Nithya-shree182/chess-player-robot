"""Switch the follower's motor torque off or on (works on Windows and Linux; port and id come from config.py).

  python src/tools/torque.py gripper     only the gripper goes limp (open/close it by hand); the arm keeps holding
  python src/tools/torque.py all         EVERY joint goes limp: the arm drops, hold it first!
  python src/tools/torque.py on          torque back on, holding wherever each joint is right now (nothing jumps)

Close any other program that uses the arm first (the pipeline, move_piece, ...): only one can have the port.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chess_player import config as C  # noqa: E402
from lerobot.motors import Motor, MotorCalibration, MotorNormMode  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402

what = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
if what not in ("gripper", "all", "on"):
    raise SystemExit(__doc__)

cal_path = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower" / f"{C.FOLLOWER_ID}.json"
cal = {k: MotorCalibration(**v) for k, v in json.loads(cal_path.read_text()).items()}
motors = {j: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if j == "gripper" else MotorNormMode.DEGREES)
          for i, j in enumerate(C.ALL_JOINTS)}
bus = FeetechMotorsBus(C.FOLLOWER_PORT, motors, calibration=cal)
bus.connect(handshake=True)
try:
    if what == "on":
        raw = bus.sync_read("Present_Position", normalize=False)
        bus.sync_write("Goal_Position", raw, normalize=False)  # hold exactly where it is, then enable: no jump
        for j in C.ALL_JOINTS:
            bus.write("Torque_Enable", j, 1)
        print("torque ON: every joint is holding where it is")
    elif what == "gripper":
        bus.write("Torque_Enable", "gripper", 0)
        print("gripper torque OFF: you can open and close it by hand. The arm joints are still holding.")
        print("When you are done:  python src/tools/torque.py on")
    else:
        for j in C.ALL_JOINTS:
            bus.write("Torque_Enable", j, 0)
        print("ALL torque OFF: the arm is limp. Support it.")
        print("When you are done:  python src/tools/torque.py on")
finally:
    bus.port_handler.closePort()
