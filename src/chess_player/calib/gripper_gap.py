"""Measure the gap between the finger pads for several gripper readings, with a ruler. Only the gripper moves.

  python -m chess_player.calib.gripper_gap

The gripper goes to each setting in turn. Measure the clear gap between the INNER faces of the two finger pads at their
tips (the part that touches a piece), in millimetres, and type it in. Saves data/gripper_gap.json, which the top grip
uses to turn "open the jaws X mm" into the gripper reading. 0 % should be about 0 mm (jaws closed).

Keep your hands off the fingers while it moves; it only drives the gripper motor (the arm joints are not touched).
"""
import json

from chess_player import config as C

SETTINGS = [0, 5, 10, 15, 20, 25, 30, 40, 50]


def main():
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    cal_path = C.Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower" / f"{C.FOLLOWER_ID}.json"
    cal = json.loads(cal_path.read_text())
    bus = FeetechMotorsBus(C.FOLLOWER_PORT, {"gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100)},
                           calibration={"gripper": MotorCalibration(**cal["gripper"])})
    bus.connect(handshake=True)
    pct, gap = [], []
    try:
        now = float(bus.sync_read("Present_Position")["gripper"])
        bus.write("Goal_Position", "gripper", now)  # hold where it is, then enable: nothing jumps
        bus.write("Torque_Enable", "gripper", 1)
        for p in SETTINGS:
            bus.write("Goal_Position", "gripper", float(p))
            import time

            time.sleep(1.2)
            reading = float(bus.sync_read("Present_Position")["gripper"])
            while True:
                s = input(f"gripper set to {p} % (reads {reading:.1f} %). Measured gap in mm (Enter = skip this one): ").strip()
                if not s:
                    break
                try:
                    g = float(s)
                    pct.append(round(reading, 2))
                    gap.append(g)
                    break
                except ValueError:
                    print("   type a number, e.g. 14.5")
        bus.write("Goal_Position", "gripper", 0.0)
        time.sleep(1.0)
    finally:
        bus.write("Torque_Enable", "gripper", 0)
        bus.port_handler.closePort()
    if len(pct) < 4:
        raise SystemExit("need at least 4 measurements: nothing saved")
    order = sorted(range(len(pct)), key=lambda i: pct[i])
    out = {"pct": [pct[i] for i in order], "gap_mm": [gap[i] for i in order]}
    if any(b < a for a, b in zip(out["gap_mm"], out["gap_mm"][1:])):
        print("WARNING: the gaps do not grow with the setting: check the measurements")
    C.GAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    C.GAP_FILE.write_text(json.dumps(out, indent=2))
    print("saved", C.GAP_FILE, out)
    slope = (out["gap_mm"][-1] - out["gap_mm"][0]) / (out["pct"][-1] - out["pct"][0])
    print(f"about {slope:.2f} mm of gap per 1 % (the old rig: 1.78)")


if __name__ == "__main__":
    main()
