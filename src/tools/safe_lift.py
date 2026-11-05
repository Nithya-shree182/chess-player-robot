"""Recovery: from wherever the follower is, lift the jaw tip straight up 90 mm, then go to the saved look/home pose.

  python src/tools/safe_lift.py

Use it after a failed move (the arm is left holding wherever the grasp stopped). Gentle and slow; nothing sweeps sideways
at board height. Close every other arm program first (only one can own the port).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from chess_player import config as C  # noqa: E402
from chess_player.motion.arm import Arm, look_pose  # noqa: E402
from chess_player.motion.ik import IK  # noqa: E402

arm = Arm()  # holds the present position, torque on, nothing jumps
ik = IK(wrist_roll_deg=C.WRIST_ROLL_DEG)
try:
    now = arm.present()
    q = np.array([now[j] for j in C.ARM_JOINTS])
    print("present joints:", np.round(q, 1))
    p = ik.fk(q)[:3, 3]
    for dz in (0.03, 0.06, 0.09):  # straight up in three steps
        sol, err, tilt = ik.solve(p + np.array([0, 0, dz]), q)
        print(f"lift {dz * 1000:.0f} mm: IK err {err:.1f} mm, tilt {tilt:.0f} deg")
        if err > 5:
            print("IK cannot lift cleanly; stopping the lift here")
            break
        arm.move_to(dict(zip(C.ARM_JOINTS, sol)), 1.5)
        q = sol
    print("going to the look pose")
    arm.move_to({**look_pose(), "gripper": C.GRIPPER_LOOK}, 4.0)
    arm.settle()
    print("final joints:", {j: round(v, 1) for j, v in arm.present().items()})
finally:
    arm.close()
