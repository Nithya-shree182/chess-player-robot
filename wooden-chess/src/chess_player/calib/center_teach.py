"""Teach the centre of a square, per piece, by hand-guiding the follower with the leader arm (board EMPTY).

  scripts/center_teach.sh

The squares come in a fixed order, h1 g1 ... a1, h2 ... a8 (already saved ones are skipped; --from d3 restarts there). Nothing to type: Enter saves and moves on. The follower
first glides to the leader's pose over 3 s (keep the leader where the follower is, or near it), then copies it. Put the gripper tip where it should be to take that piece
from the CENTRE of that square, press Enter: the follower's real joint angles, the tip position (robot frame) and the tip position in
board coordinates are saved in data/center_teach.json under "<square>", with the offset from the nominal square centre
(robot frame: +X = right, toward the h-file; +Y = forward, toward rank 8; Z = tip height above the board surface).
'q' + Enter (or Ctrl+C) quits. Height / grip / speed are NOT decided here.
"""
import json
import select
import sys
import time
from datetime import datetime

import chess
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.arm import Arm
from chess_player.motion.ik import IK

FILE = C.DATA / "center_teach.json"


def find_leader():
    # NOT imported from teach.record_demo: importing that module starts a background thread that swallows everything typed on stdin
    # (that was why the first version never saw the square / the Enter).
    import glob
    import os

    if C.Path("/dev/so100_leader").exists():
        return "/dev/so100_leader"
    follower = os.path.realpath(C.FOLLOWER_PORT)
    others = [p for p in sorted(glob.glob("/dev/ttyACM*")) if os.path.realpath(p) != follower]
    if not others:
        raise SystemExit("leader arm not found: plug in its USB cable")
    return others[0]


def board_from_robot(T: BoardToRobot, p_robot_m):
    """Inverse of BoardToRobot.__call__ (board mm x, y, z) for a robot-frame point in metres."""
    x = y = 0.0
    z = 0.0
    for _ in range(4):  # z_plane makes the map depend on (x, y): iterate
        dz = 0.0
        if T.z_plane is not None:
            dz = (T.z_plane[0] + T.z_plane[1] * (x - 105.0) + T.z_plane[2] * (y - 105.0)) / 1000.0
        b = T.R.T @ (np.asarray(p_robot_m) - T.t - np.array([0.0, 0.0, dz])) * 1000.0
        if T.A is not None:
            xy = np.linalg.solve(T.A, b[:2] - T.b)
        else:
            xy = b[:2]
        x, y, z = float(xy[0]), float(xy[1]), float(b[2])
    return np.array([x, y, z])


def main():
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

    leader = SOLeader(SOLeaderTeleopConfig(port=find_leader(), id=C.LEADER_ID))
    leader.connect(calibrate=False)
    arm = Arm(use_sag_model=False)
    ik = IK()
    T = BoardToRobot.load()
    data = json.loads(FILE.read_text()) if FILE.exists() else {}
    print(f"saved so far: {len(data)} entries in {FILE}")

    last = {"goal": None, "bad": 0}

    def leader_goal():
        """The leader's pose. A single failed serial read (a USB hiccup: 'no status packet') keeps the last pose instead of crashing."""
        try:
            act = leader.get_action()
        except ConnectionError as e:
            last["bad"] += 1
            if last["goal"] is None or last["bad"] > 40:
                raise
            if last["bad"] in (1, 10):
                print(f"\n  (leader read glitch, keeping the last pose: {str(e)[:60]})", flush=True)
            time.sleep(0.02)
            return last["goal"]
        last["bad"] = 0
        last["goal"] = {j: float(act[f"{j}.pos"]) for j in C.ALL_JOINTS}
        return last["goal"]

    def follower_present():
        for _ in range(5):
            try:
                return arm._read_present()
            except ConnectionError:
                time.sleep(0.05)
        raise ConnectionError("follower does not answer")

    try:
        order = [chess.square_name(chess.square(f, r)) for r in range(8) for f in range(7, -1, -1)]  # h1 g1 ... a1, h2 g2 ... a8
        todo = [q for q in order if q not in data]
        if "--from" in sys.argv:  # --from d3: restart the sequence there (overwrites what is saved from there on)
            todo = order[order.index(sys.argv[sys.argv.index("--from") + 1].lower()):]
        if not todo:
            print("every square is already saved")
            return
        print(f"sequence: {' '.join(todo[:12])}{' ...' if len(todo) > 12 else ''}   ({len(todo)} squares to go)")
        print("Enter = save this square and go to the next | 'skip' = leave this square out | 'back' = redo the previous square | 'q' = quit")
        arm.move_to(leader_goal(), 3.0)  # glide to the leader's pose once, then just follow it
        t_last = 0.0
        i = 0
        while i < len(todo):
            sq = todo[i]
            sq_i = chess.parse_square(sq)
            print(f"\n[{i + 1}/{len(todo)}] TELEOP LIVE: put the tip on the CENTRE of {sq.upper()}, then press Enter.", flush=True)
            ans = None
            while ans is None:
                arm._write(leader_goal())
                time.sleep(1.0 / C.CONTROL_HZ)
                if time.time() - t_last > 0.7:  # live readout so you can see it is running
                    t_last = time.time()
                    bq = board_from_robot(T, ik.fk([follower_present()[j] for j in C.ARM_JOINTS])[:3, 3])
                    print(f"\r  tip: x {bq[0]:6.1f}  y {bq[1]:6.1f}  height {bq[2]:5.1f} mm   ({sq} centre: x {square_center_mm(sq_i)[0]:.1f}, y {square_center_mm(sq_i)[1]:.1f})   ",
                          end="", flush=True)
                if select.select([sys.stdin], [], [], 0)[0]:
                    ans = sys.stdin.readline().strip().lower()
            print()
            if ans == "q":
                break
            if ans == "skip":
                i += 1
                continue
            if ans == "back":
                i = max(0, i - 1)
                continue
            time.sleep(0.3)
            q = follower_present()
            qa = [q[j] for j in C.ARM_JOINTS]
            p_m = ik.fk(qa)[:3, 3]
            b = board_from_robot(T, p_m)
            nom = square_center_mm(sq_i)
            d = b[:2] - nom
            data[sq] = {
                "piece": "*", "square": sq, "when": datetime.now().isoformat(timespec="seconds"),
                "joints": {j: round(float(v), 2) for j, v in q.items()},
                "tip_robot_mm": [round(float(v) * 1000, 2) for v in p_m],
                "tip_board_mm": [round(float(v), 2) for v in b],
                "square_centre_board_mm": [round(float(v), 2) for v in nom],
                "offset_from_centre_mm": {"x": round(float(d[0]), 2), "y": round(float(d[1]), 2)},
                "tip_height_above_board_mm": round(float(b[2]), 2),
                "gripper_pct": round(float(q["gripper"]), 2),
            }
            if FILE.exists():  # a dated copy before every save, so a bad save can never lose taught centres
                bk = FILE.parent / "center_teach_backups"
                bk.mkdir(exist_ok=True)
                (bk / f"center_teach.{datetime.now().strftime('%Y%m%d-%H%M%S')}.json").write_text(FILE.read_text())
            FILE.write_text(json.dumps(data, indent=1))
            print(f"  saved {sq}: {d[0]:+.1f} mm X (right), {d[1]:+.1f} mm Y (forward) from the nominal centre, "
                  f"{b[2]:.1f} mm above the board ({len(data)} saved)")
            i += 1
        print("sequence finished" if i >= len(todo) else "stopped")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        try:
            arm.freeze()
        except Exception:
            pass
        arm.close()
        print("done:", FILE)


if __name__ == "__main__":
    main()
