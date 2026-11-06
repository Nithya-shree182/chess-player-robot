"""Read an arm's joints (READ ONLY, nothing moves or is powered) and say where the jaw tip is, in board mm, and how that
compares with where YOLO sees a piece. Hold the arm in your own top-down pose over a piece and run this:

  python -m chess_player.calib.compare_pose --square c2                 # leader arm (the one you move by hand)
  python -m chess_player.calib.compare_pose --square c2 --source follower
  python -m chess_player.calib.compare_pose --square c2 --watch         # live, once a second; Ctrl+C to stop
  python -m chess_player.calib.compare_pose --square c2 --save          # append the reading to data/pose_checks.json

It prints (1) the joints as the FOLLOWER would take them (the leader's reading mapped through the two calibrations, the
same mapping teleop uses), (2) the jaw tip in board mm, its height and the tool's tilt from vertical, (3) the offset between
the tip and the centre of the piece that YOLO sees on that square, and (4) the joints the maths wants for a straight-down tip
over that piece, with the difference. If you held the tip exactly over the piece, (3) is the board->robot calibration error.
"""
import argparse
import json
import time

import chess
import numpy as np

from chess_player import config as C
from chess_player.calib.locate_piece import grab, locate
from chess_player.geometry import BoardToRobot, mm_to_square
from chess_player.motion.ik import IK
from chess_player.teach.record_poses import FOLLOWER_CAL, LEADER_CAL, Cal, J, make_bus, read_raw

OUT = C.DATA / "pose_checks.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--square", default="c2", help="the piece you are holding the arm over")
    ap.add_argument("--source", choices=["leader", "follower"], default="leader")
    ap.add_argument("--url", default=str(C.EXT_CAMERA_URL))
    ap.add_argument("--clearance", type=float, default=25.0, help="for the maths pose: tip height above the piece's top")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    sq = chess.parse_square(a.square)
    got = locate(grab(a.url), sq)
    if got is None:
        raise SystemExit(f"YOLO sees no piece on {a.square}")
    sym, conf, xy, _box, _view, _base = got
    T, ik = BoardToRobot.load(), IK()
    fcal, lcal = Cal(FOLLOWER_CAL), Cal(LEADER_CAL)
    bus = make_bus(C.LEADER_PORT if a.source == "leader" else C.FOLLOWER_PORT, LEADER_CAL if a.source == "leader" else FOLLOWER_CAL)
    try:
        want_z = C.PIECES[sym.lower()]["height"] + a.clearance
        seed = np.array([0.0, 0.0, 0.0, 0.0, C.WRIST_ROLL_DEG])
        math_q, _err, _tilt = ik.solve(T(xy[0], xy[1], want_z), seed, C.GRASP_MAX_TILT)
        print(f"piece: {sym} on {a.square} (YOLO {conf:.2f}) at board mm x {xy[0]:.1f}, y {xy[1]:.1f}")
        while True:
            raw = read_raw(bus)
            units = fcal.units(fcal.from_frac(lcal.frac(raw))) if a.source == "leader" else fcal.units(raw)
            q = units[:5]
            Tm = ik.fk(q)
            pb = T.R.T @ (Tm[:3, 3] - T.t) * 1000
            tilt = float(np.degrees(np.arccos(np.clip(-Tm[2, 2], -1, 1))))
            over = mm_to_square(pb[0], pb[1])
            d = pb[:2] - np.asarray(xy)
            print(f"\n{a.source} joints as the follower takes them (deg, gripper %): "
                  + ", ".join(f"{j} {v:.1f}" for j, v in zip(J, units)))
            print(f"  jaw tip in board mm: x {pb[0]:.1f}, y {pb[1]:.1f}, height {pb[2]:.1f}; over square "
                  f"{chess.square_name(over) if over is not None else 'off the board'}; tool tilt {tilt:.1f} deg from vertical")
            print(f"  tip minus piece centre: dx {d[0]:+.1f} mm, dy {d[1]:+.1f} mm  (distance {np.hypot(*d):.1f} mm)")
            print("  maths pose for a straight-down tip over the piece at "
                  f"{want_z:.0f} mm: " + ", ".join(f"{j} {v:.1f}" for j, v in zip(C.ARM_JOINTS, math_q))
                  + "   | yours minus maths: " + ", ".join(f"{v:+.1f}" for v in (q - math_q)))
            if a.save:
                log = json.loads(OUT.read_text()) if OUT.exists() else []
                log.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "square": a.square, "piece": sym, "source": a.source,
                            "yolo_board_mm": [round(float(xy[0]), 1), round(float(xy[1]), 1)],
                            "tip_board_mm": [round(float(v), 1) for v in pb], "tilt_deg": round(tilt, 1),
                            "joints": [round(float(v), 2) for v in units]})
                OUT.write_text(json.dumps(log, indent=1))
                print(f"  saved to {OUT}")
            if not a.watch:
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            bus.port_handler.closePort()
        except Exception:
            pass


if __name__ == "__main__":
    main()
