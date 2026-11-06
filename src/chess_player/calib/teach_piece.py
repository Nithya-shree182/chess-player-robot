"""Teach the vision-guided grasp of one piece type from a pose you hold by hand. READ ONLY: nothing is moved or powered.

  python -m chess_player.calib.teach_piece measure --piece r    # step 1: the piece stands on the CENTRE of c2, arm out of view.
                                                                #   Measures how far YOLO's position for this piece is from the true centre.
  python -m chess_player.calib.teach_piece record  --piece r    # step 2: torque OFF, you hold the grasp on the piece (jaws closed
                                                                #   on it, tip where you want the grip). Reads the six joints.
Both write data/vision_grasps.json (the entry grasp_vision.py uses). Add --dry to print without saving.

record stores: the grasp depth (tip height you held + 2.8 mm margin, the same margin as the pawn), the jaw reading and gap you held,
the tip offset from the piece centre (minus half the gap along the jaw-opening direction: the tip frame is the FIXED finger's
face), the YOLO bias from `measure`, and the opening used to approach (the taught maximum).
"""
import argparse
import json
import time

import chess
import numpy as np

from chess_player import config as C
from chess_player.calib.locate_piece import grab, locate
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.ik import IK
from chess_player.teach.record_poses import FOLLOWER_CAL, Cal, make_bus, read_raw

OUT = C.DATA / "vision_grasps.json"
MARGIN_MM = 2.8
CENTRE = np.asarray(square_center_mm(chess.C2))


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def measure(a):
    got = locate(grab(str(C.EXT_CAMERA_URL)), chess.C2)
    if got is None:
        raise SystemExit("YOLO sees no piece on c2")
    sym, conf, xy, *_ = got
    print(f"YOLO sees '{sym}' on c2 (confidence {conf:.2f}) at board mm ({xy[0]:.1f}, {xy[1]:.1f}); the true centre is ({CENTRE[0]:.1f}, {CENTRE[1]:.1f})")
    if sym.lower() != a.piece:
        print(f"WARNING: expected '{a.piece}' but YOLO says '{sym.lower()}'. Not saved.")
        return
    bias = [round(float(xy[0] - CENTRE[0]), 1), round(float(xy[1] - CENTRE[1]), 1)]
    print(f"YOLO bias for this piece: {bias} mm (assuming it stands exactly on the centre)")
    if not a.dry:
        d = load()
        d.setdefault(a.piece, {})["yolo_base_point_bias_mm"] = bias
        OUT.write_text(json.dumps(d, indent=2))
        print("saved")


def record(a):
    fcal, T, ik = Cal(FOLLOWER_CAL), BoardToRobot.load(), IK()
    bus = make_bus(C.FOLLOWER_PORT, FOLLOWER_CAL)
    rows = []
    try:
        for _ in range(6):
            rows.append(fcal.units(read_raw(bus)))
            time.sleep(0.3)
    finally:
        bus.port_handler.closePort()
    R = np.array(rows)
    if np.ptp(R, axis=0).max() > 1.0:
        raise SystemExit("the arm is moving: hold it still and run again")
    u = R.mean(0)
    Tm = ik.fk(u[:5])
    tip = T.R.T @ (Tm[:3, 3] - T.t) * 1000
    tilt = float(np.degrees(np.arccos(np.clip(-Tm[2, 2], -1, 1))))
    d = T.R.T @ np.asarray(ik.jaw_open_direction(u[:5]))
    d = d / np.linalg.norm(d)
    gap = float(np.interp(u[5], C._GAP_PCT, C._GAP_MM))
    off = -d[:2] * gap / 2
    print(f"joints {dict(zip(C.ALL_JOINTS, np.round(u, 2).tolist()))}")
    print(f"tip board mm ({tip[0]:.1f}, {tip[1]:.1f}), HEIGHT {tip[2]:.1f} mm; tilt {tilt:.1f} deg; jaws {u[5]:.2f} % (gap ~{gap:.1f} mm)")
    print(f"jaw-opening direction ({d[0]:+.2f}, {d[1]:+.2f}); tip offset from the piece centre: ({off[0]:+.1f}, {off[1]:+.1f}) mm; depth to command: {tip[2] + MARGIN_MM:.1f} mm")
    if gap < 4 or gap > 30:
        print("WARNING: the jaw gap looks wrong for a held piece (closed on nothing, or wide open). Not saved.")
        return
    if a.dry:
        return
    dd = load()
    e = dd.setdefault(a.piece, {})
    e.update({"depth_mm": round(float(tip[2] + MARGIN_MM), 1), "open_pct": C.GRIP_OPEN_MAX_PCT,
              "tip_offset_from_piece_centre_mm": [round(float(off[0]), 1), round(float(off[1]), 1)],
              "hand_held_reading_pct": round(float(u[5]), 2), "hand_held_gap_mm": round(gap, 1),
              "carry_height_mm": e.get("carry_height_mm", 45.0), "place_extra_mm": 2.0, "grip": "hand-taught",
              "place_correction_mm": e.get("place_correction_mm", dd.get("p", {}).get("place_correction_mm", [0.0, 0.0])),
              "taught": time.strftime("%Y-%m-%d %H:%M:%S"),
              "notes": "taught by holding the arm on the piece; tip frame = fixed finger face"})
    e.setdefault("yolo_base_point_bias_mm", [0.0, 0.0])
    OUT.write_text(json.dumps(dd, indent=2))
    print(f"saved the '{a.piece}' grasp to {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["measure", "record"])
    ap.add_argument("--piece", required=True, choices=list("pnbrqk"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    {"measure": measure, "record": record}[a.cmd](a)


if __name__ == "__main__":
    main()
