"""Teach / test the grasp of ONE piece type on the wooden rig. It runs the SAME code as the game (motion/wooden_pick_place.py), so what
you tune and save here is exactly what the game then does.

  python -m chess_player.calib.pick_place e7 e5 --piece n                      # the arm moves (asks Enter first)
  python -m chess_player.calib.pick_place e7 e5 --piece q --dry-run            # simulated arm: reachability + plan
  python -m chess_player.calib.pick_place e7 e5 --piece q --grip-tip 24 --squeeze 2.5 --save     # try numbers, keep them
  python -m chess_player.calib.pick_place e2 e4 --piece p --others "d3:p f3:P e6:K"   # plan/run with neighbours standing on the board

Numbers (mm unless noted), from the CENTRE of the named square:
  shift-x/-y   tip (the FIXED finger's inner face) offset from the piece centre; negative x = toward the a-file
  grip-tip     tip height above the board while gripping (tall pieces: grip higher on the body)
  open-pct     jaw opening on the way down;  squeeze  % tighter than the contact reading while carrying (harder = safer hold)
  vert-speed   mm/s for lifting and lowering (slow for tall pieces);  lift  minimum lift above the grip
  place-dx/-dy centring correction at the place square;  place-extra  how much above grip-tip the piece is released
The piece is always lifted straight up to the safe carry height (bottom of the piece >= 70 mm) before any sideways move whenever
other pieces are on the board. --save stores the numbers used for that piece in data/wooden_grasps.json and marks it taught.
The printed contact reading / gap = the width of the part gripped (2-3 % = closed on air).
"""
import argparse
import json
import time

import chess
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.arm import Arm, FakeArm, look_pose
from chess_player.motion.ik import IK
from chess_player.motion.pick_place import Transfer
from chess_player.motion.wooden_pick_place import FILE, PAWN, WoodenPickPlace, recipe

KEYS = list(PAWN)  # every tunable number


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--piece", required=True, choices=list("pnbrqk"))
    for k in KEYS:
        ap.add_argument("--" + k.replace("_", "-"), type=float, help="default: the saved / starting value")
    ap.add_argument("--others", nargs="*", default=[], help='other pieces on the board, e.g. "d3:p f3:P" (square:symbol, capital = white)')
    ap.add_argument("--save", action="store_true", help="store the numbers used in data/wooden_grasps.json for this piece")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retries", type=int, default=4, help="extra attempts after the jaws close on nothing (each shifts the pick point a little)")
    ap.add_argument("--no-camera", action="store_true", help="do not correct the pick point with the camera (taught numbers only)")
    ap.add_argument("--yes", action="store_true", help="do not ask for Enter")
    a = ap.parse_args()

    r = recipe(a.piece)
    over = {k: getattr(a, k) for k in KEYS if getattr(a, k) is not None}
    r.update(over)
    print(f"{a.piece.upper()} {a.src} -> {a.dst}  ({'TAUGHT numbers' if r['taught'] or over else 'starting numbers, not taught'})")
    print("  " + ", ".join(f"{k}={r[k]:g}" for k in KEYS))

    src = tuple(square_center_mm(chess.parse_square(a.src)))
    dst = tuple(square_center_mm(chess.parse_square(a.dst)))
    obstacles = {}
    for o in a.others:
        sq, sym = o.split(":")
        obstacles[tuple(square_center_mm(chess.parse_square(sq)))] = sym
    t = Transfer(a.piece, src, dst, f"{a.piece} {a.src}-{a.dst}")
    T, ik = BoardToRobot.load(), IK()
    arm = FakeArm() if a.dry_run else Arm()
    pp = WoodenPickPlace(arm, ik, T, log=print)
    pp.override = over
    if not a.dry_run:  # a picture of the grip point before the jaws close: data/pick_debug/
        import time as _t
        import urllib.request as _u

        _dbg = C.DATA / "pick_debug"
        _dbg.mkdir(exist_ok=True)

        def _snap(tag):
            try:
                (_dbg / f"{_t.strftime('%H%M%S')}_{a.piece}_{a.src}_{tag}.jpg").write_bytes(_u.urlopen(str(C.EXT_CAMERA_URL).replace("/video", "/shot"), timeout=5).read())
            except Exception as e:
                print("  (no picture:", e, ")")

        pp.snap = _snap
    try:
        if a.dry_run:
            print(pp.plan_transfer(t, obstacles or None)["info"])
        else:
            if not a.yes:
                input("Only the piece on the source square (and the --others you listed). Hand near the power switch. Enter = go: ")
            ref = r.get("cam_ref")
            if ref is not None and not a.no_camera:  # measure the piece with the arm at home, out of the camera's view
                from chess_player.vision.ext_camera import ExtCamera
                from chess_player.vision.piece_locator import PieceLocator

                cam = ExtCamera()
                try:
                    time.sleep(1.5)
                    m = PieceLocator(cam, log=print).measure(np.array(src, float), a.piece)
                finally:
                    cam.close()
                if m is not None:
                    pp._cam_dev = {tuple(src): m - np.array(ref, float)}
            print(pp.execute(t, retries=a.retries, obstacles=obstacles or None))
            arm.move_to({**look_pose(), "gripper": C.GRIPPER_LOOK}, 3.0)
        print("done. Off-centre at the place square: adjust --place-dx / --place-dy (+x = toward the h-file, +y = toward rank 8).")
    finally:
        arm.close()
    if a.save and not a.dry_run:
        try:
            d = json.loads(FILE.read_text())
        except (OSError, ValueError):
            d = {}
        d[a.piece] = {**d.get(a.piece, {}), **{k: r[k] for k in KEYS}}  # keeps cam_ref / complete
        FILE.write_text(json.dumps(d, indent=2))
        print("saved", FILE)


if __name__ == "__main__":
    main()
