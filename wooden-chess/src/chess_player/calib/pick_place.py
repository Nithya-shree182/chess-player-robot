"""Pick a piece from one square and put it on another, with the recipe verified on the wooden pawn (e7 -> e5, 2026-11-07).

  python -m chess_player.calib.pick_place e7 e5 --piece p                 # the arm moves (asks Enter first)
  python -m chess_player.calib.pick_place e7 e5 --piece r --dry-run       # simulated arm: checks reachability and prints the plan
  python -m chess_player.calib.pick_place e7 e5 --piece n --grip-tip 18 --shift-x -12 --save    # try other numbers, keep them

The recipe (all board mm, from the CENTRE of the named square; the piece must stand on the centre):
  1. travel at the safe height over the pick point = centre + (shift_x, shift_y): the tip (the FIXED finger's inner face) sits shift_x toward
     the a-file (negative) of the piece, jaws open (open_pct), so the piece ends up between the fingers
  2. straight down to the grip height (tip target above the board), SLOW close until the pads touch (contact), light squeeze
  3. lift lift_mm straight up, carry in a straight line, lower
  4. the held piece sits against the fixed finger, half the contact gap from the tip: so the tip goes to centre - gap/2 + place_dx,
     (place_dx = the measured centring correction), same y, then down to grip_tip + place_extra, SLOW release, straight up, home.
Numbers per piece type live in data/wooden_grasps.json (defaults below = the pawn); `--save` stores the flags you passed for that piece.
The contact reading and the gap it means are printed: that tells you the width of the part of the piece that was gripped.
"""
import argparse
import json
import time

import chess
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.arm import Arm, FakeArm, look_pose
from chess_player.motion.ik import IK, solve_or_raise
from chess_player.motion.slow_grip import slow_close, slow_release

FILE = C.DATA / "wooden_grasps.json"
PAWN = dict(open_pct=16.1, shift_x=-10.6, shift_y=2.7, grip_tip=15.0, lift=30.0, place_dx=3.0, place_dy=0.0, place_extra=2.0)
PATH_STEP_MM = 6.0
CARRY_MM_PER_S = 25.0


def recipe(piece):
    d = dict(PAWN)
    if piece != "p":  # guess: the fixed finger a little clear of the base; tune with the flags
        d["shift_x"] = -C.PIECES[piece]["diameter"] / 2.0
    try:
        d.update(json.loads(FILE.read_text()).get(piece, {}))
    except (OSError, ValueError):
        pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--piece", required=True, choices=list("pnbrqk"))
    for k in PAWN:
        ap.add_argument("--" + k.replace("_", "-"), type=float, help=f"default {PAWN[k]} (pawn) or the saved value")
    ap.add_argument("--save", action="store_true", help="store the numbers used in data/wooden_grasps.json for this piece")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="do not ask for Enter")
    a = ap.parse_args()
    r = recipe(a.piece)
    for k in PAWN:
        v = getattr(a, k)
        if v is not None:
            r[k] = v
    H = C.PIECES[a.piece]["height"]
    src = np.array(square_center_mm(chess.parse_square(a.src)), float)
    dst = np.array(square_center_mm(chess.parse_square(a.dst)), float)
    T, ik = BoardToRobot.load(), IK()
    look = {**look_pose(), "gripper": C.GRIPPER_LOOK}
    open_pct = min(r["open_pct"], C.GRIP_OPEN_MAX_PCT)
    pick_xy = src + np.array([r["shift_x"], r["shift_y"]])
    z_above = H + 25.0

    print(f"{a.piece.upper()} {a.src} -> {a.dst}: pick tip ({pick_xy[0]:.1f}, {pick_xy[1]:.1f}) grip height {r['grip_tip']:.0f} mm, jaws {open_pct:.1f} %, lift {r['lift']:.0f} mm")
    arm = FakeArm() if a.dry_run else Arm()
    if not a.dry_run and not a.yes:
        input("Pieces: only the one on the source square near its path. Hand near the power switch. Enter = go: ")
    seed = np.array([look[j] for j in C.ARM_JOINTS])
    seed[4] = C.WRIST_ROLL_DEG

    def ik_at(xy, z, what):
        nonlocal seed
        q = solve_or_raise(ik, T(xy[0], xy[1], z), seed, C.GRASP_MAX_TILT, what=what)
        seed = q
        return q

    def safe_z(xy):
        """The travel height over xy: the standard one, or the highest the arm can reach there (far corners need it lower)."""
        for z in np.arange(C.Z_SAFE, z_above - 1.0, -5.0):
            q, err, tilt = ik.solve(T(xy[0], xy[1], z), seed, C.GRASP_MAX_TILT)
            if err <= C.IK_MAX_ERR_MM and tilt <= C.GRASP_MAX_TILT:
                return float(z)
        raise SystemExit(f"cannot reach {xy} at any height above the piece")

    def pose(q, grip):
        return {**dict(zip(C.ARM_JOINTS, q)), "gripper": float(grip)}

    def line(xy0, z0, xy1, z1, what):
        """Straight Cartesian line (IK every PATH_STEP_MM) from where the arm is, held at the current gripper opening."""
        grip = float(arm.joints()["gripper"])
        L = float(np.linalg.norm(np.r_[xy1 - xy0, z1 - z0]))
        n = max(1, int(np.ceil(L / PATH_STEP_MM)))
        wps = [arm.joints()]
        for i in range(1, n + 1):
            f = i / n
            wps.append(pose(ik_at(xy0 + (xy1 - xy0) * f, z0 + (z1 - z0) * f, what), grip))
        arm.move_path(wps, max(0.8, L / CARRY_MM_PER_S))
        arm.settle()

    try:
        z_hi = safe_z(pick_xy)
        hi = ik_at(pick_xy, z_hi, "pick, safe height")
        arm.move_to(pose(hi, open_pct))
        arm.settle()
        line(pick_xy, z_hi, pick_xy, z_above, "pick, above the piece")
        line(pick_xy, z_above, pick_xy, r["grip_tip"], "pick, grip height")
        contact, fin = (8.3, 8.2) if a.dry_run else slow_close(arm)
        if contact is None:
            print(f"  NO CONTACT (jaws closed to {fin:.1f} %): nothing in the hand. Aborting; lifting.")
            slow_release(arm, open_pct)
            line(pick_xy, r["grip_tip"], pick_xy, z_hi, "abort lift")
            arm.move_to(look, 3.0)
            raise SystemExit(1)
        gap = float(np.interp(contact, C._GAP_PCT, C._GAP_MM))
        print(f"  gripped: contact at {contact:.1f} % = {gap:.1f} mm gap (the width of the gripped part), squeezing to {fin:.1f} %")
        z_c = r["grip_tip"] + r["lift"]
        line(pick_xy, r["grip_tip"], pick_xy, z_c, "lift")
        place_xy = dst + np.array([-gap / 2.0 + r["place_dx"], r["shift_y"] + r["place_dy"]])
        print(f"  carrying to ({place_xy[0]:.1f}, {place_xy[1]:.1f}) (square centre {dst[0]:.0f}, {dst[1]:.0f}); place height {r['grip_tip'] + r['place_extra']:.0f} mm")
        line(pick_xy, z_c, place_xy, z_c, "carry")
        line(place_xy, z_c, place_xy, r["grip_tip"] + r["place_extra"], "lower")
        slow_release(arm, open_pct)
        line(place_xy, r["grip_tip"] + r["place_extra"], place_xy, z_c, "lift away")
        line(place_xy, z_c, place_xy, safe_z(place_xy), "to travel height")
        arm.move_to(look, 3.0)
        print("done. If the piece is off-centre, adjust --place-dx / --place-dy (mm; +x = toward the h-file, +y = toward rank 8), then rerun with --save.")
    finally:
        arm.close()
    if a.save:
        try:
            d = json.loads(FILE.read_text())
        except (OSError, ValueError):
            d = {}
        d[a.piece] = {k: r[k] for k in PAWN}
        FILE.write_text(json.dumps(d, indent=2))
        print("saved", FILE)


if __name__ == "__main__":
    main()
