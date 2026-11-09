"""The simplest possible pick and place, built on what the visit_centres run proved (the arm lands within ~1-2 mm of the taught centres).

NOTHING is corrected or adjusted except the fixed finger's offset beside the piece. Per move:
  1. gripper opens to the piece's open_pct while the arm goes (at carry height) to the TAUGHT tip x, y of the source square
     (data/center_teach.json -> tip_board_mm, exactly, no finger offsets, no board map, no camera, no sag correction, no shift search);
  2. straight down to the piece's taught height (grip_tip in data/wooden_grasps.json);
  3. gripper closes to (contact_pct - squeeze);
  4. straight up to the carry height, across to the TAUGHT x, y of the destination square, straight down to grip_tip + place_extra;
  5. gripper opens to contact_pct + release_pct (capped at open_pct), straight up, back to the look pose.
Heights are board-frame mm (0 = the board surface as calibrated). A square without a taught centre is refused before anything moves.
"""
import json
import time

import chess
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot
from chess_player.motion.arm import Arm, FakeArm, look_pose
from chess_player.motion.ik import IK
from chess_player.motion.wooden_pick_place import CARRY_EXTRA_MM, SAFE_CARRY_BOTTOM_MM, recipe

SIDE_SPEED = 40.0  # mm/s sideways at carry height
UP_SPEED = 30.0
CLOSE_S = 1.0
# TEMPORARY test offsets (2026-11-09, user): grip 10 mm higher on the piece (the piece is carried and put down with the same extra height, otherwise it
# would be pushed into the board), every taught centre 5 mm toward -X (left), re-grip when the jaws close on nothing.
TEMP_DZ = 0.0  # was +10.0 (user's test): at 23.5 mm the pads were on the pawn's HEAD (30 mm tall); back to the taught 13.5 mm
TEMP_DX = -5.0
TEMP_PLACE_DZ = 8.0  # the place height only: 8 mm higher (user, 2026-11-09)
EMPTY_PCT = 3.0  # jaws closed to less than this = nothing in the hand
TRIES = 3
OPEN_S = 1.0


def taught_xy(square):
    d = json.loads((C.DATA / "center_teach.json").read_text())
    if square not in d:
        raise SystemExit(f"the centre of {square} was not taught (scripts/center_teach.sh --from {square})")
    # The PHYSICAL place is what was taught: tip_robot_mm (the arm's real tip in the robot frame). It is turned into board x, y with the
    # CURRENT board->robot calibration, so after a recalibration the arm still goes to the same physical spots (the stored board x, y were made
    # with the old calibration and would send it 5-15 mm elsewhere).
    from chess_player.calib.center_teach import board_from_robot

    xy = board_from_robot(BoardToRobot.load(), np.array(d[square]["tip_robot_mm"], float) / 1000.0)[:2]
    return np.array(xy, float) + np.array([TEMP_DX, 0.0])


def run(piece, src, dst, arm=None, log=print, home=True):
    """home=False: no trip to the look pose before/after (for a sequence of moves with one arm connection). Returns (legs, gripped)."""
    r = recipe(piece)
    gap = float(np.interp(r["contact_pct"], C._GAP_PCT, C._GAP_MM))  # jaw gap (mm) at the piece's contact width
    # The taught centre is where the CLOSED jaw tip was put. With the jaws open the FIXED finger must stand beside the piece, not on it
    # (at the centre it lands on the pawn's head). The finger offset from the centre is the piece's taught shift_x / shift_y (place: the held
    # piece rests against the fixed finger, so the tip goes gap/2 + place_dx beside the destination centre).
    p_src = taught_xy(src) + np.array([r["shift_x"], r["shift_y"]])
    p_dst = taught_xy(dst) + np.array([-gap / 2.0 + r["place_dx"], r["shift_y"] + r["place_dy"]])
    T, ik = BoardToRobot.load(), IK()
    z_pick = r["grip_tip"] + TEMP_DZ
    z_place = z_pick + r["place_extra"] + TEMP_PLACE_DZ
    close_pct = max(0.0, r["contact_pct"] - r["squeeze"])
    open_pct = min(r["open_pct"], C.GRIP_OPEN_MAX_PCT)
    release_pct = min(open_pct, r["contact_pct"] + r["release_pct"])
    look = {**look_pose(), "gripper": open_pct}
    seed = np.array([look[j] for j in C.ARM_JOINTS], float)
    seed[4] = C.WRIST_ROLL_DEG

    def ik_at(xy, z, sd):
        q, err, tilt = ik.solve(T(xy[0], xy[1], z), sd, C.GRASP_MAX_TILT)
        if err > C.IK_MAX_ERR_MM or tilt > C.GRASP_MAX_TILT:
            raise SystemExit(f"unreachable: IK error {err:.1f} mm, tilt {tilt:.0f} deg at ({xy[0]:.0f}, {xy[1]:.0f}, {z:.0f}); nothing moved")
        return q

    # carry height: the highest of the usual (always-high carry, +CARRY_EXTRA_MM) that the arm reaches over BOTH squares
    z_carry = SAFE_CARRY_BOTTOM_MM + CARRY_EXTRA_MM + z_pick
    while z_carry > z_pick + 10:
        try:
            for xy in (p_src, p_dst, (p_src + p_dst) / 2):
                ik_at(xy, z_carry, seed)
            break
        except SystemExit:
            z_carry -= 5.0
    else:
        raise SystemExit("the arm cannot carry anywhere above the grip height between these squares; nothing moved")

    def leg(sd, xy0, z0, xy1, z1, speed, grip):
        L = float(np.linalg.norm(np.r_[xy1 - xy0, z1 - z0]))
        n = max(1, int(np.ceil(L / 6.0)))
        wps = []
        for i in range(1, n + 1):
            f = i / n
            sd = ik_at(xy0 + (xy1 - xy0) * f, z0 + (z1 - z0) * f, sd)
            wps.append({**dict(zip(C.ARM_JOINTS, sd)), "gripper": grip})
        return wps, max(0.6, L / speed), sd

    # plan every leg first (nothing moves if something is unreachable)
    legs = []
    sd = seed
    first_q = ik_at(p_src, z_carry, seed)
    sd = first_q
    for name, a0, za, b0, zb, sp, g in (
        ("down to the piece", p_src, z_carry, p_src, z_pick, r["vert_speed"], open_pct),
        ("lift", p_src, z_pick, p_src, z_carry, r["vert_speed"], close_pct),
        ("carry", p_src, z_carry, p_dst, z_carry, SIDE_SPEED, close_pct),
        ("lower", p_dst, z_carry, p_dst, z_place, r["vert_speed"], close_pct),
        ("up", p_dst, z_place, p_dst, z_carry, UP_SPEED, release_pct),
    ):
        w, dur, sd = leg(sd, a0, za, b0, zb, sp, g)
        legs.append((name, w, dur))
    log(f"{piece.upper()} {src}->{dst}: taught tip {p_src.round(1).tolist()} -> {p_dst.round(1).tolist()} | pick z {z_pick:g}, place z {z_place:g}, carry z {z_carry:g} mm | "
        f"jaws open {open_pct:g} %, close to {close_pct:.1f} %, release {release_pct:.1f} %")
    if isinstance(arm, FakeArm):
        return legs, True
    arm = arm or Arm(use_sag_model=False)
    gripped = False
    try:
        if home:
            arm.move_to(look, 4.0)
        arm.move_to({**dict(zip(C.ARM_JOINTS, first_q)), "gripper": open_pct}, 4.0 if home else 3.0)
        arm.move_path([arm.joints()] + legs[0][1], legs[0][2])  # down to the piece, jaws open
        time.sleep(0.5)
        held = False
        for attempt in range(TRIES):
            arm.gripper(1.0, CLOSE_S)  # close fully: a piece stops the jaws at its width, an empty hand lets them shut
            time.sleep(0.6)
            p = float(arm.present()["gripper"])
            if p >= EMPTY_PCT:
                held = True
                arm.gripper(max(0.0, p - r["squeeze"]), 0.3)  # hold at the piece's width minus the squeeze
                time.sleep(0.4)
                log(f"  gripped at {p:.1f} % (attempt {attempt + 1})")
                break
            log(f"  attempt {attempt + 1}: the jaws closed to {p:.1f} %, nothing in the hand: opening and trying again")
            arm.gripper(open_pct, OPEN_S)
            time.sleep(0.6)
        if not held:
            log("  nothing gripped after %d tries: going up and home" % TRIES)
            arm.move_path([arm.joints()] + [{**w, "gripper": open_pct} for w in legs[1][1]], legs[1][2])
            if home:
                arm.move_to(look, 4.0)
            return legs, False
        for name, w, dur in legs[1:4]:  # lift, carry, lower (jaws stay at close_pct)
            arm.move_path([arm.joints()] + w, dur)
        time.sleep(0.3)
        arm.gripper(release_pct, OPEN_S)  # let go
        time.sleep(0.5)
        arm.move_path([arm.joints()] + legs[4][1], legs[4][2])  # straight up
        gripped = True
        if home:
            arm.move_to(look, 4.0)
    except KeyboardInterrupt:
        log("\nstopped, freezing the arm")
        arm.freeze()
        raise
    return legs, gripped
