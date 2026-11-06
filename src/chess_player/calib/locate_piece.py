"""Find a piece with the board camera + YOLO, compute where it is for the ROBOT, and (optionally) hover the jaws over it.

  python -m chess_player.calib.locate_piece --square c2                 # dry run: prints the maths, moves nothing
  python -m chess_player.calib.locate_piece --square c2 --go            # the arm moves (asks Enter first)
  python -m chess_player.calib.locate_piece --square c2 --go --clearance 25   # tip this many mm above the piece's top

The chain (all numbers are printed):
  1. YOLO box of the piece -> its base point in the picture (bottom-centre of the box, nudged up: the box bottom is the
     front edge of the base, not its centre)
  2. picture -> board mm: the grid homography (the board is a plane, and the base stands on it, so this is exact for the base)
  3. board mm -> robot frame: data/board_to_robot.json (the 17-point touch fit; its residual is the main error source)
  4. inverse kinematics for the jaw tip at (x, y, z), tool pointing straight down (tilt as small as the arm allows), wrist roll
     fixed at the play angle, jaws open sideways.
With --go the arm goes above the piece at the safe height, then down to the hover height, and HOLDS there (torque on) until
you press Enter, so you can compare it with your own top-down pose; then it goes back up and home.
The gripper stays CLOSED (a closed jaw is the narrowest) unless --open is given; the largest opening ever allowed is
config.GRIP_OPEN_MAX_PCT (taught by hand).
"""
import argparse
import time
import urllib.request

import chess
import cv2
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot, mm_to_square, square_center_mm
from chess_player.motion.ik import IK, solve_or_raise
from chess_player.vision import detect_board as D
from chess_player.vision import ext_camera as E


def grab(url):
    if url.startswith("http") and url.rstrip("/").endswith("video"):  # the camera window also serves single pictures
        url = url.rsplit("/", 1)[0] + "/shot"
        data = urllib.request.urlopen(url, timeout=5).read()
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    cap = cv2.VideoCapture(url)
    for _ in range(5):
        ok, f = cap.read()
    cap.release()
    return f


def locate(frame, square):
    """(symbol, conf, board_xy_mm, box, view) of the piece YOLO sees on `square`, or None."""
    view = E.view(frame)
    boxes = []
    D.detect_pieces(frame, view, conf=0.2, boxes_out=boxes)
    best = None
    for sym, conf, x1, y1, x2, y2, sq, h, note in boxes:
        if sq != square:
            continue
        if best is None or conf > best[1]:
            base = np.array([(x1 + x2) / 2, y2 - 0.12 * (y2 - y1)])
            xy = view.px_to_mm([base])[0]
            best = (sym, conf, xy, (x1, y1, x2, y2), view, base)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--square", default="c2")
    ap.add_argument("--url", default=str(C.EXT_CAMERA_URL), help="board camera (the camera window's /video stream)")
    ap.add_argument("--clearance", type=float, default=25.0, help="mm between the jaw tip and the piece's top")
    ap.add_argument("--height", type=float, help="tip height above the board in mm (overrides --clearance)")
    ap.add_argument("--open", type=float, default=0.0, help="gripper opening in %% while hovering (never above the taught maximum)")
    ap.add_argument("--go", action="store_true", help="move the real arm")
    a = ap.parse_args()

    sq = chess.parse_square(a.square)
    frame = grab(a.url)
    got = locate(frame, sq)
    if got is None:
        raise SystemExit(f"YOLO sees no piece on {a.square}")
    sym, conf, xy, box, view, base_px = got
    centre = np.asarray(square_center_mm(sq))
    spec = C.PIECES[sym.lower()]
    z = a.height if a.height is not None else spec["height"] + a.clearance
    print(f"1. YOLO: {chess.piece_name(chess.Piece.from_symbol(sym).piece_type)} ({sym}) on {a.square}, confidence {conf:.2f}")
    print(f"   box x {box[0]:.0f}-{box[2]:.0f}, y {box[1]:.0f}-{box[3]:.0f} px -> base point at pixel ({base_px[0]:.1f}, {base_px[1]:.1f})")
    print(f"2. board mm: x {xy[0]:.1f}, y {xy[1]:.1f}   (the centre of {a.square} is x {centre[0]:.1f}, y {centre[1]:.1f}; "
          f"the piece is off by dx {xy[0] - centre[0]:+.1f}, dy {xy[1] - centre[1]:+.1f} mm)")
    print(f"   hover height: {z:.0f} mm above the board ({spec['height']:.0f} mm piece + clearance)")
    T, ik = BoardToRobot.load(), IK()
    target = T(xy[0], xy[1], z)
    print(f"3. robot frame (m): x {target[0]:.4f}, y {target[1]:.4f}, z {target[2]:.4f}   (board->robot fit residual {T.residual_mm if hasattr(T, 'residual_mm') else 'n/a'} mm)")
    from chess_player.motion.arm import look_pose

    look = {**look_pose(), "gripper": C.GRIPPER_LOOK}
    q0 = np.array([look[j] for j in C.ARM_JOINTS])
    q0[4] = C.WRIST_ROLL_DEG
    hi = solve_or_raise(ik, T(xy[0], xy[1], C.Z_SAFE), q0, what="safe height")
    lo, err, tilt = ik.solve(target, hi, C.GRASP_MAX_TILT)
    pb = T.R.T @ (ik.fk(lo)[:3, 3] - T.t) * 1000
    print(f"4. IK joints (deg) {dict(zip(C.ARM_JOINTS, np.round(lo, 1)))}")
    print(f"   position error {err:.2f} mm, tool tilt from vertical {tilt:.1f} deg, predicted tip in board mm "
          f"({pb[0]:.1f}, {pb[1]:.1f}, {pb[2]:.1f}); safe height joints {np.round(hi, 1)}")
    if err > C.IK_MAX_ERR_MM:
        raise SystemExit(f"not reachable: error {err:.1f} mm")
    if not a.go:
        print("\n(dry run: nothing moved. Add --go to move the arm.)")
        return

    from chess_player.motion.arm import Arm

    opening = min(a.open, C.GRIP_OPEN_MAX_PCT)
    print(f"\nThe arm will go above {a.square} at {C.Z_SAFE:.0f} mm, then down to {z:.0f} mm and HOLD (gripper {opening:.1f} %).")
    input("Hands clear, a hand near the power switch. Enter = go, Ctrl+C = cancel: ")
    arm = Arm()
    try:
        arm.move_to({**dict(zip(C.ARM_JOINTS, hi)), "gripper": 0.0})
        arm.settle()
        arm.move_to({**dict(zip(C.ARM_JOINTS, lo)), "gripper": opening}, C.DESCEND_S)
        arm.settle()  # the shoulder stops 2-4 deg short at full stretch: measure and add it
        now = arm.present()
        got_tip = T.R.T @ (ik.fk(np.array([now[j] for j in C.ARM_JOINTS]))[:3, 3] - T.t) * 1000
        print(f"HOLDING. measured tip in board mm: x {got_tip[0]:.1f}, y {got_tip[1]:.1f}, z {got_tip[2]:.1f}   "
              f"(wanted x {xy[0]:.1f}, y {xy[1]:.1f}, z {z:.1f})")
        print("Compare it with the piece by eye now (and with your own top-down pose on the leader).")
        input("Enter = go back up and home: ")
        arm.move_to({**dict(zip(C.ARM_JOINTS, hi)), "gripper": 0.0}, C.DESCEND_S)
        arm.move_to(look, 2.0)
    finally:
        arm.close()


if __name__ == "__main__":
    main()
