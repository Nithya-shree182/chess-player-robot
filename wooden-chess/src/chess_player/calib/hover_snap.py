"""Hover the jaw tip over several squares and PHOTOGRAPH it with the overhead camera: no clicking or typing needed.

  python -m chess_player.calib.hover_snap [--squares a8 h8 ...] [--height 15]

The arm MOVES (the same hover as hover_test). At each square it holds still for ~2.5 s while the phone takes a
picture; the zoomed crop shows the square's true centre (green crosshair, placed where the tip would appear at that
height) and a millimetre grid in board axes, saved to data/hover_check/<square>.jpg. The offset of the tip from the
crosshair is then read off the picture (by eye or by a person) and fed to apply_estimates / nudge fitting.

Board axes in the pictures: x = along the files (a -> h), y = along the ranks (1 -> 8); labels on the grid say which.
"""
import argparse
import json
import time

import chess
import cv2
import numpy as np

from chess_player import config as C
from chess_player.calib.nudge_calibrate import DEFAULT_SQUARES, MIN_HEIGHT_MM
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.arm import Arm, FakeArm, look_pose
from chess_player.motion.ik import IK, solve_or_raise
from chess_player.vision import ext_camera as E

HALF_PX = 60  # half-size of the saved crop in camera pixels (~30 mm)
ZOOM = 10
OUT = C.DATA / "hover_check"


def annotate(frame, view, xy0, height, name):
    """Zoomed crop around the target with the crosshair and a 2 mm / 10 mm grid in board axes. Returns (image, info)."""
    plane = view.mm_to_px([xy0])[0]
    par = view.project([(xy0[0], xy0[1], height)])[0] - view.project([(xy0[0], xy0[1], 0.0)])[0]
    centre = plane + par  # where the tip appears when it is `height` above the true centre
    x0, y0 = centre[0] - HALF_PX, centre[1] - HALF_PX
    pad = 2 * HALF_PX
    big = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    crop = big[int(round(y0)) + pad : int(round(y0)) + pad + 2 * HALF_PX, int(round(x0)) + pad : int(round(x0)) + pad + 2 * HALF_PX]
    out = cv2.resize(crop, (2 * HALF_PX * ZOOM, 2 * HALF_PX * ZOOM), interpolation=cv2.INTER_CUBIC)
    ox, oy = int(round(x0)), int(round(y0))

    def to_out(p):
        return int(round((p[0] - ox) * ZOOM)), int(round((p[1] - oy) * ZOOM))

    # mm grid in board axes through the crosshair (shifted by the height parallax so it is the tip's height plane)
    for k in range(-20, 21, 2):
        major = k % 10 == 0
        col = (0, 255, 255) if major else (0, 160, 160)
        a = view.mm_to_px([(xy0[0] + k, xy0[1] - 20), (xy0[0] + k, xy0[1] + 20)]) + par  # constant x
        b = view.mm_to_px([(xy0[0] - 20, xy0[1] + k), (xy0[0] + 20, xy0[1] + k)]) + par  # constant y
        cv2.line(out, to_out(a[0]), to_out(a[1]), col, 1 if not major else 2, cv2.LINE_AA)
        cv2.line(out, to_out(b[0]), to_out(b[1]), col, 1 if not major else 2, cv2.LINE_AA)
        if major and k:
            cv2.putText(out, f"x{k:+d}", to_out(a[0] + (3, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, f"y{k:+d}", to_out(b[0] + (3, -4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    cx, cy = to_out(centre)
    cv2.line(out, (cx - 40, cy), (cx + 40, cy), (60, 230, 120), 3, cv2.LINE_AA)
    cv2.line(out, (cx, cy - 40), (cx, cy + 40), (60, 230, 120), 3, cv2.LINE_AA)
    cv2.putText(out, f"{name}  x=files(a->h) y=ranks(1->8)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    # camera pixels per mm along the board x and y axes, near the target (for converting a pixel offset to mm)
    px_per_mm_x = float(np.linalg.norm(view.mm_to_px([(xy0[0] + 5, xy0[1])])[0] - view.mm_to_px([(xy0[0] - 5, xy0[1])])[0]) / 10)
    px_per_mm_y = float(np.linalg.norm(view.mm_to_px([(xy0[0], xy0[1] + 5)])[0] - view.mm_to_px([(xy0[0], xy0[1] - 5)])[0]) / 10)
    return out, {"square": name, "centre_px": centre.tolist(), "parallax_px": par.tolist(),
                 "px_per_mm_x": px_per_mm_x, "px_per_mm_y": px_per_mm_y, "zoom": ZOOM}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squares", nargs="+", default=DEFAULT_SQUARES)
    ap.add_argument("--height", type=float, default=15.0)
    ap.add_argument("--hold", type=float, default=2.5, help="seconds at each square (photo at ~1.2 s)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.height < MIN_HEIGHT_MM:
        raise SystemExit(f"--height {a.height} is too low: use at least {MIN_HEIGHT_MM:.0f}.")
    OUT.mkdir(parents=True, exist_ok=True)
    T, ik = BoardToRobot.load(apply_correction=False), IK()
    cam = E.ExtCamera()
    time.sleep(1.0)
    view = E.view(cam.grab(max_age_s=5.0))
    arm = FakeArm() if a.dry_run else Arm()
    look = {**look_pose(), "gripper": C.GRIPPER_LOOK}
    meta = []
    try:
        q = np.array([look[j] for j in C.ARM_JOINTS])
        q[4] = C.WRIST_ROLL_DEG
        for name in a.squares:
            xy0 = square_center_mm(chess.parse_square(name))
            hi = solve_or_raise(ik, T(*xy0, C.Z_SAFE), q, what=f"{name} high")
            arm.move_to({**dict(zip(C.ARM_JOINTS, hi)), "gripper": 0.0})
            arm.settle()
            lo = solve_or_raise(ik, T(*xy0, a.height), hi, C.GRASP_MAX_TILT, what=f"{name} low")
            arm.move_to(dict(zip(C.ARM_JOINTS, lo)), C.DESCEND_S)
            arm.settle()
            time.sleep(0.8)
            frames = [cam.grab(max_age_s=3.0)]
            time.sleep(0.4)
            frames.append(cam.grab(max_age_s=3.0))
            img, info = annotate(frames[-1], view, xy0, a.height, name)
            cv2.imwrite(str(OUT / f"{name}.jpg"), img)
            cv2.imwrite(str(OUT / f"{name}_full.jpg"), frames[-1])
            now = arm.present()
            tip = T.R.T @ (ik.fk(np.array([now[j] for j in C.ARM_JOINTS]))[:3, 3] - T.t) * 1000
            info.update(model_tip_board_mm=[float(tip[0]), float(tip[1]), float(tip[2])], target_mm=[float(xy0[0]), float(xy0[1])])
            meta.append(info)
            print(f"{name}: photo saved (model says tip at x {tip[0]:.1f} y {tip[1]:.1f} z {tip[2]:.1f}; wanted {xy0[0]:.1f}, {xy0[1]:.1f}, {a.height:.0f})")
            time.sleep(max(0.0, a.hold - 1.2))
            up = solve_or_raise(ik, T(*xy0, C.Z_SAFE), lo, what=f"{name} lift")
            arm.move_to(dict(zip(C.ARM_JOINTS, up)), C.DESCEND_S)
            q = up
        (OUT / "meta.json").write_text(json.dumps({"height_mm": a.height, "squares": meta}, indent=2))
        print("done: pictures in", OUT)
    finally:
        try:
            arm.move_to(look, 3.0)
        except Exception as e:
            print("(could not return to the look pose:", e, ")")
        arm.close()
        cam.close()


if __name__ == "__main__":
    main()
