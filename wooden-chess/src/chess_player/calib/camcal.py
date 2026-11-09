"""Calibrate the ARM to the BOARD with the CAMERA: no hand guiding, no nudging by eye, about 4 minutes. The arm MOVES (empty board).

  python -m chess_player.calib.camcal                 # measure, fit, check on fresh squares, save (backs up the old files)
  python -m chess_player.calib.camcal --learn-bias    # ONCE per tool/camera height, with a calibration you trust (see below)
  python -m chess_player.calib.camcal --check-only    # touch down on 3 squares with the CURRENT calibration and report the error
  python -m chess_player.calib.camcal --dry-run       # same flow on a simulated arm (no pictures of a tip: only tests the plumbing)

Why this works. The camera grid (scripts/regrid.sh) turns a picture pixel on the board's surface into board mm, to about 1 mm. The arm's
forward kinematics turn its MEASURED joint angles into the tip position in the robot frame, with no calibration involved. If the tip
touches the board, both describe the same point:  board (x, y, 0)  <->  robot (X, Y, Z).  Eight such pairs fit the rigid board->robot
transform (calib/board_to_robot does this from hand-guided touches that are good to 10-15 mm; here the "hand" is the arm and the
"eye" is the camera). The surface height comes for free: it is the height where the tip stopped following the command (the same
ground-sense as calib/find_board_z).

Per point: the arm goes to a safe height over a square (placed with whatever calibration exists, even a poor or an old one: it only has
to be over the board), descends in small steps until the tip stops following the command, waits for the camera window to catch up, and
the tip is found in the picture as the LOWEST point of what changed against a reference picture taken with the arm at home. Because it
touches the surface, the pixel->mm map is exact (no height parallax from the low camera).

Tip bias. The lowest changed pixel is the front-bottom edge of the jaw, a few mm toward the camera from the jaw tip the kinematics
describe. That offset is a property of the tool and the camera angle, not of the setup: --learn-bias measures it once, against a
calibration the user trusts (the nudged one), and every later run subtracts it (data/camcal_bias.json).

Outputs: data/board_to_robot.json (new, method "camcal"), data/board_correction.json and data/board_residual.json are moved aside (they
corrected the OLD rigid fit and no longer apply), data/camcal.json (the numbers), pictures in data/camcal/ (tip found, per point).
"""
import argparse
import json
import shutil
import time

import chess
import cv2
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot, fit_rigid, square_center_mm
from chess_player.motion.arm import Arm, FakeArm, look_pose
from chess_player.motion.ik import IK, solve_or_raise

OUT_DIR = C.DATA / "camcal"
RESULT = C.DATA / "camcal.json"
BIAS_FILE = C.DATA / "camcal_bias.json"
FIT_SQUARES = ["c3", "f3", "d4", "e5", "c6", "f6", "b4", "g5"]  # ranks 3-6: clear of the robot base, spread over the board
CHECK_SQUARES = ["e3", "d6", "g4"]  # never used in the fit
BIAS_SQUARES = ["c3", "f3", "d4", "e5", "c6", "f6"]
LAG_MM = 1.2
SHOT_URL = "http://127.0.0.1:8771/shot"
PASS_MM, WARN_MM = 3.0, 5.0


# ------------------------------------------------------------------ camera
def grab(url=SHOT_URL, n=5):
    from chess_player.calib.auto_grid import grab_median

    return grab_median(url, n=n, gap=0.2)


def detect_tip(ref, cur, pred_px, roi=95, thr=38, min_px=120):
    """Tip pixel (x, y) = the lowest point of what changed between the arm-at-home picture and now, searched only around pred_px (where
    the old calibration says the tip is). Returns (px, n_changed, mask) or None. Rows with fewer than 4 changed pixels are specks, not the tip."""
    d = cv2.absdiff(cv2.GaussianBlur(ref, (5, 5), 0), cv2.GaussianBlur(cur, (5, 5), 0)).max(axis=2)
    m = (d > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    x0, y0 = int(pred_px[0]) - roi, int(pred_px[1]) - roi
    x1, y1 = int(pred_px[0]) + roi, int(pred_px[1]) + roi
    roi_m = np.zeros_like(m)
    roi_m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]
    if int(roi_m.sum()) < min_px:
        return None
    rows = roi_m.sum(axis=1)
    ys = np.where(rows >= 4)[0]
    if len(ys) == 0:
        return None
    yb = int(ys.max())
    xs = np.where(roi_m[max(yb - 4, 0):yb + 1].any(axis=0))[0]
    if len(xs) == 0 or xs.max() - xs.min() > 45:  # a wide bottom edge is a shadow or a hand, not a jaw tip
        return None
    return (float(np.median(xs)), float(yb)), int(roi_m.sum()), roi_m


def px_to_board(view, px):
    return np.asarray(view.px_to_mm([np.asarray(px, float)])[0], float)


def board_to_px(view, mm):
    return np.asarray(view.mm_to_px([tuple(mm)])[0], float)


# ------------------------------------------------------------------ arm
def fk_robot(ik, joints):
    """Tip position in the robot frame [m] from the MEASURED joints."""
    return ik.fk(np.array([joints[j] for j in C.ARM_JOINTS]))[:3, 3]


def touch_down(arm, ik, T, xy, q_prev, start=30.0, fine=12.0, floor=-30.0, log=print):
    """Slow descent over board point xy (mm, board frame of T) until the tip stops following. Returns (contact joints, q at contact)
    or (None, q). The arm ends at the contact height, pressed at most ~1 mm."""
    hi = solve_or_raise(ik, T(*xy, C.Z_SAFE), q_prev, what="high")
    arm.move_to({**dict(zip(C.ARM_JOINTS, hi)), "gripper": 0.0})
    arm.settle()
    z = start
    q = solve_or_raise(ik, T(*xy, z), hi, C.GRASP_MAX_TILT, what=f"z={z:.0f}")
    arm.move_to(dict(zip(C.ARM_JOINTS, q)), C.DESCEND_S)
    arm.settle()

    def tz(now):  # tip height in T's board frame from the measured joints
        return float((T.R.T @ (fk_robot(ik, now) - T.t))[2] * 1000)

    base = tz(arm.present()) - z
    hits = 0
    while z > floor:
        z -= 1.0 if z <= fine else 2.0
        q = solve_or_raise(ik, T(*xy, z), q, C.GRASP_MAX_TILT, what=f"z={z:.0f}")
        arm.move_to(dict(zip(C.ARM_JOINTS, q)), 0.5)
        time.sleep(0.3)
        lag = tz(arm.present()) - z - base
        if lag > LAG_MM:
            hits += 1
            if hits >= 2:
                z += 1.0  # back off 1 mm: resting on the surface, not pressing
                q = solve_or_raise(ik, T(*xy, z), q, C.GRASP_MAX_TILT, what="rest")
                arm.move_to(dict(zip(C.ARM_JOINTS, q)), 0.4)
                time.sleep(0.4)
                return arm.present(), q
        else:
            hits = 0
    return None, q


def lift_up(arm, ik, T, xy, q):
    up = solve_or_raise(ik, T(*xy, C.Z_SAFE), q, what="lift")
    arm.move_to(dict(zip(C.ARM_JOINTS, up)), C.DESCEND_S)
    return up


def home(arm):
    look = {**look_pose(), "gripper": C.GRIPPER_LOOK}
    arm.move_to(look, 2.5)
    return look


# ------------------------------------------------------------------ one measurement
def measure_points(arm, ik, T_prior, view, squares, url, log=print, tag="fit", sleep=2.5):
    """For each square: touch, find the tip in the picture. Returns a list of dicts (square, cmd_xy, board_px, board_mm, robot_m, ...)."""
    look = home(arm)
    time.sleep(4.0)
    ref = grab(url)
    q = np.array([look[j] for j in C.ARM_JOINTS])
    q[4] = C.WRIST_ROLL_DEG
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for name in squares:
        xy = square_center_mm(chess.parse_square(name))
        try:
            now, q = touch_down(arm, ik, T_prior, xy, q, log=log)
        except Exception as e:  # unreachable over there with the old placement
            log(f"  {name}: skipped ({e})")
            continue
        if now is None:
            log(f"  {name}: NO contact (the surface is lower than 30 mm below where the old calibration puts it): skipped")
            q = lift_up(arm, ik, T_prior, xy, q)
            continue
        p = fk_robot(ik, now)
        pred = board_to_px(view, (T_prior.R.T @ (p - T_prior.t))[:2] * 1000)
        time.sleep(sleep)  # the camera window averages still pixels: let it catch up with the arm
        cur = grab(url)
        det = detect_tip(ref, cur, pred)
        img = cur.copy()
        cv2.circle(img, tuple(int(v) for v in pred), 6, (255, 0, 0), 1)
        if det is None:
            log(f"  {name}: the tip was NOT found in the picture (see {OUT_DIR / f'{tag}_{name}.jpg'}): skipped")
            cv2.imwrite(str(OUT_DIR / f"{tag}_{name}.jpg"), img)
            q = lift_up(arm, ik, T_prior, xy, q)
            continue
        px, n, _m = det
        mm = px_to_board(view, px)
        cv2.circle(img, tuple(int(v) for v in px), 5, (0, 0, 255), -1)
        cv2.imwrite(str(OUT_DIR / f"{tag}_{name}.jpg"), img)
        log(f"  {name}: tip at pixel ({px[0]:.0f}, {px[1]:.0f}) = board ({mm[0]:.1f}, {mm[1]:.1f}) mm; robot ({p[0]*1000:.1f}, {p[1]*1000:.1f}, {p[2]*1000:.1f}) mm")
        out.append(dict(square=name, cmd_xy=[float(xy[0]), float(xy[1])], px=list(map(float, px)), board_mm=list(map(float, mm)),
                        robot_m=list(map(float, p))))
        q = lift_up(arm, ik, T_prior, xy, q)
    home(arm)
    return out


def load_bias():
    try:
        return np.array(json.loads(BIAS_FILE.read_text())["bias_mm"], float)
    except (OSError, ValueError, KeyError):
        return np.zeros(2)


# ------------------------------------------------------------------ fit
def fit_points(pts, bias):
    """Rigid board->robot fit from touch pairs (the tip bias removed). Returns (T, residuals mm)."""
    board = np.array([[p["board_mm"][0] - bias[0], p["board_mm"][1] - bias[1], 0.0] for p in pts])
    robot = np.array([p["robot_m"] for p in pts])
    return fit_rigid(board, robot)


def backup(path, stamp):
    if path.exists():
        dst = path.with_name(f"{path.stem}.before-camcal-{stamp}{path.suffix}")
        shutil.move(str(path), str(dst))
        return dst
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=SHOT_URL)
    ap.add_argument("--learn-bias", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--squares", nargs="+", default=None)
    a = ap.parse_args()

    from chess_player.vision.ext_camera import view as make_view
    view = make_view(grab(a.url, n=3))
    try:
        T0 = BoardToRobot.load()
    except (OSError, ValueError):
        T0 = BoardToRobot.assumed(robot_side="white")
        print("no calibration file: starting from an assumed placement (board 110 mm in front of the robot)")
    ik = IK()
    arm = FakeArm() if a.dry_run else Arm()
    stamp = time.strftime("%m%d-%H%M")
    try:
        if a.learn_bias:
            print("== Learning the tip bias against the CURRENT calibration (it must be a good one, e.g. nudged by eye) ==")
            pts = measure_points(arm, ik, T0, view, a.squares or BIAS_SQUARES, a.url, tag="bias")
            if len(pts) < 4:
                raise SystemExit("fewer than 4 usable points: bias not saved (look at data/camcal/bias_*.jpg)")
            d = np.array([np.array(p["board_mm"]) - np.array(p["cmd_xy"]) for p in pts])
            bias, sd = d.mean(0), d.std(0)
            print(f"tip bias (seen minus commanded, board mm): x {bias[0]:+.2f}, y {bias[1]:+.2f}; scatter x {sd[0]:.2f}, y {sd[1]:.2f}  ({len(pts)} points)")
            if max(sd) > 3.0:
                print("scatter above 3 mm: the current calibration is not good enough to learn from, or the tip detection is off (see pictures). NOT saved.")
                return
            if not a.dry_run:
                BIAS_FILE.write_text(json.dumps({"bias_mm": [float(bias[0]), float(bias[1])], "scatter_mm": [float(sd[0]), float(sd[1])],
                                                 "points": pts, "time": stamp}, indent=2))
                print("saved", BIAS_FILE)
            return

        bias = load_bias()
        if not BIAS_FILE.exists():
            print("NOTE: no tip bias learned yet (--learn-bias): the result may be off by a constant few mm toward the camera.")
        if a.check_only:
            pts = measure_points(arm, ik, T0, view, a.squares or CHECK_SQUARES, a.url, tag="check")
            report_check(pts, bias)
            return

        print("== Camera calibration: the arm touches the board at 8 squares; the camera finds the tip each time ==")
        pts = measure_points(arm, ik, T0, view, a.squares or FIT_SQUARES, a.url, tag="fit")
        if len(pts) < 5:
            raise SystemExit(f"only {len(pts)} usable points (need 5): nothing changed. Look at data/camcal/fit_*.jpg")
        T, res = fit_points(pts, bias)
        # drop one outlier at most (a mis-found tip) and re-fit
        if len(pts) > 5 and res.max() > 2.5 * max(np.median(res), 0.8):
            k = int(res.argmax())
            print(f"  dropping {pts[k]['square']} (residual {res[k]:.1f} mm, the others {np.round(np.delete(res, k), 1)})")
            pts = [p for i, p in enumerate(pts) if i != k]
            T, res = fit_points(pts, bias)
        tilt = np.degrees(np.arccos(np.clip(T.R[2, 2], -1, 1)))
        print(f"\nfit residuals (mm): {np.round(res, 1)}  RMS {T.residual_mm:.2f}; board surface {T.t[2]*1000:.1f} mm above the base frame, tilt {tilt:.1f} deg")
        if T.residual_mm > 3.0:
            print("WARNING: RMS above 3 mm: not saved. Look at data/camcal/fit_*.jpg: is the tip marked (red dot) at the jaw tip? Is the grid current (regrid.sh)?")
            return
        if a.dry_run:
            print("dry run: nothing saved")
            return
        moved = [backup(p, stamp) for p in (C.BOARD_TO_ROBOT, C.BOARD_CORRECTION, C.DATA / "board_residual.json")]
        T.save(method="camcal", points=pts, tip_bias_mm=bias.tolist(), time=stamp)
        RESULT.write_text(json.dumps({"time": stamp, "rms_mm": T.residual_mm, "residuals_mm": res.tolist(), "bias_mm": bias.tolist(),
                                      "n": len(pts)}, indent=2))
        print("saved", C.BOARD_TO_ROBOT, "(old files moved aside:", ", ".join(m.name for m in moved if m) or "none", ")")

        print("\n== Check on squares that were NOT used in the fit ==")
        T1 = BoardToRobot.load()
        pts = measure_points(arm, ik, T1, view, CHECK_SQUARES, a.url, tag="check")
        report_check(pts, bias)
    finally:
        arm.close()


def report_check(pts, bias):
    """Error of the tip against the commanded square centre, from the picture (tip bias removed)."""
    if not pts:
        print("no usable check points")
        return 1
    e = np.array([np.array(p["board_mm"]) - bias - np.array(p["cmd_xy"]) for p in pts])
    n = np.linalg.norm(e, axis=1)
    for p, v, m in zip(pts, e, n):
        print(f"  {p['square']}: tip {v[0]:+.1f}, {v[1]:+.1f} mm from the square centre ({m:.1f} mm)")
    worst = float(n.max())
    verdict = "PASS" if worst <= PASS_MM else ("WARN (usable, re-run camcal if grasps miss)" if worst <= WARN_MM else "FAIL (re-run camcal)")
    print(f"check: mean {n.mean():.1f} mm, worst {worst:.1f} mm -> {verdict}")
    return 0 if worst <= WARN_MM else 1


if __name__ == "__main__":
    main()
