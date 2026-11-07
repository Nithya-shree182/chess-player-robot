"""Marker-based hand-eye calibration: find the board->robot transform from the arm's own kinematics and the board camera.

A small bright dot (green / pink / blue sticker) is stuck on the gripper's FIXED part (not the moving jaw), on the face that
looks toward the camera. The arm visits poses over the EMPTY board; at each one the measured joints give the marker's 3-D
position through forward kinematics (up to the marker's unknown offset on the gripper), and the camera gives its pixel.
A nonlinear least-squares fit then finds the camera's pose in the robot frame (and the marker offset); together with the
camera's pose relative to the board (from the grid) that is board->robot, with no hand-touching.

  python -m chess_player.calib.handeye_marker selftest            # the maths on synthetic data, no arm, no marker
  python -m chess_player.calib.handeye_marker detect --color green # find the marker in the live picture (arm not moved)
  python -m chess_player.calib.handeye_marker collect             # dry run: the poses it would visit (nothing moves)
  python -m chess_player.calib.handeye_marker collect --go        # the arm moves; saves data/handeye_marker/samples.json
  python -m chess_player.calib.handeye_marker fit                 # solve, report; --apply writes data/board_to_robot.json
                                                                  # (old one backed up) and retires board_correction.json
Needs the camera window (http://127.0.0.1:8771/shot) and the grid fitted for the current camera position.
"""
import argparse
import json
import shutil
import time
import urllib.request

import cv2
import numpy as np
from scipy.optimize import least_squares

from chess_player import config as C
from chess_player.geometry import BoardToRobot
from chess_player.motion.ik import IK
from chess_player.vision import ext_camera as E

OUT = C.DATA / "handeye_marker"
SHOT_URL = "http://127.0.0.1:8771/shot"
COLORS = {  # HSV ranges (OpenCV: H 0-179). The arm is orange, the board black/grey/beige, so these are rare in the scene.
    "green": ((40, 90, 60), (85, 255, 255)),
    "pink": ((145, 70, 80), (178, 255, 255)),
    "blue": ((95, 110, 50), (130, 255, 255)),
}


def shot(url=SHOT_URL):
    return cv2.imdecode(np.frombuffer(urllib.request.urlopen(url, timeout=3).read(), np.uint8), cv2.IMREAD_COLOR)


def find_marker(frame, color="green"):
    """(cx, cy, area) of the marker blob, or None. Needs one clearly largest blob of the colour."""
    lo, hi = COLORS[color]
    mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = sorted(((cv2.contourArea(c), c) for c in cnts), key=lambda b: -b[0])
    blobs = [b for b in blobs if 12 <= b[0] <= 4000]
    if not blobs or (len(blobs) > 1 and blobs[1][0] > 0.4 * blobs[0][0]):
        return None
    m = cv2.moments(blobs[0][1])
    return m["m10"] / m["m00"], m["m01"] / m["m00"], blobs[0][0]


# ---------------------------------------------------------------- the maths
def marker_robot_points(ik, joints_deg, offset_tool):
    """Marker position (robot frame, m) for each row of joints [pan, lift, elbow, flex, roll], given its offset (m) in the
    gripper_link frame."""
    pts = []
    for q in joints_deg:
        qd = {j: ik._rad(j, v) for j, v in zip(C.ARM_JOINTS, q)}
        Tg = ik.chain.fk_to(qd, "gripper_link")
        pts.append((Tg @ np.append(offset_tool, 1.0))[:3])
    return np.array(pts)


def tool_offset_of_tip(ik, q):
    """The jaw tip expressed in the gripper_link frame (a sensible starting guess for the marker offset)."""
    qd = {j: ik._rad(j, v) for j, v in zip(C.ARM_JOINTS, q)}
    Tg, Tt = ik.chain.fk_to(qd, "gripper_link"), ik.fk(q)
    return (np.linalg.inv(Tg) @ np.append(Tt[:3, 3], 1.0))[:3]


def project(K, rvec, tvec, pts_robot):
    px, _ = cv2.projectPoints(np.asarray(pts_robot, np.float64).reshape(-1, 1, 3), rvec, tvec, K, None)
    return px.reshape(-1, 2)


def initial_camera_pose(view, T):
    """Camera pose in the robot frame (rvec, tvec in m) implied by the grid pose and the CURRENT board->robot transform."""
    Rb, _ = cv2.Rodrigues(view.rvec)
    tb = view.tvec.reshape(3) / 1000.0
    R0 = Rb @ T.R.T
    t0 = tb - R0 @ T.t
    return cv2.Rodrigues(R0)[0].reshape(3), t0


def fit_handeye(ik, joints, pixels, K, rvec0, tvec0, off0, fit_focal=False):
    """Least squares over (camera pose in robot frame, marker offset in the gripper frame[, focal]). Returns (result dict)."""
    joints, pixels = np.asarray(joints, float), np.asarray(pixels, float)

    def unpack(x):
        Kx = K.copy()
        if fit_focal:
            Kx[0, 0] = Kx[1, 1] = x[9]
        return x[:3], x[3:6], x[6:9], Kx

    def resid(x):
        rv, tv, off, Kx = unpack(x)
        return (project(Kx, rv, tv, marker_robot_points(ik, joints, off)) - pixels).ravel()

    x0 = np.r_[rvec0, tvec0, off0] if not fit_focal else np.r_[rvec0, tvec0, off0, K[0, 0]]
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=3.0, x_scale=np.r_[np.full(3, 0.05), np.full(3, 0.05), np.full(3, 0.02), [100.0] * fit_focal])
    rv, tv, off, Kx = unpack(sol.x)
    err = np.linalg.norm(resid(sol.x).reshape(-1, 2), axis=1)
    return {"rvec": rv, "tvec": tv, "offset": off, "K": Kx, "px_err": err, "rms_px": float(np.sqrt(np.mean(err ** 2)))}


def board_to_robot_from(view, rvec, tvec):
    """BoardToRobot from the grid pose (board mm -> camera) and the camera pose in the robot frame (robot m -> camera m)."""
    Rb, _ = cv2.Rodrigues(view.rvec)
    tb = view.tvec.reshape(3) / 1000.0
    Rr, _ = cv2.Rodrigues(np.asarray(rvec, float))
    R = Rr.T @ Rb  # board mm -> robot m has an extra /1000 handled in BoardToRobot.__call__
    t = Rr.T @ (tb - np.asarray(tvec, float))
    return R, t


# ---------------------------------------------------------------- poses to visit
def pose_plan(T, ik, spots=(40, 100, 160), heights=(45, 80, 115)):
    from chess_player.motion.arm import look_pose
    from chess_player.motion.ik import solve_or_raise

    look = look_pose()
    q = np.array([look[j] for j in C.ARM_JOINTS])
    q[4] = C.WRIST_ROLL_DEG
    plan, skipped = [], []
    for ix, x in enumerate(spots):
        for y in (spots if ix % 2 == 0 else spots[::-1]):  # snake, to keep the travel short
            try:
                hi = solve_or_raise(ik, T(x, y, C.Z_SAFE), q, what="safe")
                levels = []
                for z in heights:
                    levels.append((z, solve_or_raise(ik, T(x, y, z), hi, C.GRASP_MAX_TILT, what=f"{x},{y},{z}")))
                plan.append(((x, y), hi, levels))
                q = hi
            except Exception as e:  # unreachable spot
                skipped.append(((x, y), str(e)[:60]))
    return plan, skipped


def collect(args):
    T, ik = BoardToRobot.load(), IK()
    plan, skipped = pose_plan(T, ik)
    n = sum(len(p[2]) for p in plan)
    print(f"{len(plan)} spots x up to 3 heights = {n} poses" + (f"; skipped (unreachable): {skipped}" if skipped else ""))
    for (x, y), _hi, levels in plan:
        print(f"  board ({x:3d},{y:3d}) mm at heights {[z for z, _ in levels]}")
    if not args.go:
        print("\n(dry run: nothing moved. Add --go: board EMPTY, hands off the arm, a hand near the power switch.)")
        return
    from chess_player.motion.arm import Arm, look_pose
    from chess_player.teach.record_poses import FOLLOWER_CAL, Cal, make_bus, read_raw

    from chess_player.vision import detect_board as D

    f0 = shot()
    npieces = len(D.detect_pieces(f0, E.view(f0), conf=0.15))
    bus = make_bus(C.FOLLOWER_PORT, FOLLOWER_CAL)
    cal = Cal(FOLLOWER_CAL)
    a = cal.units(read_raw(bus))
    time.sleep(1.5)
    b = cal.units(read_raw(bus))
    bus.port_handler.closePort()
    if npieces or np.abs(a - b).max() > 0.5:
        raise SystemExit(f"NOT starting: {npieces} pieces on the board / the arm is being moved ({np.abs(a - b).max():.1f} deg)")
    m0 = find_marker(f0, args.color)
    print(f"marker {args.color} in the parked view: {'found at (%.0f, %.0f)' % m0[:2] if m0 else 'not visible (fine if the arm is parked high)'}")
    OUT.mkdir(parents=True, exist_ok=True)
    samples, arm = [], Arm()
    look = {**look_pose(), "gripper": C.GRIPPER_LOOK}
    try:
        for (x, y), hi, levels in plan:
            arm.move_to({**dict(zip(C.ARM_JOINTS, hi)), "gripper": 0.0})
            arm.settle()
            for z, lo in levels:
                arm.move_to({**dict(zip(C.ARM_JOINTS, lo)), "gripper": 0.0}, C.DESCEND_S)
                arm.settle()
                time.sleep(0.8)
                now = arm.present()
                q = [float(now[j]) for j in C.ARM_JOINTS]
                frame = shot()
                mk = find_marker(frame, args.color)
                tag = f"x{x}_y{y}_z{z}"
                cv2.imwrite(str(OUT / f"{tag}.jpg"), frame)
                print(f"  {tag}: marker " + ("at (%.1f, %.1f) area %d" % mk if mk else "NOT FOUND"), flush=True)
                if mk:
                    samples.append({"tag": tag, "joints": q, "pixel": [mk[0], mk[1]], "area": mk[2]})
            arm.move_to({**dict(zip(C.ARM_JOINTS, hi)), "gripper": 0.0}, C.DESCEND_S)
        arm.move_to(look, 2.0)
    finally:
        arm.close()
    (OUT / "samples.json").write_text(json.dumps(samples, indent=1))
    print(f"\nsaved {len(samples)} samples to {OUT / 'samples.json'}; next: fit")


def fit_cmd(args):
    samples = json.loads((OUT / "samples.json").read_text())
    if len(samples) < 8:
        raise SystemExit(f"only {len(samples)} samples: need at least 8")
    f = cv2.imread(sorted(OUT.glob("*.jpg"))[0].as_posix())
    view, T, ik = E.view(f), BoardToRobot.load(apply_correction=False), IK()
    K = view.K.copy()
    rv0, tv0 = initial_camera_pose(view, T)
    q0 = np.array(samples[0]["joints"])
    res = fit_handeye(ik, [s["joints"] for s in samples], [s["pixel"] for s in samples], K, rv0, tv0,
                      tool_offset_of_tip(ik, q0), fit_focal=args.focal)
    print(f"{len(samples)} samples; reprojection error: RMS {res['rms_px']:.2f} px, worst {res['px_err'].max():.2f} px "
          f"(about {res['rms_px'] / 3.3:.1f} mm on the board)")
    print(f"marker offset on the gripper (mm): {np.round(res['offset'] * 1000, 1).tolist()}; focal {res['K'][0, 0]:.0f} px")
    worst = np.argsort(-res["px_err"])[:3]
    print("worst samples:", [(samples[i]["tag"], round(float(res["px_err"][i]), 1)) for i in worst])
    R, t = board_to_robot_from(view, res["rvec"], res["tvec"])
    new = BoardToRobot(R, t)
    old = BoardToRobot.load()
    print("\nwhere the new transform sends board points, compared with the current one (+ the shift file), in mm:")
    for name, (x, y) in {"a1": (12.5, 12.5), "h1": (187.5, 12.5), "h8": (187.5, 187.5), "a8": (12.5, 187.5), "d4": (87.5, 87.5)}.items():
        d = (new(x, y, 40.0) - old(x, y, 40.0)) * 1000
        print(f"  {name}: differs by ({d[0]:+.1f}, {d[1]:+.1f}, {d[2]:+.1f}) mm in the robot frame")
    if args.apply:
        shutil.copy(C.BOARD_TO_ROBOT, C.BOARD_TO_ROBOT.with_suffix(".before-handeye.json"))
        new.save(residual_mm=round(res["rms_px"] / 3.3, 2), method="marker hand-eye 2026-11-06", samples=len(samples))
        if C.BOARD_CORRECTION.exists():
            shutil.move(C.BOARD_CORRECTION, C.BOARD_CORRECTION.with_suffix(".retired.json"))
        print("\nAPPLIED: data/board_to_robot.json replaced (old one: board_to_robot.before-handeye.json); board_correction.json retired.")


def detect_cmd(args):
    f = shot()
    m = find_marker(f, args.color)
    print(f"marker {args.color}: " + ("found at pixel (%.1f, %.1f), area %d px" % m if m else "NOT found (none, or several similar blobs)"))
    g = f.copy()
    if m:
        cv2.circle(g, (int(m[0]), int(m[1])), 18, (0, 255, 255), 2)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "detect.jpg"), g)
    print("picture:", OUT / "detect.jpg")


def selftest():
    """Synthetic: a known board->robot transform and camera, 27 poses, 1 px noise; the fit must recover the transform."""
    rng = np.random.default_rng(1)
    T_true = BoardToRobot.load(apply_correction=False)
    ik = IK()
    plan, _ = pose_plan(T_true, ik)
    joints = [lo for _xy, _hi, levels in plan for _z, lo in levels]
    off_true = tool_offset_of_tip(ik, joints[0]) + np.array([0.006, -0.004, 0.012])
    f = cv2.imread(sorted(C.DATA.glob("ext_ref_frame.jpg"))[0].as_posix())
    view = E.view(f)
    K = view.K.copy()
    rv_t, tv_t = initial_camera_pose(view, T_true)  # a consistent "truth" camera
    # the "current" (wrong) transform the fit starts from: shifted 5 cm and 2 degrees
    wrong = BoardToRobot(T_true.R, T_true.t + np.array([0.04, -0.05, 0.0]))
    rv0, tv0 = initial_camera_pose(view, wrong)
    pix = project(K, rv_t, tv_t, marker_robot_points(ik, joints, off_true)) + rng.normal(0, 1.0, (len(joints), 2))
    res = fit_handeye(ik, joints, pix, K, rv0, tv0, tool_offset_of_tip(ik, joints[0]))
    R, t = board_to_robot_from(view, res["rvec"], res["tvec"])
    got = BoardToRobot(R, t)
    errs = [np.linalg.norm((got(x, y, 40.0) - T_true(x, y, 40.0)) * 1000) for x, y in [(12.5, 12.5), (187.5, 12.5), (187.5, 187.5), (12.5, 187.5), (87.5, 87.5)]]
    start = [np.linalg.norm((wrong(x, y, 40.0) - T_true(x, y, 40.0)) * 1000) for x, y in [(12.5, 12.5), (87.5, 87.5)]]
    print(f"{len(joints)} synthetic poses, 1 px noise: starting error {np.round(start, 1).tolist()} mm -> recovered error per square "
          f"{np.round(errs, 2).tolist()} mm; reprojection RMS {res['rms_px']:.2f} px")
    assert max(errs) < 3.0, "the fit does not recover the transform"
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "detect", "collect", "fit"])
    ap.add_argument("--color", default="green", choices=list(COLORS))
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--focal", action="store_true", help="also fit the focal length")
    a = ap.parse_args()
    {"selftest": lambda: selftest(), "detect": lambda: detect_cmd(a), "collect": lambda: collect(a), "fit": lambda: fit_cmd(a)}[a.cmd]()


if __name__ == "__main__":
    main()
