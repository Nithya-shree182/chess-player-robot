"""Inverse kinematics for the SO-101 with the jaws pointing straight down.

Wraps placo directly rather than lerobot.model.kinematics.RobotKinematics so we can (a) fix wrist_roll,
(b) ask only for the approach axis to point down (5 DOF can't take a full 6-D pose), and (c) iterate to
convergence and report the residual.

Joint angles in and out are LeRobot calibrated degrees (use_degrees=True). LeRobot's 0 deg is the middle of
the range swept during calibration, while so101_new_calib.urdf's zero is the homing pose (raw 2047), so each
joint is shifted by (range_mid - 2047) * 360/4095 before going into the URDF (urdf_offsets_deg). Without this
the board-corner fit was 15 mm RMS; with it 3.9 mm (2026-10-17).
"""
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares

from chess_player import config as C


# ---- Minimal URDF kinematic chain (replaces placo, which has no Windows wheels). The SO-101 chain is serial:
# base_link -> shoulder_pan -> shoulder_lift -> elbow_flex -> wrist_flex -> wrist_roll -> gripper_frame_link.
def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


class _Chain:
    """Forward kinematics from the URDF: fixed origin transforms with a z-axis rotation per revolute joint."""

    def __init__(self, urdf_path, tip_link):
        joints = {}
        for j in ET.parse(str(urdf_path)).getroot().findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
            T = np.eye(4)
            T[:3, :3] = _rpy_to_R(*rpy)
            T[:3, 3] = xyz
            lim = j.find("limit")
            joints[j.find("child").get("link")] = dict(
                name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"), T=T,
                child=j.find("child").get("link"),
                axis=[float(v) for v in j.find("axis").get("xyz", "0 0 1").split()] if j.find("axis") is not None else None,
                limits=(float(lim.get("lower")), float(lim.get("upper"))) if lim is not None and lim.get("lower") else None)
        self.joints = joints  # by child link (also the branches that are not on the base -> tip chain, e.g. the moving jaw)
        chain, link = [], tip_link
        while link in joints:
            chain.append(joints[link])
            link = joints[link]["parent"]
        self.chain = chain[::-1]
        for j in self.chain:
            if j["type"] == "revolute" and not np.allclose(j["axis"], [0, 0, 1]):
                raise ValueError(f"joint {j['name']}: only z-axis revolute joints are supported")
        self.limits = {j["name"]: j["limits"] for j in self.chain if j["type"] == "revolute"}

    def fk(self, q_rad):
        """4x4 pose of the tip link; q_rad = {joint name: angle (rad)}."""
        T = np.eye(4)
        for j in self.chain:
            T = T @ j["T"]
            if j["type"] == "revolute":
                T = T @ _rotz(q_rad.get(j["name"], 0.0))
        return T

    def fk_to(self, q_rad, link):
        """Pose of an ancestor link of the tip (e.g. 'gripper_link') in the base frame."""
        T = np.eye(4)
        for j in self.chain:
            T = T @ j["T"]
            if j["type"] == "revolute":
                T = T @ _rotz(q_rad.get(j["name"], 0.0))
            if j["child"] == link:
                return T
        raise KeyError(link)

DOWN =np.array([0.0, 0.0, -1.0])
TOOL_AXIS = np.array([0.0, 0.0, 1.0])  # gripper_frame_link z points out of the jaws (checked with FK at zero)


CAL_PATH = C.Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower" / f"{C.FOLLOWER_ID}.json"


def urdf_offsets_deg():
    """URDF angle = LeRobot degrees + offset, per arm joint (0 for wrist_roll, whose range is the full turn)."""
    if not CAL_PATH.exists():
        return {j: 0.0 for j in C.ARM_JOINTS}
    cal = json.loads(CAL_PATH.read_text())
    out = {j: ((cal[j]["range_min"] + cal[j]["range_max"]) / 2 - 2047) * 360 / 4095 for j in C.ARM_JOINTS}
    out["wrist_roll"] -= C.WRIST_ROLL_ZERO_SHIFT_DEG  # its zero moved when the follower was matched to the leader
    return out


def calibrated_limits_deg():
    """Joint ranges recorded by LeRobot calibration, in LeRobot degrees, shrunk by JOINT_MARGIN_DEG."""
    if not CAL_PATH.exists():
        return {}
    cal = json.loads(CAL_PATH.read_text())
    out = {}
    for j in C.ARM_JOINTS:
        lo, hi = cal[j]["range_min"], cal[j]["range_max"]
        mid = (lo + hi) / 2
        out[j] = ((lo - mid) * 360 / 4095 + C.JOINT_MARGIN_DEG, (hi - mid) * 360 / 4095 - C.JOINT_MARGIN_DEG)
    return out


class IK:
    FREE = C.ARM_JOINTS[:4]  # pan, lift, elbow, flex are solved; wrist_roll is fixed (see wrist_roll_deg)

    def __init__(self, wrist_roll_deg=C.WRIST_ROLL_DEG):
        self.chain = _Chain(C.URDF, C.EE_FRAME)
        self.wrist_roll = wrist_roll_deg
        self.off = urdf_offsets_deg()
        # Intersect URDF limits with the real calibrated range so IK never asks for an unreachable angle.
        lim = {j: self.chain.limits[j] for j in self.FREE}
        for j, (lo, hi) in calibrated_limits_deg().items():
            if j in lim:
                lim[j] = (max(lim[j][0], self._rad(j, lo)), min(lim[j][1], self._rad(j, hi)))
        self.lo = np.array([lim[j][0] for j in self.FREE])
        self.hi = np.array([lim[j][1] for j in self.FREE])

    def _rad(self, j, lerobot_deg):
        return np.deg2rad(lerobot_deg + self.off[j])

    def _deg_of(self, j, rad):
        return float(np.rad2deg(rad)) - self.off[j]

    def _pose(self, x):
        q = {j: v for j, v in zip(self.FREE, x)}
        q["wrist_roll"] = self._rad("wrist_roll", self.wrist_roll)
        return self.chain.fk(q)

    def fk(self, q_deg):
        """Pose (4x4, meters) of the jaw tip for arm joint angles [pan, lift, elbow, flex, roll] in LeRobot degrees."""
        q = {j: self._rad(j, v) for j, v in zip(C.ARM_JOINTS, q_deg)}
        return self.chain.fk(q)

    def jaw_open_direction(self, q_deg):
        """Unit vector (robot frame, perpendicular to the tool axis) along which the moving fingertip travels when the
        gripper opens, for arm joint angles [pan, lift, elbow, flex, roll] in LeRobot degrees. The fixed finger is on
        the other side. (Replaces the placo-based computation of the original code.)"""
        q = {j: self._rad(j, v) for j, v in zip(C.ARM_JOINTS, q_deg)}
        Tj0 = self.chain.fk_to(q, "gripper_link") @ self.chain.joints["moving_jaw_so101_v1_link"]["T"]
        tip = self.chain.fk(q)
        on_jaw = np.linalg.inv(Tj0) @ np.r_[tip[:3, 3], 1.0]  # the fingertip point, attached to the moving jaw
        u = (Tj0 @ _rotz(0.6) @ on_jaw)[:3] - tip[:3, 3]  # where it moves when the jaw opens 0.6 rad
        u -= np.dot(u, tip[:3, 2]) * tip[:3, 2]
        return u / np.linalg.norm(u)

    def _solve_once(self, target_m, seed_deg, align_w, iters):
        target_m = np.asarray(target_m, float)
        x0 = np.array([self._rad(j, v) for j, v in zip(self.FREE, seed_deg[:4])])
        x0 = np.clip(x0, self.lo + 1e-6, self.hi - 1e-6)
        sw = np.sqrt(align_w)

        def resid(x):  # soft tasks: position (weight 1), tool axis pointing down (weight align_w), tiny regularisation
            T = self._pose(x)
            # position residual x10: millimetres matter more than a few degrees of tilt (checked on the old
            # calibration data: with weight 1 the round-trip error was up to 3 mm, too sloppy for placing pieces)
            return np.concatenate([10.0 * (T[:3, 3] - target_m), sw * (T[:3, :3] @ TOOL_AXIS - DOWN), 1e-3 * (x - x0)])

        sol = least_squares(resid, x0, bounds=(self.lo, self.hi), xtol=1e-12, ftol=1e-12, gtol=1e-12,
                            max_nfev=max(iters, 20) * 4)
        T = self._pose(sol.x)
        q = [self._deg_of(j, v) for j, v in zip(self.FREE, sol.x)] + [self.wrist_roll]
        err = float(np.linalg.norm(T[:3, 3] - target_m) * 1000)
        tilt = float(np.degrees(np.arccos(np.clip(T[:3, 2] @ DOWN, -1, 1))))
        return np.array(q), err, tilt

    def solve(self, target_m, seed_deg, max_tilt_deg=90.0, iters=80):
        """Joint angles (deg, 5 values) putting the jaw tip at target_m (robot frame, meters), pointing down.

        The wrist can't point straight down everywhere (wrist_flex saturates when the tip is high or far), so
        the down-alignment is relaxed step by step until position is met. Returns (q_deg, pos_err_mm,
        tilt_deg) of the most vertical solution within IK_MAX_ERR_MM, or the best-position one if none is.
        Seed with the previous solution to avoid elbow flips.
        """
        target_m = np.asarray(target_m, float)
        best = None
        for w in (0.3, 0.03, 3e-3, 3e-4, 0.0):
            q, err, tilt = self._solve_once(target_m, seed_deg, w, iters)
            if err <= C.IK_MAX_ERR_MM and tilt <= max_tilt_deg:
                return q, err, tilt
            if best is None or err < best[1]:
                best = (q, err, tilt)
        return best


class Unreachable(RuntimeError):
    pass


def solve_or_raise(ik, target_m, seed_deg, max_tilt_deg=90.0, what=""):
    q, err, tilt = ik.solve(target_m, seed_deg, max_tilt_deg)
    if err > C.IK_MAX_ERR_MM or tilt > max_tilt_deg:
        raise Unreachable(
            f"{what}: IK error {err:.1f} mm, tilt {tilt:.0f} deg at {np.round(np.asarray(target_m) * 1000)} mm"
        )
    return q
