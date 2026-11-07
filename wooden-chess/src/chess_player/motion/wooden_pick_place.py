"""The wooden-rig grasp recipe (verified on the pawn, e7 <-> e5, 2026-11-07) as a drop-in replacement for PickPlace.

Same interface as motion.pick_place.PickPlace (plan_transfer / execute / check / play / go_look), so the session, the game loop, the
UI and the move planner (captures, castling, en passant, promotion) work unchanged. One transfer = one piece carried from a source point
to a target point (a square centre, a graveyard slot or a spare):

  1. over the pick point at a safe travel height (the tip = the fixed finger's inner face sits `shift_x` toward the a-file of the piece
     centre), jaws open; if a neighbour stands on the h-file side (where the moving finger opens) the jaws open only a little more than
     the piece is wide
  2. straight down to the grip height, SLOW close until the pads touch (contact), light squeeze. No contact = nothing in the hand
  3. straight up, straight carry (high enough to clear every piece near the path), straight down
  4. the held piece sits against the fixed finger, half the contact gap from the tip, so the tip goes to centre - gap/2 + place_dx;
     SLOW release, straight up

The numbers per piece type are in data/wooden_grasps.json (see calib/pick_place.py --save); defaults = the verified pawn.
"""
import json

import numpy as np

from chess_player import config as C
from chess_player.motion.arm import FakeArm, look_pose
from chess_player.motion.ik import Unreachable
from chess_player.motion.pick_place import Blocked, GraspFailed, PickPlace, PieceDropped, Transfer, surface_z
from chess_player.motion.slow_grip import slow_close, slow_release

FILE = C.DATA / "wooden_grasps.json"
PAWN = dict(open_pct=16.1, shift_x=-10.6, shift_y=2.7, grip_tip=15.0, lift=30.0, place_dx=3.0, place_dy=0.0, place_extra=2.0,
            contact_pct=8.3)
PATH_STEP_MM = 6.0
CARRY_MM_PER_S = 25.0
SLIP_PCT = 2.5  # the jaws closing this far beyond the contact reading while carrying = the piece slipped out
NEIGHBOUR_BAND = (14.0, 44.0)  # h-file side: a piece this far along +x of the source (and within +-16 mm in y) is in the moving finger's way
CLEAR_ABOVE_MM = 8.0  # the carried piece's bottom stays this far above the tallest piece near its path


def recipe(piece):
    """Numbers for one piece type: the pawn's, a guessed fixed-finger distance for the others, then whatever was saved."""
    piece = piece.lower()
    d = dict(PAWN)
    if piece != "p":
        d["shift_x"] = -C.PIECES[piece]["diameter"] / 2.0
    try:
        d.update(json.loads(FILE.read_text()).get(piece, {}))
    except (OSError, ValueError):
        pass
    return d


def _seg_dist(c, a, b):
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    ab = b - a
    n = float(ab @ ab)
    f = 0.0 if n < 1e-9 else float(np.clip((c - a) @ ab / n, 0.0, 1.0))
    return float(np.linalg.norm(c - (a + ab * f)))


class WoodenPickPlace(PickPlace):
    # ---------------------------------------------------------------- planning (a dry run of the same code on a simulated arm)
    def plan_transfer(self, t: Transfer, obstacles=None):
        info = self._run(FakeArm(), t, obstacles, real=False)
        return {"info": info}

    # ---------------------------------------------------------------- execution
    def execute(self, t: Transfer, retries=2, obstacles=None):
        return self._run(self.arm, t, obstacles, real=True, retries=retries)

    def _run(self, arm, t: Transfer, obstacles, real, retries=2):
        sym = t.piece.lower()
        r = recipe(sym)
        H = C.PIECES[sym]["height"]
        obstacles = obstacles or {}
        src, dst = np.array(t.src, float), np.array(t.dst, float)
        zs, zd = surface_z(tuple(src)), surface_z(tuple(dst))
        T, ik, log = self.T, self.ik, self.log
        look = {**look_pose(), "gripper": C.GRIPPER_LOOK}
        seed = np.array([look[j] for j in C.ARM_JOINTS], float)
        seed[4] = C.WRIST_ROLL_DEG
        stat = dict(tilt=0.0, secs=0.0)

        def ik_at(xy, z, what):
            nonlocal seed
            q, err, tilt = ik.solve(T(xy[0], xy[1], z), seed, C.GRASP_MAX_TILT)
            if err > C.IK_MAX_ERR_MM or tilt > C.GRASP_MAX_TILT:
                raise Unreachable(f"{t.label}: {what}: IK error {err:.1f} mm, tilt {tilt:.0f} deg at ({xy[0]:.0f}, {xy[1]:.0f}, {z:.0f}) mm")
            seed = q
            stat["tilt"] = max(stat["tilt"], float(tilt))
            return q

        def reachable(xy, z):
            try:
                q, err, tilt = ik.solve(T(xy[0], xy[1], z), seed, C.GRASP_MAX_TILT)
            except Exception:
                return False
            return err <= C.IK_MAX_ERR_MM and tilt <= C.GRASP_MAX_TILT

        def pose(q, grip):
            return {**dict(zip(C.ARM_JOINTS, q)), "gripper": float(grip)}

        def line(xy0, z0, xy1, z1, what, grip=None):
            """Straight Cartesian line (IK every PATH_STEP_MM) from the arm's present place."""
            g = float(arm.joints()["gripper"]) if grip is None else grip
            xy0, xy1 = np.asarray(xy0, float), np.asarray(xy1, float)
            L = float(np.linalg.norm(np.r_[xy1 - xy0, z1 - z0]))
            n = max(1, int(np.ceil(L / PATH_STEP_MM)))
            wps = [arm.joints()]
            for i in range(1, n + 1):
                f = i / n
                wps.append(pose(ik_at(xy0 + (xy1 - xy0) * f, z0 + (z1 - z0) * f, what), g))
            dur = max(0.8, L / CARRY_MM_PER_S)
            stat["secs"] += dur + 0.4
            why = arm.move_path(wps, dur, check=holding_check if holding else None)
            if why:
                return why
            arm.settle()

        def safe_z(xy, z_min):
            for z in np.arange(C.Z_SAFE, z_min - 1.0, -5.0):
                if reachable(xy, float(z)):
                    return float(z)
            raise Unreachable(f"{t.label}: ({xy[0]:.0f}, {xy[1]:.0f}) cannot be reached above the piece")

        holding = False
        contact_pct = [r["contact_pct"]]

        def holding_check():
            if not real:
                return None
            if float(arm.present()["gripper"]) < contact_pct[0] - SLIP_PCT:
                return "the jaws closed further: the piece slipped out"
            return None

        def crowded_h_side(c):
            lo, hi = NEIGHBOUR_BAND
            return any(abs(xy[1] - c[1]) < 16 and lo < xy[0] - c[0] < hi for xy in obstacles)

        open_pct = min(r["open_pct"], C.GRIP_OPEN_MAX_PCT)
        open_pick = min(open_pct, r["contact_pct"] + 2.5) if crowded_h_side(src) else open_pct
        pick_xy = src + np.array([r["shift_x"], r["shift_y"]])
        z_above = H + 25.0 + zs
        z_grip = r["grip_tip"] + zs
        z_hi = safe_z(pick_xy, z_above)
        gap_est = float(np.interp(r["contact_pct"], C._GAP_PCT, C._GAP_MM))

        # travel height: lift_mm above the grip, or higher to clear the tallest piece within reach of the carry path
        place_est = dst + np.array([-gap_est / 2.0 + r["place_dx"], r["shift_y"] + r["place_dy"]])
        tall = 0.0
        for xy, s in obstacles.items():
            if _seg_dist(xy, src, dst) < 31.0:
                tall = max(tall, C.PIECES[s.lower()]["height"])
        z_c_want = max(z_grip + r["lift"], (tall + CLEAR_ABOVE_MM + r["grip_tip"] + zs) if tall else 0.0)
        z_c = None
        for z in np.arange(z_c_want, z_grip + r["lift"] - 1.0, -5.0):
            if all(reachable(p, float(z)) for p in (pick_xy, place_est, (pick_xy + place_est) / 2)):
                z_c = float(z)
                break
        if z_c is None:
            for z in np.arange(z_grip + r["lift"] - 5.0, z_grip + 4.0, -5.0):  # lower than wanted: only if nothing tall is near
                if not tall and all(reachable(p, float(z)) for p in (pick_xy, place_est, (pick_xy + place_est) / 2)):
                    z_c = float(z)
                    break
        if z_c is None:
            raise Blocked(f"{t.label}: cannot carry it high enough ({z_c_want:.0f} mm) to clear the pieces near the path")
        if tall and z_c < z_c_want - 0.1:
            raise Blocked(f"{t.label}: the carry path needs {z_c_want:.0f} mm to clear a {tall:.0f} mm piece, the arm reaches {z_c:.0f} mm")

        log(f"  transfer: {t.label}: pick tip ({pick_xy[0]:.0f}, {pick_xy[1]:.0f}), grip height {z_grip:.0f} mm, jaws {open_pick:.1f} %"
            + (" (crowded on the h-file side)" if open_pick < open_pct else "") + f", carry at {z_c:.0f} mm")

        # ---- pick
        for attempt in range(retries + 1):
            arm.move_to(pose(ik_at(pick_xy, z_hi, "pick, travel height"), open_pick))
            stat["secs"] += 2.0
            arm.settle()
            line(pick_xy, z_hi, pick_xy, z_above, "pick, above the piece", open_pick)
            line(pick_xy, z_above, pick_xy, z_grip, "pick, grip height", open_pick)
            if real:
                contact, fin = slow_close(arm)
            else:
                contact, fin = r["contact_pct"], r["contact_pct"] - 0.2
                stat["secs"] += 6.0
            if contact is not None:
                break
            log(f"  attempt {attempt + 1}: the jaws closed on nothing ({fin:.1f} %)")
            slow_release(arm, open_pick)
            line(pick_xy, z_grip, pick_xy, z_hi, "retreat", open_pick)
        else:
            raise GraspFailed(f"{t.label}: nothing in the hand after {retries + 1} tries")
        contact_pct[0] = float(contact)
        gap = float(np.interp(contact, C._GAP_PCT, C._GAP_MM))
        log(f"  gripped: contact at {contact:.1f} % = {gap:.1f} mm")

        # ---- carry
        place_xy = dst + np.array([-gap / 2.0 + r["place_dx"], r["shift_y"] + r["place_dy"]])
        z_place = r["grip_tip"] + r["place_extra"] + zd
        holding = True
        try:
            for step in (
                lambda: line(pick_xy, z_grip, pick_xy, z_c, "lift"),
                lambda: line(pick_xy, z_c, place_xy, z_c, "carry"),
                lambda: line(place_xy, z_c, place_xy, z_place, "lower"),
            ):
                why = step()
                if why:
                    raise PieceDropped(f"{t.label}: {why}")
        except PieceDropped:
            holding = False
            log("  piece dropped while moving: opening and going up")
            arm.gripper(open_pct)
            try:
                line(*self._here(arm, ik, T), "escape up", open_pct)
            except Exception:
                pass
            raise
        holding = False

        # ---- release
        crowded_dst = crowded_h_side(dst)
        let_go = min(open_pct, contact + 4.0) if crowded_dst else open_pct
        if real:
            slow_release(arm, let_go)
        else:
            stat["secs"] += 5.0
        line(place_xy, z_place, place_xy, z_c, "lift away", let_go)
        return dict(step=t.label, z_travel_mm=round(z_c), max_grasp_tilt_deg=round(stat["tilt"]), est_seconds=round(stat["secs"]),
                    jaw_shift_mm=round(-r["shift_x"], 1), route="recipe", grasp="recipe", open_pct=round(open_pick, 1),
                    tall_near_path_mm=tall)

    @staticmethod
    def _here(arm, ik, T):
        """(xy, z, xy, z_up) of a straight line up from where the arm is (board mm), for the escape after a drop."""
        q = np.array([arm.joints()[j] for j in C.ARM_JOINTS])
        p = T.R.T @ (ik.fk(q)[:3, 3] - T.t) * 1000
        return p[:2], max(float(p[2]), 0.0), p[:2], C.Z_SAFE - 20.0
