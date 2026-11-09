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
from chess_player.geometry import mm_to_square
from chess_player.motion.arm import FakeArm, look_pose
from chess_player.motion.ik import Unreachable
from chess_player.motion.pick_place import (Blocked, GraspFailed, PickPlace, PieceDropped, Transfer, obstacles_for, plan_move,
                                            surface_z)
from chess_player.motion.slow_grip import slow_close, slow_release

FILE = C.DATA / "wooden_grasps.json"
PAWN = dict(open_pct=16.1, shift_x=-10.6, shift_y=2.7, grip_tip=15.0, lift=30.0, place_dx=3.0, place_dy=0.0, place_extra=2.0,
            contact_pct=8.3, squeeze=C.HOLD_SQUEEZE_PCT, vert_speed=25.0, release_pct=4.0)
# STARTING POINTS for the pieces not taught yet (NOT verified): tall pieces are gripped higher on the body (a low grip on a 55-60 mm
# piece lets the top swing and the piece tips over), squeezed a little harder and lifted/lowered more slowly. Teach each type once with
# calib/pick_place.py --save: the saved numbers (marked "taught") replace these.
START = {
    "p": {},
    "r": dict(grip_tip=15.0),
    "n": dict(grip_tip=17.0, squeeze=1.8, vert_speed=18.0),
    "b": dict(grip_tip=20.0, squeeze=1.8, vert_speed=18.0),
    "q": dict(grip_tip=23.0, squeeze=2.2, vert_speed=14.0),
    "k": dict(grip_tip=25.0, squeeze=2.2, vert_speed=14.0),
}
PATH_STEP_MM = 6.0
CARRY_MM_PER_S = 25.0  # sideways carry; lifting and lowering use the piece's vert_speed
SLIP_PCT = 2.5  # the jaws closing this far beyond the contact reading while carrying = the piece slipped out
NEIGHBOUR_BAND = (14.0, 44.0)  # h-file side: a piece this far along +x of the source (and within +-16 mm in y) is in the moving finger's way
CLEAR_ABOVE_MM = 8.0  # the carried piece's bottom stays this far above the tallest piece near its path
# ALWAYS lift straight up until the carried piece's bottom is this high above the board (the king is 60 mm + 10 mm) before any sideways
# move, whenever other pieces stand on the board. Lower only where the arm cannot reach it (far ranks), and then only if no tall piece
# is near the path.
SAFE_CARRY_BOTTOM_MM = 70.0
# When the jaws close on nothing the pick does NOT retry blindly: it opens, retreats, shifts the pick point by the next step (x = across the
# jaws, where the play is only 2-3 mm; then a little in y) and tries again. The shift that finally grips is REMEMBERED per piece type and square
# (data/pick_offsets.json) and added the next time, so the board's small position errors are learned by use instead of taught by hand.
SEARCH_STEPS = [(0.0, 0.0), (2.5, 0.0), (-2.5, 0.0), (5.0, 0.0), (-5.0, 0.0), (0.0, 3.0), (0.0, -3.0)]
OFFSETS_FILE = C.DATA / "pick_offsets.json"
MAX_LEARNED_MM = 6.0
MIN_CONTACT_PCT = 4.0  # a contact reading below this is the jaws shut on nothing (the thinnest piece, the pawn, holds at about 8 %)
MAX_RETRIES_TALL = 1  # extra tries (after the first) for rook, bishop, queen, king
# Grip sanity check: the jaws closing on the pawn at a width far from the taught contact means they caught it off-centre (head, edge, a
# neighbour). Such a grip often slips or pushes the piece, so it is let go and re-tried with the next shift. 0 turns the check off.
CONTACT_TOL_PCT = float(C._os.environ.get("CHESS_CONTACT_TOL", "1.5"))


def _learned(sym, square):
    # 2026-11-08: NOT applied by default. The same square needed +5 mm in one run and +2.5 in the next (the piece stands a few mm differently
    # every time), and a stored -5 at d4 made a pick push the pawn. They are still RECORDED (data/pick_offsets.json); set CHESS_USE_LEARNED=1 to apply them.
    if C._os.environ.get("CHESS_USE_LEARNED") != "1":
        return np.zeros(2)
    try:
        v = json.loads(OFFSETS_FILE.read_text()).get(sym, {}).get(square)
        return np.array(v, float) if v else np.zeros(2)
    except (OSError, ValueError):
        return np.zeros(2)


def _remember(sym, square, off):
    try:
        d = json.loads(OFFSETS_FILE.read_text())
    except (OSError, ValueError):
        d = {}
    off = np.clip(np.asarray(off, float), -MAX_LEARNED_MM, MAX_LEARNED_MM)
    d.setdefault(sym, {})[square] = [round(float(off[0]), 2), round(float(off[1]), 2)]
    OFFSETS_FILE.write_text(json.dumps(d, indent=2))


# Per-square arm error measured by calib/autocal.py (data/board_residual.json): E(x, y) = how far from the commanded point the arm really puts a
# piece, as a smooth surface over the board. Every pick/place is moved by -(E(square) - E(the square this type's grasp was TAUGHT on)), so a
# taught grip stays valid where it was taught and the part of the error that depends on WHERE on the board it is, is removed everywhere else.
RESIDUAL_FILE = C.DATA / "board_residual.json"
TAUGHT_AT = {"p": ("e2", "e4"), "n": ("b1", "c3"), "q": ("d1", "d3"), "k": ("e1", "e3"), "b": ("c1", "c3"), "r": ("a1", "a3")}  # default teach squares
_resid_cache = {}


def _residual():
    """-> function xy(board mm) -> np.array([ex, ey]), or None (no file, or switched off with CHESS_NO_RESIDUAL=1)."""
    if C._os.environ.get("CHESS_NO_RESIDUAL") == "1" or not RESIDUAL_FILE.exists():
        return None
    mt = RESIDUAL_FILE.stat().st_mtime
    if _resid_cache.get("mt") != mt:
        coef = np.array(json.loads(RESIDUAL_FILE.read_text())["coef"], float)

        def at(xy):
            x = min(max(float(xy[0]), 0.0), C.PLAY_MM) / C.PLAY_MM - 0.5  # off the board (graveyard): the value at the nearest board edge
            y = min(max(float(xy[1]), 0.0), C.PLAY_MM) / C.PLAY_MM - 0.5
            return np.array([1.0, x, y, x * x, x * y, y * y]) @ coef

        _resid_cache.update(mt=mt, fn=at)
    return _resid_cache["fn"]


def _square_xy(name):
    import chess
    from chess_player.geometry import square_center_mm

    return np.array(square_center_mm(chess.parse_square(name)), float)


MAX_CAM_CORR_MM = 3.0 # the camera may move the pick point by at most this much (a bigger 'error' is a misreading, not a misplaced piece)


def recipe(piece):
    """Numbers for one piece type: the pawn's, starting guesses for the others (START), then whatever was taught and saved.
    d["taught"] is True only for numbers that came from data/wooden_grasps.json."""
    piece = piece.lower()
    d = dict(PAWN)
    if piece != "p":
        d["shift_x"] = -C.PIECES[piece]["diameter"] / 2.0
    d.update(START.get(piece, {}))
    d["taught"] = False
    try:
        saved = json.loads(FILE.read_text()).get(piece)
    except (OSError, ValueError):
        saved = None
    if saved:
        saved = dict(saved)
        d["taught"] = bool(saved.pop("complete", True))  # the staged teaching tool saves partial numbers with complete=false
        d.update(saved)
    return d


def chess_square_name(xy):
    import chess

    sq = mm_to_square(*xy)
    return chess.square_name(sq) if sq is not None else "off-board"


def _seg_dist(c, a, b):
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    ab = b - a
    n = float(ab @ ab)
    f = 0.0 if n < 1e-9 else float(np.clip((c - a) @ ab / n, 0.0, 1.0))
    return float(np.linalg.norm(c - (a + ab * f)))


class WoodenPickPlace(PickPlace):
    override = None  # numbers forced over the recipe (the teaching tool passes its command-line flags here)
    locator = None  # vision.piece_locator.PieceLocator: measures where each source piece really stands (None = taught numbers only)
    snap = None  # optional callable(tag): save a camera picture (the pick tool uses it to record the grip point before closing)
    _cam_dev = None  # {source xy: np.array([dx, dy])} measured at the start of this move, arm at home and out of the camera's view

    def play(self, board, move, slots, recover=None):
        self._cam_dev = {}
        if self.locator is not None and not isinstance(self.arm, FakeArm):
            for t, _ in list(obstacles_for(board, slots, plan_move(board, move, slots))):
                sym = t.piece.lower()
                ref = recipe(sym).get("cam_ref")
                if ref is None or tuple(t.src) in self._cam_dev:
                    continue
                m = self.locator.measure(np.array(t.src, float), sym)
                if m is not None:
                    self._cam_dev[tuple(t.src)] = m - np.array(ref, float)
        return super().play(board, move, slots, recover=recover)

    # ---------------------------------------------------------------- planning (a dry run of the same code on a simulated arm)
    def plan_transfer(self, t: Transfer, obstacles=None):
        info = self._run(FakeArm(), t, obstacles, real=False)
        return {"info": info}

    # ---------------------------------------------------------------- execution
    def execute(self, t: Transfer, retries=4, obstacles=None):
        return self._run(self.arm, t, obstacles, real=not isinstance(self.arm, FakeArm), retries=retries)  # a simulated arm cannot feel contact

    def _run(self, arm, t: Transfer, obstacles, real, retries=2):
        sym = t.piece.lower()
        r = recipe(sym)
        if self.override:
            r.update(self.override)
        if not r["taught"] and not self.override:
            self.log(f"  WARNING: the grasp for '{sym}' has not been taught (data/wooden_grasps.json): using starting numbers")
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

        def line(xy0, z0, xy1, z1, what, grip=None, speed=None):
            """Straight Cartesian line (IK every PATH_STEP_MM) from the arm's present place."""
            g = float(arm.joints()["gripper"]) if grip is None else grip
            xy0, xy1 = np.asarray(xy0, float), np.asarray(xy1, float)
            L = float(np.linalg.norm(np.r_[xy1 - xy0, z1 - z0]))
            n = max(1, int(np.ceil(L / PATH_STEP_MM)))
            wps = [arm.joints()]
            for i in range(1, n + 1):
                f = i / n
                wps.append(pose(ik_at(xy0 + (xy1 - xy0) * f, z0 + (z1 - z0) * f, what), g))
            dur = max(0.8, L / (speed or CARRY_MM_PER_S))
            stat["secs"] += dur + 0.4
            why = arm.move_path(wps, dur, check=holding_check if holding else None)
            if why:
                return why
            arm.settle()

        def safe_z(xy, z_min, z_top=None):
            for z in np.arange(C.Z_SAFE if z_top is None else z_top, z_min - 1.0, -5.0):
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
        dev = (self._cam_dev or {}).get(tuple(t.src))
        if dev is not None:  # the camera saw the piece off its square centre (relative to where it stood when the grasp was taught)
            if float(np.linalg.norm(dev)) > MAX_CAM_CORR_MM:
                log(f"  camera: the piece looks {dev[0]:+.1f}, {dev[1]:+.1f} mm off, more than {MAX_CAM_CORR_MM:.0f} mm: ignored")
            else:
                pick_xy = pick_xy + dev
                log(f"  camera: piece stands {dev[0]:+.1f} mm (x, +toward the h-file) and {dev[1]:+.1f} mm (y, +toward rank 8) from "
                    f"where it was taught; pick point moved by that")
        elif r.get("cam_ref") is None and real:
            log("  camera centring: not available for this piece type (re-teach it with calib/teach_stages to record its camera reference)")
        z_above = H + 25.0 + zs
        z_grip = r["grip_tip"] + zs
        # APPROACH HEIGHT = the same safe height as the carry (a carried piece's bottom 70 mm up), not "as high as the arm reaches":
        # the tip goes to the square's pick point at that height, THEN straight down. If the arm cannot reach it there (far ranks)
        # it takes the highest reachable height below, never above this one.
        z_hi = safe_z(pick_xy, z_above, SAFE_CARRY_BOTTOM_MM + r["grip_tip"] + zs if obstacles else None)
        gap_est = float(np.interp(r["contact_pct"], C._GAP_PCT, C._GAP_MM))

        # travel height. The piece goes straight up to z_c BEFORE any sideways move. z_c = the grip height + lift, or, with other pieces on
        # the board, the global safe height (the carried piece's bottom clears the tallest piece, the king), or higher still to clear
        # the tallest piece near this carry path.
        place_est = dst + np.array([-gap_est / 2.0 + r["place_dx"], r["shift_y"] + r["place_dy"]])
        tall = 0.0
        for xy, s in obstacles.items():
            if _seg_dist(xy, src, dst) < 31.0:
                tall = max(tall, C.PIECES[s.lower()]["height"])
        z_path = (tall + CLEAR_ABOVE_MM + r["grip_tip"] + zs) if tall else 0.0
        z_global = (SAFE_CARRY_BOTTOM_MM + r["grip_tip"] + zs) if obstacles else 0.0
        z_c_want = max(z_grip + r["lift"], z_path, z_global)
        z_c = None
        for z in np.arange(z_c_want, z_grip + 4.0, -5.0):
            if all(reachable(p, float(z)) for p in (pick_xy, place_est, (pick_xy + place_est) / 2)):
                z_c = float(z)
                break
        if z_c is None:
            raise Blocked(f"{t.label}: the arm cannot carry it anywhere near {z_c_want:.0f} mm between these squares")
        if z_c < z_path - 0.1:
            raise Blocked(f"{t.label}: the carry path needs {z_path:.0f} mm to clear a {tall:.0f} mm piece, the arm reaches {z_c:.0f} mm")
        if z_c < z_c_want - 0.1:
            log(f"  note: carry height {z_c:.0f} mm (the safe {z_c_want:.0f} mm is out of reach here; no tall piece near the path)")

        log(f"  transfer: {t.label}: pick tip ({pick_xy[0]:.0f}, {pick_xy[1]:.0f}), grip height {z_grip:.0f} mm, jaws {open_pick:.1f} %"
            + (" (crowded on the h-file side)" if open_pick < open_pct else "") + f", carry at {z_c:.0f} mm")

        # ---- per-square arm error (autocal): relative to the square this piece type was taught on
        resid = _residual()
        corr_dst = np.zeros(2)
        if resid is not None:
            t_src, t_dst = (r.get("taught_square"), r.get("taught_dst"))
            d_src, d_dst = TAUGHT_AT.get(sym, ("e2", "e4"))
            corr_src = resid(src) - resid(_square_xy(t_src or d_src))
            corr_dst = resid(dst) - resid(_square_xy(t_dst or d_dst))
            pick_xy = pick_xy - corr_src
            log(f"  board map: pick moved by {-corr_src[0]:+.1f}, {-corr_src[1]:+.1f} mm, place by {-corr_dst[0]:+.1f}, {-corr_dst[1]:+.1f} mm "
                f"(arm error on these squares relative to where the grip was taught)")
        # ---- pick
        # Tall back-rank pieces stand 4 mm from their neighbours: every extra try risks the fingers landing on the king or a bishop and
        # pushing it. So they get at most 2 tries (then the human is asked); only the pawn and the knight keep the full search.
        if sym not in ("p", "n"):
            retries = min(retries, MAX_RETRIES_TALL)
        sq_name = chess_square_name(t.src)
        learned = _learned(sym, sq_name) if real else np.zeros(2)
        if learned.any():
            log(f"  learned offset for {sym.upper()} on {sq_name}: {learned[0]:+.1f}, {learned[1]:+.1f} mm (from earlier picks)")
        base_pick = pick_xy + learned
        step = np.zeros(2)
        for attempt in range(retries + 1):
            step = np.array(SEARCH_STEPS[min(attempt, len(SEARCH_STEPS) - 1)], float)
            if attempt:
                log(f"  attempt {attempt + 1}: shifting the pick point by {step[0]:+.1f}, {step[1]:+.1f} mm")
            pick_xy = base_pick + step
            arm.move_to(pose(ik_at(pick_xy, z_hi, "pick, travel height"), open_pick))
            stat["secs"] += 2.0
            arm.settle()
            line(pick_xy, z_hi, pick_xy, z_above, "pick, above the piece", open_pick)
            line(pick_xy, z_above, pick_xy, z_grip, "pick, grip height", open_pick)
            if real:  # what the arm really did at the grip point: joints vs commanded (a gap = the arm is not where the plan thinks)
                pr = arm.present()
                gap_deg = {j: round(float(pr[j]) - float(seed[i]), 1) for i, j in enumerate(C.ARM_JOINTS)}
                log(f"  at the grip point: real joints minus commanded (deg) {gap_deg}; jaws at {float(pr['gripper']):.1f} %")
                if self.snap:
                    self.snap("before_close")
            if real:
                contact, fin = slow_close(arm, squeeze=r["squeeze"])
            else:
                contact, fin = r["contact_pct"], r["contact_pct"] - 0.2
                stat["secs"] += 6.0
            if contact is not None and real and contact < MIN_CONTACT_PCT:
                log(f"  attempt {attempt + 1}: the jaws closed fully ({contact:.1f} %): nothing in the hand")
                contact, fin = None, contact  # fully closed = no piece, whatever the load reading said
            if contact is not None and real and CONTACT_TOL_PCT > 0 and attempt < retries and abs(contact - r["contact_pct"]) > CONTACT_TOL_PCT:
                log(f"  attempt {attempt + 1}: gripped at {contact:.1f} % but the taught width is {r['contact_pct']:.1f} % (off by more than "
                    f"{CONTACT_TOL_PCT:g}): probably off-centre, letting go and trying again")
                contact, fin = None, contact
                slow_release(arm, open_pick)
                line(pick_xy, z_grip, pick_xy, z_hi, "retreat", open_pick, speed=r["vert_speed"])
                continue
            if contact is not None:
                break
            log(f"  attempt {attempt + 1}: the jaws closed on nothing ({fin:.1f} %)")
            slow_release(arm, open_pick)
            line(pick_xy, z_grip, pick_xy, z_hi, "retreat", open_pick, speed=r["vert_speed"])
        else:
            raise GraspFailed(f"{t.label}: nothing in the hand after {retries + 1} tries")
        if real and step.any():  # it only gripped after a shift: remember it for this square
            _remember(sym, sq_name, learned + step)
            log(f"  the shift {step[0]:+.1f}, {step[1]:+.1f} mm worked: remembered for {sym.upper()} on {sq_name} (total {learned[0] + step[0]:+.1f}, {learned[1] + step[1]:+.1f})")
        contact_pct[0] = float(contact)
        gap = float(np.interp(contact, C._GAP_PCT, C._GAP_MM))
        log(f"  gripped: contact at {contact:.1f} % = {gap:.1f} mm")

        # ---- carry
        place_xy = dst + np.array([-gap / 2.0 + r["place_dx"], r["shift_y"] + r["place_dy"]]) - corr_dst
        z_place = r["grip_tip"] + r["place_extra"] + zd
        holding = True
        try:
            for step in (
                lambda: line(pick_xy, z_grip, pick_xy, z_c, "lift", speed=r["vert_speed"]),
                lambda: line(pick_xy, z_c, place_xy, z_c, "carry"),
                lambda: line(place_xy, z_c, place_xy, z_place, "lower", speed=r["vert_speed"]),
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
        # let go by opening only release_pct past the contact reading (about 8 mm of gap at 4 %), never above the taught maximum
        let_go = min(open_pct, contact + r["release_pct"], C.GRIP_OPEN_MAX_PCT)
        if real:
            slow_release(arm, let_go)
        else:
            stat["secs"] += 5.0
        line(place_xy, z_place, place_xy, z_c, "lift away", let_go, speed=r["vert_speed"])
        return dict(step=t.label, z_travel_mm=round(z_c), max_grasp_tilt_deg=round(stat["tilt"]), est_seconds=round(stat["secs"]),
                    jaw_shift_mm=round(-r["shift_x"], 1), route="recipe", grasp="recipe", open_pct=round(open_pick, 1),
                    tall_near_path_mm=tall)

    @staticmethod
    def _here(arm, ik, T):
        """(xy, z, xy, z_up) of a straight line up from where the arm is (board mm), for the escape after a drop."""
        q = np.array([arm.joints()[j] for j in C.ARM_JOINTS])
        p = T.R.T @ (ik.fk(q)[:3, 3] - T.t) * 1000
        return p[:2], max(float(p[2]), 0.0), p[:2], C.Z_SAFE - 20.0
