"""Teach ONE piece type its grasp, stage by stage, with the human validating every stage (wooden rig).

  python -m chess_player.calib.teach_stages --piece n --square e2 --dst e4

Commands are read one per line from stdin (type them, or feed a file with `tail -f cmds | python -u -m ...`). The piece must stand
on the centre of --square, facing the way you will always set it (the knight: toward the robot). Four stages, each ends with `ok`
(= validated: the numbers are SAVED in data/wooden_grasps.json and the joint poses + camera shots in data/teach/<piece>/):

  1 APPROACH  the tip (fixed finger's inner face) goes above the piece at the safe height, then straight down to just under half the
              piece height (--grip-frac) and STOPS there with the jaws open.  x 2 | y -1 | z -3  nudge the tip (mm, relative);
              open 14 sets the jaw opening.  ok -> saves shift_x, shift_y, grip_tip, open_pct.
  2 GRIP      close  = slow closing until contact (stops on the jaw lagging the command OR the motor load rising), light squeeze.
              open  = let go again (back to the approach opening) and retry;  tighter 0.5 | looser 0.5  changes the squeeze (%).
              ok -> saves contact_pct, squeeze.
  3 LIFT      lift = straight up, slowly, to the safe carry height (the carried piece's bottom >= 70 mm above the board).
              speed 12 = lift/lower speed (mm/s), then lift again.  ok -> saves vert_speed.
  4 PLACE     carry = straight sideways to above --dst;  place = straight down onto it;  nudge;  release [N] = slow release, opening only N % (default 4) past the contact reading;
              leave = lift away.  ok -> saves place_dx, place_dy, place_extra and marks the piece TAUGHT, then the arm goes home.
Any stage: status | photo | home (abort, nothing more saved) .

NUDGING (stages 1 and 4) works like nudge_calibrate, FROM YOUR SEAT (the human's side, the robot at the far end):
    d3  3 mm to your right (toward the a-file)      a3  3 mm to your left (toward the h-file)
    w2  2 mm AWAY from you (toward the robot)       s2  2 mm TOWARD you (toward rank 8)
    r2  2 mm UP                                      f2  2 mm DOWN        (no number = 1 mm; capital letter = 5 mm; "d3 w2 f1" = several)
    also "up 5" / "down 5".  Enter on an empty line = ok.
--keys: single key presses (w a s d r f, capitals = 5 mm), Enter = ok, Esc = home; start typing any other word (close, lift, carry ...) + Enter; words that start with a move letter (release, speed, status) need a ':' first, e.g. :release
"""
import argparse
import json
import re
import sys
import time

import chess
import cv2
import numpy as np

from chess_player import config as C
from chess_player.calib.locate_piece import grab
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.arm import Arm, FakeArm, look_pose
from chess_player.motion.ik import IK, Unreachable
from chess_player.motion.pick_place import surface_z
from chess_player.motion.slow_grip import STEP_PCT, slow_release
from chess_player.motion.wooden_pick_place import FILE, PATH_STEP_MM, PAWN, SAFE_CARRY_BOTTOM_MM, recipe

DIRS = {"a": (1, 0, 0), "d": (-1, 0, 0), "w": (0, -1, 0), "s": (0, 1, 0), "r": (0, 0, 1), "f": (0, 0, -1)}  # board x toward h-file, y toward rank 8, z up
MOVE_RE = re.compile(r"(?:[adwsrf]\d*\.?\d*\s*)+", re.I)


def parse_move(text):
    """'d3 w2 F' -> (dx, dy, dz) mm from the human's seat, or None if it is not a move."""
    t = text.strip()
    w = t.lower().split()
    if len(w) == 2 and w[0] in ("up", "down"):
        try:
            return (0.0, 0.0, float(w[1]) * (1 if w[0] == "up" else -1))
        except ValueError:
            return None
    if not t or not MOVE_RE.fullmatch(t):
        return None
    v = np.zeros(3)
    for ch, num in re.findall(r"([adwsrfADWSRF])(\d*\.?\d*)", t):
        n = float(num) if num else (5.0 if ch.isupper() else 1.0)
        v += np.array(DIRS[ch.lower()]) * n
    return tuple(v)


def read_key_command(prompt):
    """One key press (Linux terminal). Move keys come back as text ('d', 'W'), Enter = 'ok', Esc = 'home'. Any other letter starts a typed
    command: that letter plus the rest of the line you type (so `close` + Enter just works). Each move key moves 1 mm (capital 5 mm);
    for "d3 w2" style moves use the typed mode (no --keys)."""
    import os
    import termios
    import tty

    fd = sys.stdin.fileno()
    sys.stdout.write(prompt)
    sys.stdout.flush()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = os.read(fd, 1).decode(errors="ignore")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch in ("\r", "\n"):
        print("ok")
        return "ok"
    if ch == "\x1b":
        print("home")
        return "home"
    if ch == "\x03":
        raise KeyboardInterrupt
    if ch == ":":  # a typed command that starts with a move letter: :release  :speed 12  :status
        sys.stdout.write(":")
        sys.stdout.flush()
        return input()
    if ch.isalpha() and ch.lower() not in DIRS:
        sys.stdout.write(ch)
        sys.stdout.flush()
        return ch + input()  # echo is back on: the rest of a typed command (close, tighter 0.5, lift ...)
    print(ch)
    return ch


LAG_PCT = 0.8  # contact = the measured jaw is this far behind the command ...
LOAD_JUMP = 6.0  # ... or the gripper motor load is this many % above its free-running load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece", required=True, choices=list("pnbrqk"))
    ap.add_argument("--square", default="e2")
    ap.add_argument("--dst", default="e4")
    ap.add_argument("--grip-frac", type=float, default=0.45, help="approach height as a fraction of the piece height")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wide", action="store_true",
                    help="approach with the full taught opening; default: the narrow opening the game uses next to neighbours (contact + 2.5 %%)")
    ap.add_argument("--keys", action="store_true", help="single key presses (run it in your own terminal)")
    a = ap.parse_args()

    sym = a.piece
    H = C.PIECES[sym]["height"]
    r = recipe(sym)
    src = np.array(square_center_mm(chess.parse_square(a.square)), float)
    dst = np.array(square_center_mm(chess.parse_square(a.dst)), float)
    zs, zd = surface_z(tuple(src)), surface_z(tuple(dst))
    T, ik = BoardToRobot.load(), IK()
    look = {**look_pose(), "gripper": C.GRIPPER_LOOK}
    seed = np.array([look[j] for j in C.ARM_JOINTS], float)
    seed[4] = C.WRIST_ROLL_DEG
    out = C.DATA / "teach" / (sym + "_dry" if a.dry_run else sym)
    out.mkdir(parents=True, exist_ok=True)
    arm = FakeArm() if a.dry_run else Arm()
    stages_log = {}

    def ik_at(xy, z):
        nonlocal seed
        q, err, tilt = ik.solve(T(xy[0], xy[1], z), seed, C.GRASP_MAX_TILT)
        if err > C.IK_MAX_ERR_MM or tilt > C.GRASP_MAX_TILT:
            raise Unreachable(f"IK error {err:.1f} mm, tilt {tilt:.0f} deg at ({xy[0]:.0f}, {xy[1]:.0f}, {z:.0f})")
        seed = q
        return q

    def pose(q, grip):
        return {**dict(zip(C.ARM_JOINTS, q)), "gripper": float(grip)}

    def tip_now():
        q = np.array([arm.present()[j] for j in C.ARM_JOINTS])
        return T.R.T @ (ik.fk(q)[:3, 3] - T.t) * 1000

    cur = np.zeros(3)  # where the tip is commanded (board mm)

    def line(p1, speed=25.0, grip=None):
        """Straight Cartesian line from the commanded tip to p1 (x, y, z)."""
        nonlocal cur
        p1 = np.asarray(p1, float)
        g = float(arm.joints()["gripper"]) if grip is None else grip
        L = float(np.linalg.norm(p1 - cur))
        n = max(1, int(np.ceil(L / PATH_STEP_MM)))
        wps = [arm.joints()] + [pose(ik_at((cur + (p1 - cur) * i / n)[:2], (cur + (p1 - cur) * i / n)[2]), g) for i in range(1, n + 1)]
        arm.move_path(wps, max(0.8, L / speed))
        arm.settle()
        cur = p1

    def reach(xy, z):
        try:
            q, err, tilt = ik.solve(T(xy[0], xy[1], z), seed, C.GRASP_MAX_TILT)
        except Exception:
            return False
        return err <= C.IK_MAX_ERR_MM and tilt <= C.GRASP_MAX_TILT

    def photo(tag):
        files = []
        try:
            f = grab(str(C.EXT_CAMERA_URL))
            p = out / f"{tag}.jpg"
            cv2.imwrite(str(p), f)
            files.append(str(p))
        except Exception as e:  # camera window not running
            print("  (no camera picture:", e, ")")
        return files

    def snapshot(tag):
        try:
            tip = tip_now().round(1).tolist()
        except Exception:
            tip = None
        stages_log[tag] = dict(joints={k: round(float(v), 2) for k, v in arm.present().items()}, tip_board_mm=tip, photos=photo(tag),
                               numbers={k: v for k, v in r.items() if k != "taught"})
        (out / "stages.json").write_text(json.dumps(stages_log, indent=2))
        print(f"  joints {stages_log[tag]['joints']}\n  tip (board mm) {tip}; picture saved: {stages_log[tag]['photos']}")

    def save(complete=False):
        if a.dry_run:
            print("  (dry run: not saved)")
            return
        try:
            d = json.loads(FILE.read_text())
            bk = C.DATA / "teach" / "backups"
            bk.mkdir(parents=True, exist_ok=True)
            (bk / f"wooden_grasps.{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(d, indent=2))
        except (OSError, ValueError):
            d = {}
        keep = {k: v for k, v in r.items() if k != "taught"}
        keep["taught_square"], keep["taught_dst"] = a.square, a.dst  # the board map (autocal) corrects relative to these
        d[sym] = {**keep, "complete": complete}
        FILE.write_text(json.dumps(d, indent=2))
        print(f"  SAVED {sym} ({'taught, complete' if complete else 'partial: stages so far'}) in {FILE.name}")

    def nudge(v):
        nonlocal cur
        new = cur + np.asarray(v, float)
        if new[2] < zs + 6.0:
            print("  refused: below 6 mm")
            return
        try:
            line(new, speed=15.0)
        except Unreachable as e:
            print("  unreachable:", e)
            return
        print(f"  tip now (board mm) {tip_now().round(1).tolist()}  (commanded {np.round(cur, 1).tolist()})")

    open_pct = min(r["open_pct"], C.GRIP_OPEN_MAX_PCT)
    centre_tip = src + np.array([r["shift_x"], r["shift_y"]])
    z0 = a.grip_frac * H + zs
    print(f"Piece {sym.upper()} ({H:.0f} mm) on {a.square}, facing as you set it. Approach: tip over ({centre_tip[0]:.1f}, {centre_tip[1]:.1f}), "
          f"down to {z0:.1f} mm, jaws {open_pct:.1f} %.")
    # approach at the SAME safe height as the carry (piece-bottom 70 mm up), over the pick point, then straight down
    z_hover = SAFE_CARRY_BOTTOM_MM + r["grip_tip"] + zs
    z_hi = next((float(z) for z in np.arange(z_hover, z0, -5.0) if reach(centre_tip, float(z))), None)
    if z_hi is None:
        raise SystemExit("cannot reach above the piece")
    if not a.dry_run:  # every time: the wrong --piece letter silently overwrote another piece's numbers twice
        names = dict(p="pawn", r="rook", n="knight", b="bishop", q="queen", k="king")
        warn = f"  (WARNING: '{sym}' is already taught and will be overwritten)" if r["taught"] else ""
        ans = input(f"Teaching the {names[sym].upper()} on {a.square}.{warn} Is a {names[sym]} standing there? Type '{names[sym]}' to confirm: ")
        if ans.strip().lower() != names[sym]:
            raise SystemExit(f"cancelled: nothing changed. Run it again with --piece set to the piece that is really on {a.square} (p r n b q k).")
    input("The piece must be CENTRED on the square now (the camera records where it stands). Hands clear, a hand near the power switch. Enter = go: ")
    r.pop("cam_ref", None)  # a stale reference from an earlier teach must never survive
    if not a.dry_run:  # measured with the arm at home (out of the camera's view), BEFORE it moves
        try:
            from chess_player.vision.ext_camera import ExtCamera
            from chess_player.vision.piece_locator import PieceLocator

            _cam = ExtCamera()
            try:
                time.sleep(1.5)
                m = PieceLocator(_cam, log=print).measure(src, sym)
            finally:
                _cam.close()
        except Exception as e:  # camera window not running ...
            m = None
            print(f"  camera: {e}")
        if m is None:
            print("  WARNING: no camera reference recorded (the camera could not measure the piece): this teach will work WITHOUT camera centring")
        else:
            r["cam_ref"] = [round(float(m[0]), 2), round(float(m[1]), 2)]
            print(f"  camera reference recorded: the piece stands {m[0]:+.1f}, {m[1]:+.1f} mm from the square centre (saved as cam_ref)")
    approach_open = open_pct
    if r.get("taught") and "contact_pct" in r and not a.wide:
        approach_open = min(open_pct, r["contact_pct"] + 2.5)
        print(f"  approach with the narrow opening the game uses next to neighbours: {approach_open:.1f} % (use --wide for {open_pct:.1f} %)")

    holding_check_on = False
    z_c = None
    st4 = dict(carried=False, lowered=False, released_at=None)  # what stage 4 has really done
    try:
        arm.move_to(pose(ik_at(centre_tip, z_hi), approach_open))
        arm.settle()
        cur = np.array([*centre_tip, z_hi])
        line([*centre_tip, z0 + 20.0], 25.0)
        line([*centre_tip, z0], 12.0)
        stage = 1
        print("\nSTAGE 1 APPROACH: the tip stopped at the height above. Look, nudge from your seat (d/a right/left, w/s away/toward you, r/f up/down; 3 = mm, capital = 5), then Enter/ok.")
        snapshot("stage1_approach")
        free_load = None
        while True:
            try:
                cmd = (read_key_command(f"stage{stage}> ") if a.keys else input(f"stage{stage}> ")).strip()
            except EOFError:
                break
            mv = parse_move(cmd) if stage in (1, 4) else None
            cmd = "ok" if cmd == "" else (cmd if mv is not None else cmd.lower())
            parts = cmd.split()
            c = parts[0]
            try:
                if c == "home":
                    print("  aborted, nothing more saved")
                    break
                if c == "status":
                    print(f"  stage {stage}; tip {tip_now().round(1).tolist()}; gripper {arm.present()['gripper']:.1f} %; numbers {r}")
                elif c == "photo":
                    print("  ", photo(f"manual_{int(time.time())}"))
                elif mv is not None:
                    nudge(mv)
                elif c == "open":
                    if stage == 4 and len(parts) == 1:  # `open` in stage 4 = let go: only release_pct past the contact reading
                        to = min(r["contact_pct"] + r["release_pct"], C.GRIP_OPEN_MAX_PCT)
                    else:
                        to = min(float(parts[1]) if len(parts) > 1 else open_pct, C.GRIP_OPEN_MAX_PCT)
                    if stage == 1:
                        open_pct = r["open_pct"] = to
                    if stage == 4 and st4["lowered"] and st4["released_at"] is None:
                        st4["released_at"] = cur.copy()
                    print(f"  jaws {slow_release(arm, to):.1f} %")
                elif c == "close" and stage == 2:
                    cmd_pct = float(arm.joints()["gripper"])
                    free_load = arm.gripper_load() if not a.dry_run else 0.0
                    contact = None
                    while cmd_pct > 1.5:
                        cmd_pct -= STEP_PCT
                        arm.gripper(cmd_pct, 0.35)
                        time.sleep(0.12)
                        p = float(arm.present()["gripper"])
                        ld = arm.gripper_load() if not a.dry_run else 0.0
                        if p > cmd_pct + LAG_PCT or ld > free_load + LOAD_JUMP:
                            contact = p
                            print(f"  CONTACT at {p:.1f} % (command {cmd_pct:.1f} %, load {ld:.0f} % vs free {free_load:.0f} %)")
                            break
                    if contact is None:
                        print(f"  closed on AIR (jaws {arm.present()['gripper']:.1f} %): nothing between the pads. `open`, adjust, retry")
                    else:
                        arm.gripper(max(0.0, contact - r["squeeze"]), 0.8)
                        arm.dwell()
                        r["contact_pct"] = float(contact)
                        gap = float(np.interp(contact, C._GAP_PCT, C._GAP_MM))
                        print(f"  gripped: gap {gap:.1f} mm (width of the part held), squeezing to {arm.present()['gripper']:.1f} %")
                        snapshot("stage2_grip")
                elif c in ("tighter", "looser") and stage == 2:
                    r["squeeze"] = max(0.0, r["squeeze"] + (1 if c == "tighter" else -1) * float(parts[1] if len(parts) > 1 else 0.5))
                    arm.gripper(max(0.0, r["contact_pct"] - r["squeeze"]), 0.8)
                    arm.dwell()
                    print(f"  squeeze {r['squeeze']:.1f} % -> jaws {arm.present()['gripper']:.1f} %")
                elif c == "speed" and stage == 3:
                    r["vert_speed"] = float(parts[1])
                    print(f"  lift/lower speed {r['vert_speed']:g} mm/s")
                elif c == "lift" and stage == 3:
                    grip_tip = cur[2] - zs
                    r["grip_tip"] = float(grip_tip)
                    want = max(cur[2] + r["lift"], SAFE_CARRY_BOTTOM_MM + grip_tip + zs)
                    z_c = next((float(z) for z in np.arange(want, cur[2], -5.0) if reach(cur[:2], float(z))), None)
                    if z_c is None:
                        print("  cannot lift")
                        continue
                    print(f"  lifting straight up to {z_c:.0f} mm at {r['vert_speed']:g} mm/s ...")
                    holding_check_on = True
                    line([*cur[:2], z_c], r["vert_speed"])
                    print(f"  at {z_c:.0f} mm: is the piece still held, upright? gripper {arm.present()['gripper']:.1f} % (contact was {r['contact_pct']:.1f} %)")
                    snapshot("stage3_lifted")
                elif c == "carry" and stage == 4:
                    gap = float(np.interp(r["contact_pct"], C._GAP_PCT, C._GAP_MM))
                    pl = dst + np.array([-gap / 2.0 + r["place_dx"], r["shift_y"] + r["place_dy"]])
                    if z_c is None or not reach(pl, z_c):
                        z_c = next((float(z) for z in np.arange(cur[2], zs + 20, -5.0) if reach(pl, float(z)) and reach(cur[:2], float(z))), None)
                    line([*pl, z_c], 25.0)
                    st4["carried"] = True
                    print(f"  above {a.dst}; tip {tip_now().round(1).tolist()}")
                elif c in ("place", "lower") and stage == 4:
                    if not st4["carried"]:
                        print("  carry first (the piece is still above its pick-up square)")
                        continue
                    st4["lowered"] = True
                    z_t = zd + r["grip_tip"] + r["place_extra"]
                    if cur[2] > z_t + 0.5:  # only lowers: a repeated `place` keeps your nudges
                        line([cur[0], cur[1], z_t], r["vert_speed"])
                    print(f"  down; tip {tip_now().round(1).tolist()} (the piece should be on the square centre; nudge, then release)")
                    snapshot("stage4_lowered")
                elif c == "release" and stage == 4:
                    if not st4["lowered"]:
                        print("  place first: releasing in the air would drop the piece")
                        continue
                    st4["released_at"] = cur.copy()
                    if len(parts) > 1:
                        r["release_pct"] = float(parts[1])  # release 3 = open 3 % past the contact reading
                    to = min(r["contact_pct"] + r["release_pct"], C.GRIP_OPEN_MAX_PCT)
                    print(f"  letting go: {r['release_pct']:g} % past the contact reading ({r['contact_pct']:.1f} %) -> jaws to {to:.1f} %")
                    print(f"  jaws {slow_release(arm, to):.1f} %")
                elif c in ("leave", "liftoff") and stage == 4:
                    line([cur[0], cur[1], z_c or cur[2] + 40], r["vert_speed"])
                    snapshot("stage4_placed")
                elif c == "ok":
                    if stage == 1:
                        r["shift_x"], r["shift_y"] = float(cur[0] - src[0]), float(cur[1] - src[1])
                        r["grip_tip"] = float(cur[2] - zs)
                        r["open_pct"] = float(open_pct)
                        save()
                        stage = 2
                        print("\nSTAGE 2 GRIP: `close` (slow). Watch the pads and the piece; tighter/looser 0.5 changes the squeeze; open to retry; ok when it is held well.")
                    elif stage == 2:
                        if "contact_pct" not in r or not holding_state(arm, r):
                            print("  not gripped yet: use close first")
                            continue
                        save()
                        stage = 3
                        print("\nSTAGE 3 LIFT: `lift` (slow, straight up to the safe height). speed N changes mm/s. ok when it stays held and upright.")
                    elif stage == 3:
                        if z_c is None:
                            print("  lift first")
                            continue
                        for k in ("place_dx", "place_dy", "place_extra"):
                            r[k] = PAWN[k]  # never inherit old place numbers: stage 4 measures them fresh
                        save()
                        stage = 4
                        print("\nSTAGE 4 PLACE: carry, place, nudge so it sits centred, release, leave. ok at the end.")
                    elif stage == 4:
                        if st4["released_at"] is None:
                            print("  not done: carry, place (nudge it centred), release, leave first")
                            continue
                        gap = float(np.interp(r["contact_pct"], C._GAP_PCT, C._GAP_MM))
                        rel = st4["released_at"]
                        dx = float(rel[0] - dst[0] + gap / 2.0)
                        dy = float(rel[1] - dst[1] - r["shift_y"])
                        ex = float(rel[2] - zd - r["grip_tip"])
                        if abs(dx) > 15 or abs(dy) > 15 or not -3.0 <= ex <= 8.0:
                            print(f"  these numbers look wrong (place_dx {dx:.1f}, place_dy {dy:.1f}, place_extra {ex:.1f} mm): NOT saved. "
                                  "Carry and place onto the square first.")
                            continue
                        r["place_dx"], r["place_dy"], r["place_extra"] = dx, dy, ex
                        save(complete=True)
                        break
                else:
                    print("  ? not valid in this stage (words starting with w a s d r f need a ':' first, e.g. :release)")
            except (ValueError, IndexError, KeyError):
                print("  ? bad command")
            except Unreachable as e:
                print("  unreachable:", e)
    finally:
        try:
            # a piece is in the hand only if the jaws are still at about the contact reading (not after a release / `open`)
            in_hand = "contact_pct" in r and float(arm.present()["gripper"]) < r["contact_pct"] + 1.0 and float(arm.present()["gripper"]) > 3.0
            if in_hand:
                z_floor = surface_z((cur[0], cur[1])) + r["grip_tip"] + 2.0
                if cur[2] > z_floor + 3.0:  # put it down first, never release in the air
                    print("  putting the piece down before letting go ...")
                    line([cur[0], cur[1], z_floor], 10.0)
                slow_release(arm, min(r["contact_pct"] + r["release_pct"], open_pct))
            if z_c or cur[2] > 0:
                line([cur[0], cur[1], max(cur[2], 60.0 + zs)], 25.0)
            arm.move_to(look, 3.0)
        except Exception as e:
            print("(could not return home cleanly:", e, ")")
        arm.close()


def holding_state(arm, r):
    return float(arm.present()["gripper"]) < r["contact_pct"] + 0.5


if __name__ == "__main__":
    main()
