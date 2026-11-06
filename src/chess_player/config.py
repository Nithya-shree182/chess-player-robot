"""All tunable numbers in one place. Values marked MEASURE are placeholders until measured on the real setup.

Frames
- board frame (mm): origin at the outer corner of a1, x along files a->h, y along ranks 1->8, z up from the
  board surface. Square centers are ((file + 0.5) * SQUARE_MM, (rank + 0.5) * SQUARE_MM).
- robot frame (m): the URDF base_link frame. calib/board_to_robot.py measures the transform between the two.
"""
from pathlib import Path

PKG = Path(__file__).resolve().parent  # src/chess_player
ROOT = PKG.parents[1]  # ~/so101-lerobot
DATA = ROOT / "data"  # calibration results, poses, datasets, models (created on demand)

# ---------------------------------------------------------------- hardware
import os as _os
import sys as _sys

_WIN = _sys.platform == "win32"
# 2026-11-04/05: new rig. Windows: follower COM5, leader COM6. Linux: the udev symlinks /dev/so100_follower and
# /dev/so100_leader (setup/99-so100.rules matches the NEW boards' serials 58FA082690 / 5AE6082864; see
# docs/UBUNTU_MIGRATION.md). The LeRobot calibration ids are the same on both systems: my_follower / my_leader (their
# files are in data/calibration_backup/ and are copied to ~/.cache/huggingface/lerobot/calibration by
# setup/migrate_calibration.sh).
FOLLOWER_PORT = _os.environ.get("CHESS_FOLLOWER_PORT", "COM5" if _WIN else "/dev/so100_follower")
FOLLOWER_ID = "my_follower"
LEADER_PORT = _os.environ.get("CHESS_LEADER_PORT", "COM6" if _WIN else "/dev/so100_leader")
LEADER_ID = "my_leader"
# Gripper camera role ("robot eye"). 2026-11-04: there is ONE webcam, mounted on top of the gripper; it plays the
# main (third-person) role below while the arm rests at the look pose, so the separate robot-eye role is off.
WRIST_CAMERA_ENABLED = False
CAMERA ="/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"
CAM_W, CAM_H, CAM_FPS = 640, 480, 30
# Main vision camera: sees the whole board (a USB webcam opened as MJPG 1280x720, or an MJPEG URL). A device is an
# integer index (Windows, DirectShow) or a /dev path (Linux). Windows index 1 = the Ausha webcam on the gripper
# (index 0 is the laptop's built-in camera; indexes can change if cameras are plugged in a different order).
EXT_CAMERA_URL = _os.environ.get(
    "CHESS_CAMERA_URL",
    1 if _WIN else "/dev/v4l/by-id/usb-Ausha_Webcam_Ausha_Webcam-video-index0",
)
# 2026-11-05: the phone (IP Webcam) third-person setup was dropped; the Ausha webcam was moved off the gripper and
# mounted as the fixed third-person camera instead (back to the original rig's arrangement). Override per network
# with $env:CHESS_CAMERA_URL / export CHESS_CAMERA_URL if it ever needs to point elsewhere.
# Fallback only (no board calibration yet): True = the main camera is on the robot's side (gripper webcam), so the
# back rank at the bottom of its image is the ROBOT's; False = human's side (the original rig). With a board
# calibration, the robot's colour comes from which back rank is nearer its base, whatever the camera does.
CAMERA_ON_ROBOT_SIDE = False
# old phone stream: "http://192.168.101.104:8080/stream"
EXT_CAMERA_SIZE = (1280, 720)
# A USB camera can only be opened by one program: while the pipeline runs it re-serves the raw frames here
EXT_SHARED_URL = "http://localhost:8770/stream/ext_raw"
WRIST_STREAM_URL = "http://localhost:8081/stream"  # wrist_stream.py (shares the gripper cam on the network)
EXT_WHITE_BALANCE = True  # make the webcam's colours match the robot camera before anything uses the frames
# "Zoom in" (vision/zoom.py): when a square's reading is unsure, re-read it on a full-resolution still from the phone
ZOOM_ENABLED = False  # needed the phone's full-resolution stills; no phone in this setup
HIRES_PHOTO_SIZE = "4000x2256"  # the phone app's photo size (stream stays 1280x720); 16:9 like the stream
HIRES_CORNERS = DATA / "ext_corners_hires.json"  # the board's corners in that still (fitted by hand/auto; refit if the phone moves)
ZOOM_TRIGGER_CONF = 0.6  # squares read with less confidence than this are re-read at full resolution
ZOOM_IMGSZ = 1920  # YOLO input size for the still
ZOOM_MIN_GAP_S = 4.0  # at most one background hi-res read this often
ZOOM_MAX_AGE_S = 6.0  # a hi-res reading older than this is not reused
EXT_WB_TINT = (1.0, 1.0, 1.0)  # neutral for the phone (old rig: (1.052, 1.002, 0.946) = white as the gripper camera saw it)
EXT_WB_WHITE_LEVEL = 235.0  # brightness the light squares / white pieces are brought to
WATCH_PERIOD_S = 0.3  # board watcher runs YOLO this often (~75 ms per frame on the RTX 5050)
WATCH_STABLE_N = 3  # readings that must agree before a change counts: ~1 s from the real board to the window
LIVE_SYNC_DEFAULT = True  # the window's board follows the real board (when no game is being played)
STOCKFISH = ROOT / "bin" / ("stockfish.exe" if _WIN else "stockfish")
URDF = PKG / "urdf" / "so101_new_calib.urdf"
EE_FRAME = "gripper_frame_link"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_JOINTS = ARM_JOINTS + ["gripper"]

# ---------------------------------------------------------------- calibration files
BOARD_TO_ROBOT = DATA / "board_to_robot.json"  # from calib/board_to_robot.py
BOARD_CORRECTION = DATA / "board_correction.json"  # from calib/nudge_calibrate.py (optional 2-D fix on top of it)
INTRINSICS = DATA / "camera_intrinsics.npz"  # from calib/camera_intrinsics.py (optional)
MANUAL_CORNERS = DATA / "manual_corners.json"  # wrist cam: fallback when ArUco markers aren't visible
EXT_CORNERS = DATA / "ext_corners.json"  # third-person cam: clicked a1/h1/h8/a8 playing-area corners
LOOK_POSE = DATA / "look_pose.json"  # from tools/read_positions.py --save; falls back to data/poses/home_pose.json
HOME_POSE = DATA / "poses" / "home_pose.json"
YOLO_WEIGHTS = DATA / "yolo" / "best.pt"  # our own model (vision/train_yolo.py); preferred when present
# Until then: a public chess-piece YOLOv8m (KanisornPutta/chess-model-yolov8m on Hugging Face, scanned with picklescan),
# best of 7 tested on this board (2026-10-17). Confirmed camera readings become training data for our own model.
YOLO_PRETRAINED = DATA / "models" / "pretrained_chess" / "KanisornPutta_chess-model-yolov8m" / "chess-model-yolov8m.pt"
YOLO_IMGSZ = 960
# King vs queen: YOLO mixes them up; their heights differ (64 vs 58 mm). When YOLO's confidence for a K/Q is below
# KQ_TRUST_CONF, the piece's height measured from its box (corrected by the other pieces in the frame) decides.
KQ_HEIGHT_CHECK = True
KQ_TRUST_CONF = 0.5  # was 0.85: with our own fine-tuned model the box-height guess flipped a correct K (0.80) to Q
# Live sync: a piece that stays on its square keeps its known type unless the cameras (stable over several frames)
# call it something else with at least this confidence. Both cameras agreeing raises the confidence (0.6 and 0.7
# -> 0.88), disagreeing drops it, so in practice: both cameras agree, or the webcam alone is very sure.
SYNC_TYPE_CONF = 0.8
RECHECK_TYPE_CONF = 0.5  # ⟳ Recheck: a fresh look, a bit more willing to change types
# Gripper camera as the second judge (vision/wrist_view.py), at the look pose
WRIST_CORNERS = DATA / "wrist_corners.json"  # auto-calibrated board corners in the gripper image
WRIST_GRIPPER_MASK = (0.19, 0.78, 0.60)  # fingers in the gripper image: x from, x to, y below (fractions)
WRIST_MIN_SCORE = 40.0  # light-dark contrast of the fitted grid; below = calibration doesn't fit (redo it)
WRIST_WATCH_PERIOD_S = 0.6
EYE_ADD_MIN_CONF = 0.55  # the robot eye adds a piece the webcam can't see (hidden behind others) at this confidence
LIVE_SYNC_HOLD_S = 1.2  # live sync applies a change only after the cameras have shown it this long (no flicker)

# ---------------------------------------------------------------- board (measured 2026-10-17)
# Cross-section from the playing area outwards: 8 squares | thin light line | wooden border.
# 2026-11-04: NEW board (Toy Imagine magnetic folding board): squares measured 25.0 mm with a ruler, board 248 mm
# outside (24.8 cm listing), so ~24 mm from the playing area to the outer edge (split between line and border: a
# guess, only the sum matters). The old rig had 33.3 mm squares.
SQUARE_MM = 25.0  # one square
BOARD_LINE_MM = 2.0  # line around the playing area (guess)
BOARD_BORDER_MM = 22.0  # border outside the line (guess; line + border = 24 mm)
PLAY_MM = 8 * SQUARE_MM  # 200.0 playing area (board frame origin = its a1 corner)
BOARD_OUTER_MM = PLAY_MM + 2 * (BOARD_LINE_MM + BOARD_BORDER_MM)  # 248 whole board
EDGE_MM = BOARD_LINE_MM + BOARD_BORDER_MM  # 24 from the playing area to the board's outer edge
BOARD_THICKNESS_MM = 9.0  # ESTIMATE (listing: folded board 18 mm thick, so ~9 mm unfolded): playing surface height above the table; MEASURE
# ArUco markers (DICT_4X4_50), ids 0..3 printed MARKER_MM wide, taped outside the playing area.
# MARKER_CENTERS gives each marker's center in board mm. Default: diagonally out from each corner.
ARUCO_DICT = "DICT_4X4_50"
MARKER_MM = 30.0  # MEASURE: printed black square edge
_off = 25.0  # MEASURE: distance from the board corner to the marker center along x and y
_B = 8 * SQUARE_MM
MARKER_CENTERS = {
    0: (-_off, -_off),  # beside a1
    1: (_B + _off, -_off),  # beside h1
    2: (_B + _off, _B + _off),  # beside h8
    3: (-_off, _B + _off),  # beside a8
}

# ---------------------------------------------------------------- pieces
# height mm, base diameter mm, gripper % when holding it (0 closed .. 100 open).
# Heights/diameters of k, n, r were estimated from the third-person camera (2026-10-17); p, b, q follow
# standard proportions. Refine with a ruler.
# Grip: gripper % reading when the jaws are closed on a piece. Two groups (user, 2026-10-17): king and queen share
# one width, all other pieces another. Taught with calib/teach_grip.py -> data/grip.json; defaults until then.
GRIP_FILE = DATA / "grip.json"
try:
    import json as _json

    _g = _json.loads(GRIP_FILE.read_text())
except (OSError, ValueError):
    _g = {}
GRIP_SMALL = float(_g.get("small", 10.0))  # pawn, knight, bishop, rook
GRIP_BIG = float(_g.get("big", 12.0))  # king, queen
_per = {k: float(v) for k, v in _g.get("per_piece", {}).items()}  # learned per piece (overrides small/big)
# ---- 2026-11-05: TOP GRIP. The new pieces have ~19-20 mm bases on 25 mm squares: only ~5 mm between neighbours, less
# than one finger. So each piece is held by its HEAD (8-15 mm wide), where the fingers fit between the neighbours.
GRASP_MODE = "top"  # "top": the geometric grasp model below (shift / opening / held offset computed from the jaw geometry)
# WHERE on the piece the jaws close. "head": by the narrow head (fits between neighbours, tight sideways tolerance);
# "base": at the base of the piece like the old rig (wide opening, forgiving, but the 19.5 mm bases on 25 mm squares only
# leave ~5.5 mm between neighbours: needs the neighbours on both sides to be absent). 2026-11-05: head grip closed on air.
import os as _os

# WHERE on the piece the jaws close: "neck" = the narrowest 12 mm band of each piece (its stem: 4-8 mm wide, from the width
# profiles in data/piece_profiles.json; this is what the original PLAN.md meant: grasp at ~50 % of the height, "wide base
# and graspable neck"); "head" = by the head; "base" = at the base (old rig, needs 33 mm squares).
GRIP_ON = _os.environ.get("CHESS_GRIP_ON", "neck")  # switch without editing: $env:CHESS_GRIP_ON = "base"  (PowerShell)
HOLD_SQUEEZE_PCT = 1.2  # while carrying, command the jaws this much (%, ~2.4 mm) tighter than where the piece stops them
GRASP_DEBUG_PHOTOS = False  # the Ausha webcam moved off the gripper to be the fixed camera (EXT_CAMERA_URL); it's busy there, not available for debug photos
GRASP_DEBUG_CAM = 1 if _WIN else "/dev/v4l/by-id/usb-Ausha_Webcam_Ausha_Webcam-video-index0"  # unused while GRASP_DEBUG_PHOTOS is False
BASE_GRIP_Z_MM = 4.0  # tip height for a base grasp (like the old GRASP_HEIGHT_MM)
TOP_GRIP_CLEAR_MM = 8.0  # room each side of the grasped part when the jaws are open for the
                         # approach (tried first; the planner tightens to 3 / 2 / 1.2 mm only where a neighbour is in the way).
TOP_RELEASE_OPEN_MM = 2.5  # to let go, open this much beyond the head width
# Geometry from the gripper's 3-D model (URDF meshes, 2026-11-05): the tip frame sits ON the fixed finger's inner face, so
# the approach shift toward the fixed finger = head radius + the room left of the head, and once gripped the head is
# pushed against that face: its centre is one head radius from the tip. (The old base-grasp numbers, shift 14 / held 4,
# would drag the piece ~10 mm sideways: wrong for a head grip.)
TOP_SHIFT_EXTRA_MM = 2.5  # approach shift = head radius + this (the piece is pushed this far while closing)
TOP_HELD_REL_MM = 0.0  # held piece centre from the tip = head radius + this
TOP_ARC_MM = (22.0, 38.0)  # carry lift above the grasp point (min, max): the held piece already hangs high
# Gripper reading (%) <-> gap between the finger pads (mm) of THIS gripper, measured with a ruler (calib/gripper_gap.py).
# Until measured, a rough guess (100 % = 70 mm) keeps planning possible; the arm must NOT be run top-grip before measuring.
GAP_FILE = DATA / "gripper_max_open.json"
try:  # the widest opening a grasp approach may use (%), taught by hand: data/gripper_max_open.json
    GRIP_OPEN_MAX_PCT = float(_json.loads(GAP_FILE.read_text())["pct"])
except (OSError, ValueError, KeyError):
    GRIP_OPEN_MAX_PCT = 100.0  # no limit until taught
GAP_FILE = DATA / "gripper_gap.json"
try:
    _gp = _json.loads(GAP_FILE.read_text())
    _GAP_PCT, _GAP_MM, GAP_MEASURED = list(_gp["pct"]), list(_gp["gap_mm"]), True
except (OSError, ValueError, KeyError):
    _GAP_PCT, _GAP_MM, GAP_MEASURED = [0.0, 100.0], [0.0, 70.0], False


def gap_to_pct(mm):
    import numpy as _np

    return float(_np.interp(mm, _GAP_MM, _GAP_PCT))


def _top_piece(height, base_d, head_d, head_zone, below_top):
    """One piece for top grip. head_zone = how tall the narrow head part is (the part from `height - head_zone` up is
    only head_d wide); below_top = how far below the piece's top the finger tips go.
    With GRIP_ON = "base" the grasped part is the base itself: its width, at BASE_GRIP_Z_MM."""
    if GRIP_ON == "base":
        head_d, head_zone, below_top = base_d, height, height - BASE_GRIP_Z_MM
    return dict(
        height=height, diameter=base_d, head_d=head_d, head_from=height - head_zone, grip_z=height - below_top,
        grip=gap_to_pct(head_d),  # reading when the jaws hold the head
        open=gap_to_pct(head_d + 2 * TOP_GRIP_CLEAR_MM),  # approach opening
        release_pct=gap_to_pct(head_d + TOP_RELEASE_OPEN_MM) - gap_to_pct(head_d),
        shift=head_d / 2 + TOP_SHIFT_EXTRA_MM, held=head_d / 2 + TOP_HELD_REL_MM)


# Heights from the product listing photo (pawn 25 mm and king 50 mm are printed; the others measured from the
# picture against those two: pawn/king scales agree within 1 %). Base diameters and head widths measured on the real
# pieces from the phone's full-resolution photo (+-1 mm). head_zone / below_top are first guesses: tune on the arm.
PROFILE_FILE = DATA / "piece_profiles.json"  # width of each piece at every mm of height (calib/piece_profile.py)
try:
    _PROF = _json.loads(PROFILE_FILE.read_text())
except (OSError, ValueError):
    _PROF = {}


def max_width(piece, z0, z1=None):
    """Widest part (mm) of a piece between heights z0 and z1 above its base (z1 None = up to its top)."""
    w = PIECES[piece.lower()].get("profile")
    if not w:
        return PIECES[piece.lower()]["diameter"]
    i0 = max(0, int(z0))
    seg = w[i0:] if z1 is None else w[i0 : int(z1) + 1]
    return max(seg) if seg else 0.0


def _neck_piece(sym, height, base_d):
    """One piece for the neck grip: grasp height and contact width from its width profile."""
    p = _PROF[sym]
    gz, gw = float(p["grip_z"]), float(p["grip_w"])
    d = _top_piece(height, base_d, gw, 0.0, height - gz)  # (head_d = the widest part in the pad band)
    d.update(head_from=gz, profile=p["width_by_mm"])
    return d


PIECES = {
    "p": _top_piece(25.0, 19.0, 8.0, 10.0, 8.0),  # head grip: tip 17 mm up, pads then span the 17-25 mm ball
    "n": _top_piece(33.0, 19.0, 10.0, 12.0, 7.0),  # knight: the head is off-centre and long: test last
    "b": _top_piece(37.0, 19.5, 8.0, 12.0, 7.0),
    "r": _top_piece(29.5, 19.0, 10.0, 8.0, 5.0),
    "q": _top_piece(44.0, 19.5, 12.0, 14.0, 7.0),
    "k": _top_piece(50.0, 19.5, 15.0, 16.0, 7.0),
}
if GRIP_ON == "neck" and _PROF:
    PIECES = {s: _neck_piece(s, PIECES[s]["height"], PIECES[s]["diameter"]) for s in "pnbrqk"}
if GRASP_MODE == "base":  # the old grasp: readings taught with calib/teach_grip.py
    for _k, _v in PIECES.items():
        _v.update(grip=_per.get(_k, GRIP_BIG if _k in "qk" else GRIP_SMALL), grip_z=4.0, shift=14.0, held=4.0)
# Per-piece jaw opening taught by hand (data/grip_practice_log.json, written during the gripping practice): the gripper
# reading (%) with the piece between the jaws. The approach opens to exactly that reading, so the room each side of the
# piece = (gap - head width) / 2 (never below 1.2 mm). Pieces without an entry keep TOP_GRIP_CLEAR_MM.
GRIP_PRACTICE_LOG = DATA / "grip_practice_log.json"
USE_TAUGHT_OPENING = True
if USE_TAUGHT_OPENING and GRIP_PRACTICE_LOG.exists():
    try:
        import numpy as _np2

        _names = {"pawn": "p", "knight": "n", "bishop": "b", "rook": "r", "queen": "q", "king": "k"}
        for _e in _json.loads(GRIP_PRACTICE_LOG.read_text()):
            _s = _names.get(_e.get("piece"))
            if _s in PIECES:
                _gap = float(_np2.interp(_e["gripper_pct"], _GAP_PCT, _GAP_MM))
                PIECES[_s]["taught_open_pct"] = float(_e["gripper_pct"])
                PIECES[_s]["taught_clear"] = max(1.2, (_gap - PIECES[_s]["head_d"]) / 2)
    except (OSError, ValueError, KeyError):
        pass
# Grasp taught by hand on the real square (calib/teach_grasp.py -> data/taught_grasps.json): tip shift toward the fixed
# finger, sideways offset, tip height and jaw reading, per piece type. Overrides the numbers above for those pieces.
TAUGHT_GRASPS = DATA / "taught_grasps.json"
if TAUGHT_GRASPS.exists():
    try:
        import numpy as _np3

        for _s, _g in _json.loads(TAUGHT_GRASPS.read_text()).items():
            if _s in PIECES:
                _gap = float(_np3.interp(_g["gripper_pct"], _GAP_PCT, _GAP_MM))
                PIECES[_s].update(taught_shift=_g["shift_mm"], taught_lateral=_g["lateral_mm"], grip_z=_g["grip_z"],
                                  taught_open_pct=_g["gripper_pct"],
                                  taught_clear=max(2.0, (_gap - PIECES[_s]["head_d"]) / 2))
                if _g.get("held_pct") is not None:
                    PIECES[_s]["grip"] = float(_g["held_pct"])  # the reading when really holding it
                if _g.get("release_delta_pct") is not None:
                    PIECES[_s]["release_pct"] = float(_g["release_delta_pct"])  # how much further to open to let go
    except (OSError, ValueError, KeyError):
        pass
# The grasp used to hover centred over the piece and then step sideways (joint 1 swinging left) right next to it.
# True = hover high straight over the final grasp spot and only go straight down.
DIRECT_DESCENT = True
# From the teleop demonstrations (data/demos, analyze_demo; medians of the 5 clean demos, 2026-10-17):
GRASP_HEIGHT_MM = 4.0  # jaw tip this far above the surface when grasping: pieces grabbed at the bottom (demos 0-11)
JAW_SHIFT_MM = 14.0  # the LEFT finger is fixed: the tip goes this far toward it before closing, so that finger lands
                     # beside the piece (demos: 10-17 mm, median 14)
HELD_OFFSET_MM = 4.0  # once gripped, the moving finger has pushed the piece against the fixed one: its centre is
                      # this far from the tip (demos: release tip 1-5 mm from the target centre, median 4)
# Pick-up correction measured on the real arm (2026-10-18), for the PICK only (placing unchanged): at the grasp the
# tip goes this much further toward the robot's base (along the board), further toward the fixed finger (the
# "go left" step), and lower (vertical; 0 = the demonstrated grasp height).
# 2026-11-04: the trims below were tuned on the OLD rig (old calibration, board and pieces). Reset to 0 for the new
# rig: re-tune them from real test moves after the corner touch calibration (old values: toward-arm 4, left 4,
# ARM_TARGET_OFFSET (2, 2), PLACE_EXTRA (-12, 1)).
PICK_EXTRA_TOWARD_ARM_MM = 0.0
PICK_EXTRA_LEFT_MM = 0.0
PICK_EXTRA_DOWN_MM = 0.0  # (2026-11-05: the 5 mm that was tried here for the head grip was hiding a missing settle() at the bottom of the descent: now measured instead) first top-grip tests on a pawn: closed on air 3x at the nominal height (gripper stopped at 2.6 %, the closing command); 2 mm lower still too high (user, by eye) -> 5 mm lower at the grasp (all pieces; eased in on the descent, out on the lift)
# Whole-motion trim (board mm: +x = toward the h-file, +y = toward rank 8). 2026-10-18: picks and places landed
# ~2 mm toward the top right as seen from the human's seat (toward rank 1 and the a-file), so every arm target is
# moved 2 mm toward the h-file and 2 mm toward rank 8. The collision check is unaffected (it uses the real squares).
ARM_TARGET_OFFSET_MM = (0.0, 0.0)
# Place-only trim on top of that (same axes: +x = toward the h-file, +y = toward rank 8). 2026-10-18: pick perfect;
# the place goes 12 mm to the ROBOT's left (toward the a-file) and 1 mm away from the robot (toward rank 8).
# Eased in during the carry.
PLACE_EXTRA_OFFSET_MM = (0.0, 0.0)
GRIPPER_OPEN = gap_to_pct(24.0) if GRASP_MODE == "top" else 28.4  # % open for generic moves (escape after a drop): top grip = a 24 mm gap; base grip: 28.4 (22 before the 2026-10-18 gripper re-range, x1.2914; demos 20-32 %)
DROP_DETECT_PCT = 3.9  # while carrying: jaws closing this much past the holding reading = the piece slipped out
RECOVERY_ATTEMPTS = 2  # after a failed grab / drop: look with the webcam and retry this many times
RELEASE_OPEN_PCT = 10.3  # to let go, open only this much past the holding reading (keeps the moving finger off neighbours)
RELEASE_GAP_MM = 4.0  # how far the moving finger moves away from the piece when releasing (collision model)
GRIPPER_LOOK = 0.0  # % at the look/home pose: closed (user decision 2026-10-17)
# Grasp: command (grip - GRIP_SQUEEZE) and read where the jaws stop. `grip` above is the reading at contact.
GRIP_SQUEEZE = 5.2  # % past contact (LeRobot's gripper torque limit keeps the squeeze gentle)
GRIP_EMPTY_TOL = 1.9  # jaws stopped within this of the command = nothing in the hand
GRIP_TOO_WIDE = 12.0  # reading more than this above `grip` = misgrip (caught a neighbour, piece on its side)
if GRASP_MODE == "top":
    # This gripper moves ~2 mm of jaw gap per 1 % (data/gripper_gap.json), and a head is only 8-15 mm wide, so every
    # threshold above (written for the old rig's base grasp) is rescaled: squeeze 3 mm past contact, empty = within ~1.4 mm
    # of the command, misgrip = 5 mm wider than expected, slipped = jaws closing 2 mm past the holding reading.
    # 2026-11-05 first base-grip test: the jaws stopped at 8.3 % on the pawn (16 mm: it is narrower than the 19 mm base
    # at the grasp height), the squeeze of 1.5 % was inside the gripper's normal tracking error, so the check called a REAL
    # grasp "empty" and re-opened. A firm close command (4 % = ~8 mm past the expected width; the motor's 50 % torque limit
    # keeps the force safe) makes "empty" (closes all the way) unmistakable against "holding" (stopped by the piece).
    GRIP_SQUEEZE = 4.0
    GRIP_EMPTY_TOL = 1.5
    GRIP_TOO_WIDE = 4.0
    DROP_DETECT_PCT = 1.0
    RELEASE_FREE_LOAD_PCT = 6.0
    RELEASE_STEP_PCT = 1.0
GRASP_MAX_TILT = 55.0  # deg from vertical allowed at grasp/place height; far ranks need ~45-50 deg (teleop showed tilted grasps work)

# ---------------------------------------------------------------- z levels (mm above the board surface)
Z_SAFE = max(p["height"] for p in PIECES.values()) + 66.0  # 130 mm: reachable over every square (checked)  # max travel height (used when nothing better is known)
Z_PLACE_LIFT = 3.0  # release this far above the grasp height (demos: the tip was 9-20 mm up; 3 is gentler)
LIFT_MM = 10.0  # straight up after grasping, before the arc starts
# Carry shape copied from the demos: straight up, travel high with a gentle hump, straight down over the target.
ARC_PER_MM = 0.35  # lift = this x carry distance, clamped (demos: 95-120 mm above the grasp, median 116):
ARC_MIN_MM = 90.0  # even short moves lift clearly
ARC_MAX_MM = 110.0
ARC_RISE_FRACTION = 0.85  # vertical part of the lift; the rest is the hump in the middle of the travel
DETOUR_COST = 0.4  # mm of peak height one mm of extra path length is worth (bend around vs go over)
SIDE_STEP_CLEAR_MM = 20.0  # the sideways "go left" step happens this far above the piece's top (was 12: the arm measured ~10 mm lower than commanded)
APPROACH_SLOW = 2.5  # the down / sideways-step / down approach to a piece takes this much longer (the arm was pushing pieces)
# Tight-spot grasp, learned from the crowded teleop demos (2026-10-18: neighbours on the fixed-finger side, the
# moving-finger side, both sides; analyze_demo). Used only when the normal grasp would hit a neighbour. You opened
# less, shifted less, did the sideways step above the neighbours and opened only ~2 % to let go.
TIGHT_JAW_SHIFT_MM = (8.0, 6.0)  # demos: 6-10 mm toward the fixed finger, median 8
TIGHT_GRIPPER_OPEN = 24.5  # % (demos 22.7-26.2; normal GRIPPER_OPEN 28.4)
TIGHT_RELEASE_OPEN_PCT = 4.0  # % past the holding reading to let go in a tight spot. Tried 1.5 % (demos released at
                              # +1-1.7 %, but with room around the target): on the robot the piece lifted out with the
                              # fingers twice (2026-10-18). If even 4 % would touch a neighbour, the move is refused
                              # (Robo asks for help) instead of carrying the piece away.
RELEASE_FREE_LOAD_PCT = 8.0  # gripper load above this, or jaws short of the command = still pressing on the piece
RELEASE_STEP_PCT = 2.0  # ...then open this much more (up to the grasp opening) until they don't
JAW_MM_PER_PCT = 1.78  # jaw opening per gripper % (demos: pawn 19 mm held at 15.3 %, queen 23 mm at 17.2 %, empty 4.5 %)
TRAVEL_SPEED_MM_S = 120.0
TRAVEL_CLEAR_MM = 15.0  # carried piece's bottom clears the tallest piece near the path by this (travel height per move)
CORRIDOR_MARGIN_MM = 4.0  # extra width of the carried piece's path when looking for pieces to clear
# Gripper footprint for the collision check (MEASURE roughly with a ruler; generous by default). Along the jaw
# axis u (toward the moving finger): the fixed finger reaches FINGER_LEFT_MM to the left of the tip; the moving
# finger's outer face sits at the piece width + FINGER_WIDTH_MM to the right when open around a piece.
# 2026-11-05, measured on the URDF meshes (see tip-frame notes above): fixed finger 9-12 mm thick left of the tip, moving
# finger 11-14 mm thick, both ~14 mm deep across the jaw (old guesses: 7 / 7 / 12 mm)
FINGER_LEFT_MM = 10.0
FINGER_WIDTH_MM = 12.0
FINGER_HALF_DEPTH_MM = 7.0  # half the fingers' size across the jaw axis
FINGER_LENGTH_MM = 70.0  # fingers below the gripper body: tip is ~98 mm below the wrist roll axis, motor body ends ~25 mm below it
BODY_RADIUS_MM = 25.0
TOOL_RADIUS_MM = 18.0  # (legacy, unused)

# ---------------------------------------------------------------- off-board slots (board mm)
# Captured pieces go on the table just outside the board, one slot each, nearest the robot first (the robot sits
# behind rank 8 when it plays black). Spare queens for promotion sit in SPARES. Slots are (x, y) board mm;
# their z is -BOARD_THICKNESS_MM. Check them with calib/reachability_sweep.py.
_gx = EDGE_MM + 20.0  # slot centers 20 mm beyond the board's outer edge
# Slots run along both side edges; Slots() uses the ones nearest the robot first.
GRAVEYARD = [(_B + _gx, 14.0 + 24.0 * i) for i in range(8)] + [(-_gx, 14.0 + 24.0 * i) for i in range(8)]  # 24 mm pitch: the board is only 200 mm long
SPARES = {"Q": [(_B + _gx + 32.0, 60.0)], "q": [(-_gx - 32.0, _B - 60.0)]}

# ---------------------------------------------------------------- motion
CONTROL_HZ = 30
MAX_JOINT_SPEED = 60.0  # deg/s cap; durations are stretched to respect it
MIN_MOVE_S = 0.6
DESCEND_S = 1.2  # vertical moves near pieces: slow
GRIP_S = 0.8
DWELL_S = 0.3
# Overall arm speed (1.0 = the speeds above). Scales every move: joint speed cap, carry speed, descents, gripper.
CARRY_UP_S = 2.4  # seconds to lift a gripped piece straight up (was 0.6, then 1.4: still too sharp)
CARRY_DOWN_S = 3.5  # seconds to lower it onto its square (was 0.8: too sharp, the piece was placed badly)
CARRY_TRAVEL_SLOW = 1.6  # the carry across the board takes this much longer than the nominal speed says
ARM_SPEED = 0.6  # was 0.8 on the old rig; slower while the new grasp is being tuned
MAX_JOINT_SPEED *= ARM_SPEED
TRAVEL_SPEED_MM_S *= ARM_SPEED
MIN_MOVE_S /= ARM_SPEED
DESCEND_S /= ARM_SPEED
GRIP_S /= ARM_SPEED
# 2026-10-18 the follower was re-calibrated to match the leader in teleop (src/tools/match_follower_to_leader.py):
# wrist roll reads +0.879 deg for the same physical angle, gripper % x1.2914 for the same jaw position. The values
# here are converted so the arm moves physically exactly as before.
WRIST_ROLL_ZERO_SHIFT_DEG = 0.0  # (old rig: 0.879 after matching the follower to the leader; 2026-11-04 the follower was freshly calibrated here)
WRIST_ROLL_DEG = -96.9  # 2026-11-04 new calibration: the roll held during the corner touches (median -96.9); (old rig: -86.9) # roll used for the corner touches (calibration) and grasps: jaws side by side, so a tilted
                        # gripper keeps both fingers level (at roll 0 the lower finger hit the far squares)
USE_SAG_MODEL = False  # gravity sag model (calib/sag_calibration.py) made far squares worse; use Run B's method:
                       # measure at the hover point (settle) and carry that correction down
SAG_ITERS = 3  # settle(): correction rounds after arriving at a hover / grasp point
SAG_WAIT_S = 0.25
SAG_DEADBAND_DEG = 0.3
SAG_MAX_DEG = 10.0  # never correct a joint by more than this
IK_MAX_ERR_MM = 4.0  # reject IK solutions worse than this
JOINT_MARGIN_DEG = 3.0  # stay this far inside the calibrated range

# ---------------------------------------------------------------- voice + vision-language model ("Robo")
VLM_ENABLED = True
VOICE_COMMENTARY = True  # Robo announcing game events (moves, "your move", retries...); the status bar shows them anyway
VOICE_QUIPS = False  # language-model comments after every move (chatty); answers to questions stay on
VOICE_ANSWER_SENTENCES = 1  # spoken answers are cut to this many sentences
VLM_MODEL = "Qwen/Qwen3.5-2B"  # fits the 8 GB GPU next to YOLO; "Qwen/Qwen3.5-4B" needs VLM_4BIT = True
VLM_4BIT = False
VLM_IMAGE_WIDTH = 448  # camera frames are shrunk to this width before the VLM sees them (speed)
WHISPER_MODEL = "openai/whisper-small"
VOICE_LANGUAGE = "en"
PIPER_VOICE = DATA / "models" / "piper" / "en_US-ryan-medium.onnx"
VOICE_VOLUME = 1.0  # starting values; the window's Robo panel changes them live
VOICE_SPEED = 1.0
VOICE_MIC_DEVICE = None  # arecord -D device; None = system default (PipeWire)
VOICE_WAKE_WORDS = ["robot", "robo", "roboto", "robbo", "robert"]  # Whisper sometimes hears "Robert"
VOICE_MIC_ON_AT_START = False  # mic starts switched off; turn it on with the Microphone switch in the window
VOICE_REQUIRE_WAKE = True  # spoken commands/questions must contain a wake word (typed ones never need it)
VOICE_SNR = 5.0  # speech = this many times the room's noise floor (5 = ignores more background talk)
VOICE_MIN_RMS = 600.0
VOICE_END_SILENCE_S = 0.8
VOICE_MIN_UTTERANCE_S = 0.5
VOICE_MAX_UTTERANCE_S = 12.0

# ---------------------------------------------------------------- game
ENGINE_TIME_S = 1.0
ROBOT_COLOR = "black"  # fallback only: once the webcam corners are clicked, the robot's colour is read from them
                       # (vision/ext_camera.robot_color_from_corners): rank 1 on the robot's side = robot white
