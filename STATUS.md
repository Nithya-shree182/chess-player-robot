# STATUS: SO-101 chess robot on the NEW rig (Windows laptop)

Last updated: **2026-11-06** (vision session on Ubuntu). Earlier sessions: [docs/SESSION_LOG_2026-11-05.md](docs/SESSION_LOG_2026-11-05.md) (Windows setup).

## Update 2026-11-06 (evening): picking works on an empty board for all six piece types; crowded squares are the open problem
See the section "Grasping, later the same day" at the end of [docs/SESSION_LOG_2026-11-06.md](docs/SESSION_LOG_2026-11-06.md): the working recipe, the tools, the lessons and what is NOT solved.

## Update 2026-11-06 (morning): vision is done for now
Full map of the session: [docs/SESSION_LOG_2026-11-06.md](docs/SESSION_LOG_2026-11-06.md). Short version:
- The fixed camera is the Ausha webcam looking from the human's side, raised and tilted down; the grid for it is in `data/ext_corners.json` (locked copy
  `data/ext_corners.locked-oblique.json`). Camera brightness 95 (default 128 blows out the white pieces). Refit the grid on an EMPTY board whenever the camera moves.
- **YOLO is trained on our pieces**: 80 labelled images of 12 positions in `data/dataset/` (pushed to GitHub so it can be retrained), model `data/yolo/best.pt`
  (not in git: retrain with `python -m chess_player.vision.train_yolo --epochs 60`, ~100 s). Held-out mAP50 0.995; fresh positions read 64/64 several times.
  Weak spot: a piece hidden right behind another (queen behind a pawn).
- New tools: `vision/capture_safe.py` (shots with a drift check), `vision/capture_chain.py` (build positions by telling it the moves you made).
- Pieces must be set up in the STANDARD arrangement (queen on d1/d8). The dataset was first captured with king and queen swapped; labels were fixed.
- Still open: picking pieces up (below), `board_to_robot.json` not re-verified after this session, full pipeline test with the new model.

## One-paragraph summary
The project (originally built for Ubuntu on a 33.3 mm board with other pieces) now runs on this Windows laptop with a **new
25 mm board, new pieces and a phone as the fixed camera**. Arms, board calibration, cameras, Stockfish, YOLO loading, the
UI pipeline (dry-run) and the "zoom" (full-resolution phone stills) all work. **The one thing that does not work yet is picking
a piece up reliably**: the last attempts closed the jaws on air because the piece was not exactly where the jaws were aimed.
The next idea (from the user): make the robot look (gripper camera + phone) and correct itself before closing.

## How to resume
```powershell
$env:PYTHONPATH = "D:\ntt data nithya\chess\src"
# python = C:\Users\asus\lerobot-env\Scripts\python.exe   (venv, CUDA torch 2.11+cu128, LeRobot 0.6.1, ultralytics 8.4.202)
```
First check the arm is connected: `python src\tools\read_pose.py` (at the end of session 1 the follower USB board was NOT detected:
COM5 missing, only the leader COM6; plug the follower in / switch its power on). If the arm is stuck in an awkward pose:
`python src\tools\safe_lift.py`. Torque on/off: `python src\tools\torque.py gripper|all|on`.

Pipeline (UI, no voice): `scripts\run_pipeline.ps1 [--dry-run]` then http://localhost:8770/ (stop other arm/camera programs first).

## Hardware and ports
| Thing | Details |
|---|---|
| Follower arm | CH343 board, serial `58FA082690`, **COM5**, LeRobot id `my_follower` (calibrated 2026-11-04 20:07) |
| Leader arm | CH343 board, serial `5AE6082864`, **COM6**, id `my_leader` (calibrated 20:12). The leader only worked after changing to a data USB cable |
| Calibration files | `%USERPROFILE%\.cache\huggingface\lerobot\calibration\...` (copies of the old rig's are in `data/calibration_backup/`) |
| Camera on the gripper | "Ausha Webcam", DirectShow index **1** (index 0 = laptop camera, black). Used for debug photos (`data/grasp_debug/`), not for the board |
| Fixed camera | Phone, IP Webcam app: stream `http://10.4.246.27:8050/video` (1280x720), stills `/photo.jpg` (set to 4000x2256). **The address changes with the network**: edit `EXT_CAMERA_URL` in `config.py` |
| GPU | RTX 3050 Laptop 4 GB (original rig: RTX 5050 8 GB). Voice/Qwen are OFF (start the server with `--voice` to enable) |
| Stockfish | `bin\stockfish.exe` (copied from `D:\chess-player\engines\stockfish_dl\stockfish\`) |
| YOLO | public pretrained chess model `data/models/pretrained_chess/...` (own fine-tuned `data/yolo/best.pt` NOT trained yet) |

## Board and pieces (measured)
- Board: Toy Imagine magnetic folding board, **squares 25.0 mm**, playing area 200 mm, outer 248 mm. Folds in the middle (seam between ranks 4|5).
- Orientation: rank 1 nearest the robot, **a1 dark at the near-left**; robot plays **white**. Light plastic: **tape the board down**; any movement voids the calibrations.
- Pieces (heights from the listing photo, pawn 25 and king 50 printed): P 25, R 29.5, N 33, B 37, Q 44, K 50 mm.
  Bases about 19-20 mm wide (so only ~5.5 mm between neighbouring bases). Stems/necks are thin: pawn 4-5 mm, rook 5-7, bishop 5, queen 6-7, king 6-7, knight ~10 mm.
  Width profiles per piece: `data/piece_profiles.json` (from the listing silhouette; regenerate with `calib/piece_profile.py`).
- Gripper (from the URDF meshes): the **tip frame sits on the fixed finger's inner face**; fixed finger ~10 mm thick (left of the tip), moving finger ~12 mm thick, both ~14 mm deep.
  Jaw gap vs reading (measured, `data/gripper_gap.json`): about **1.92 mm per 1 %** (5 % = 10 mm, 10 % = 20 mm, ...).
- A rubber band was added to the finger pads (2026-11-05 ~01:50).

## Calibration status
| What | File | State |
|---|---|---|
| Follower/leader arm calibration | LeRobot cache | OK |
| **Board <-> robot** | `data/board_to_robot.json` | **OK, verified**: 17-point fit (`board_to_robot --grid --ring`, RMS 9.2 mm of hand-touch noise); `hover_test` lands on a8, h8, a1, h1, d4 correctly. Backup: `board_to_robot.try3-17pt-varied-roll.json`. Do not touch unless the board or robot base moves |
| Home / look pose | `data/look_pose.json`, `data/poses/home_pose.json` | OK (arm over the left edge, out of the phone's view) |
| Phone corners, stream | `data/ext_corners.json` | **STALE**: the phone was moved closer at ~02:00. Refit on the EMPTY board: `python -m chess_player.calib.fit_phone_corners --rough "a1x,y h1x,y h8x,y a8x,y"` (rough points by eye in the 1280x720 stream; it snaps to the squares and refuses to save a bad fit) |
| Phone corners, 4000x2256 still | `data/ext_corners_hires.json` | **STALE** (same reason; the same command refits both) |
| Jaw gap table | `data/gripper_gap.json` | OK |
| Old-rig files | `*.old-rig.json`, `grip.json`, `sag_model.json` | **Not valid for this rig** (grip.json/old trims are ignored in the new grasp model) |

## What works
- Arms + LeRobot on Windows; teleop (leader -> follower) works; board calibration; hover to any square (±2 mm after the sag correction).
- Replaced `placo` (no Windows build) by a pure-Python/SciPy kinematics in `motion/ik.py` (FK identical to the old results; IK round-trips).
- Pipeline in dry-run (UI at :8770, Stockfish, YOLO on the GPU), robot colour from the board calibration, voice off by default.
- Zoom: when a square's reading is unsure (< 0.6 confidence) it is re-read on a 4000x2256 phone still (`vision/zoom.py`, background thread). Not yet useful on real games because YOLO is untrained for these pieces (52/64 squares right at the start position on the stream, 55/64 on the still).
- Planner for the new geometry (`GRASP_MODE="top"` model): neck grip plans **all 16 pawn moves in the full start row**; only the knight moves from b1/g1 are blocked.
- Tools: `calib/board_to_robot.py --grid --ring`, `calib/hover_test.py`, `calib/gripper_gap.py`, `calib/piece_profile.py`, `teach/record_poses.py` (teleop recorder, untested on real arms), `tools/torque.py`, `tools/safe_lift.py`, `tools/read_pose.py`.

## What does NOT work: picking up a piece (the open problem)
Attempts on a pawn on e2 (all 2026-11-05 ~01:00-01:45), what happened and what we learned:
1. Head grip (jaws at the pawn's ball, tight 2.5 mm room): closed on nothing 3x (gripper stopped at 2.6 % = the closing command).
2. 2 mm and 5 mm lower: same; "touched it" but no grasp.
3. Base grip (old method), neighbours removed: **the 3rd run grasped and placed the pawn** (hold at 8.3 % = 16 mm). Two bugs were found on the way and fixed: (a) the pick never ran the sag correction (`settle`) at the bottom of the descent, so jaws closed 5-10 mm too high; (b) the "empty" check was inside the gripper's tracking error, so real grasps were called empty and re-opened (squeeze now 4 %, empty tolerance 1.5 %).
4. Slower carry (lift 2.4 s, lower 3.5 s, arm speed 0.6) fixed the placement; a firm hold squeeze while carrying was added.
5. Neck grip (pad band at the stem, z = 7 mm for a pawn, jaws open ~14 mm): **closed on air (0.9 %)**. The gripper-camera photos (`data/grasp_debug/014454_*`) show no pawn between the fingers, and a full-resolution phone photo showed the **pawn standing ~4-11 mm off the centre of e2**, toward the f-file. The neck grip only leaves ~3.5 mm each side in a full row, so a hand-placed piece is missed.
6. Fix tried: wider opening when nothing is near (up to 8 mm each side) and a way to place the piece exactly (`hover_test --height 30 --squares e2 --hold 25` holds the tip over the square centre).
7. **Later runs (01:56-02:11) reached the "holding" stage at least four times** (debug photos `data/grasp_debug/015631_P_holding.jpg`, `015903_`, `021023_`, `021106_`; the code only logs "holding" after a verified grasp), so the neck grip did grasp the pawn in several runs. What happened after that (lift, carry, placement) was not reported before the session was stopped: ask the user / look at the terminal history. **The debug photos from ~01:56 on are black**: the Windows camera index of the gripper webcam changed (`GRASP_DEBUG_CAM = 1` no longer pointed at it). Check with `python -c "import cv2; ..."` or the index probe in `docs/SESSION_LOG` before relying on them.

Geometry facts that matter: base grip cannot work in a full row (5.5 mm between bases, 10 mm thick fixed finger); head/neck grip can, but needs the jaws positioned to ~±3 mm of the stem.

## Next steps (suggested order)
1. Plug the follower back in, `read_pose.py`, `safe_lift.py` if needed.
2. **The user's idea, make the pick closed-loop (look and correct):** before closing, take the gripper-camera image (and/or the phone's 4000x2256 still) and locate the piece relative to the fingers; shift the jaws sideways by the measured error, then close; if the reading says "empty" do not retry blindly but re-measure and correct. Pieces to build: (a) a piece locator on the phone still (YOLO box or a simple blob of the piece's stem near the target square, corrected for parallax, ±2 mm goal); (b) a gripper-camera check at the grasp pose (is the stem between the two fingers?) using the saved debug photos to design it; (c) apply the correction to `Transfer.src` (and to the placed square's expected position) in `pick_place._plan_transfer`.
3. Refit the phone corners (stream + still) on the empty board (and re-run `python -m chess_player.vision.zoom` to check).
4. Test the lone pawn (piece centred with the hover tool), then the full row, then each piece type (P, B, Q, K, R, N).
5. YOLO: capture a dataset with these pieces from the phone view in varied lighting (`vision/capture_dataset.py`, needs the start position and corners), fine-tune (try YOLO11/26 vs YOLOv8m), compare on a held-out session.
6. Full pipeline with the real arm; offline model downloads (Whisper/Qwen are off); venue checklist.
7. Knight grips (b1/g1), graveyard slots, castling, promotion; park pose.

## Config knobs (`src/chess_player/config.py`)
- `GRASP_MODE = "top"` (the geometric grasp model); **`GRIP_ON`**: `"neck"` (default), `"head"`, `"base"`. Switch without editing: `$env:CHESS_GRIP_ON = "base"`.
- `TOP_GRIP_CLEAR_MM = 8.0` (first-try room each side; planner tightens in crowds), `HOLD_SQUEEZE_PCT`, `GRIP_SQUEEZE/EMPTY_TOL/TOO_WIDE/DROP_DETECT_PCT` (rescaled for 2 mm per %), `CARRY_UP_S/CARRY_DOWN_S/CARRY_TRAVEL_SLOW`, `ARM_SPEED = 0.6`, `PICK_EXTRA_DOWN_MM = 0`.
- `GRASP_DEBUG_PHOTOS = True` saves gripper-camera photos before/after closing to `data/grasp_debug/`.
- Single move test: `python -m chess_player.move_piece e2 e4 --fen "8/8/8/8/8/8/4P3/8 w - - 0 1"` (lone pawn) or `--fen start`; add `--dry-run` to only plan.

## Uncommitted work
Everything above is **uncommitted** in this git checkout (see `git status`). Nothing was pushed.
