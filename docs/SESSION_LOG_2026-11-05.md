# Session log: setting the project up on the new rig (Windows), 2026-11-04 evening to 2026-11-05 02:20

Times are local (laptop clock). Those marked `~` are approximate (read off file times and tool output, not logged exactly).
Companion file: [../STATUS.md](../STATUS.md) (current state and next steps).

## Timeline
| Time | What happened |
|---|---|
| 11-04 ~19:30 | Goal: use the SO-101 (leader + follower) on this laptop. Follower found on COM5 (CH343 serial 58FA082690). Leader not detected: the USB cable was charge-only; a new cable gave COM6 (5AE6082864). Webcam "Ausha" + laptop camera present. |
| 19:40-20:12 | LeRobot 0.6.1 installed into a new venv `C:\Users\asus\lerobot-env` (Python 3.13). Arms calibrated here: follower `my_follower` (20:07), leader `my_leader` (20:12). Teleop leader -> follower verified. |
| ~20:20 | Repo `Nithya-shree182/chess-player-robot` cloned into `D:\ntt data nithya\chess` (commit `7da0b97`), submodule `external/chess-player` initialised. Reviewed README/commands/config: built for Ubuntu (bash, conda, `/dev` paths, V4L2, `placo`). New board (25 mm squares), new pieces, new location. |
| ~21:15 | Decision: stay on Windows (no dual boot). Needed: CUDA torch (the venv had CPU torch), a `placo` replacement, Stockfish. Stockfish found at `D:\chess-player\engines\stockfish_dl\stockfish\` and copied to `bin\stockfish.exe`. |
| 21:38-22:57 | CUDA PyTorch (cu128) download at ~1 MB/s; import errors while it was finishing; at the end `torch 2.11.0+cu128`, CUDA available on the RTX 3050 4 GB. Ultralytics installed with `--no-deps` (versions kept: numpy 2.2.6, opencv-headless 4.13). |
| ~21:50 | `motion/ik.py` rewritten without `placo` (own URDF forward kinematics + SciPy least squares). Checked: FK reproduces the old board fit exactly (4.93 mm, same R and t); IK round trips. |
| ~22:00 | Gripper-camera view saved as home pose (`look_pose.json`, `home_pose.json`, backups `*.old-rig.json`). Config ported (ports, ids, camera index, Stockfish path, Windows). |
| 22:22 | First board-to-robot (4 corners + 2 centres): RMS 15.6 mm: poor (touched corners inconsistently; a1 touched at its centre). Added `--grid` (9 square-intersection points) to the tool. |
| ~22:30-23:30 | Grid fits 7.9 mm then 4.3 mm; hover test reached all corners except a8 (landed on a7/b8). Found the arm stops 2-4 degrees short at full stretch (gravity, weak P gain); `settle()` after the descent fixed that in `hover_test`. Board got pushed once by the arm (hover at 8 mm); rule: never hover below 12 mm; board must be taped. |
| ~22:46-23:55 | Built and abandoned three "nudge" calibrations (typed nudges, click-the-tip in the overhead camera, photo-reading): the overhead camera cannot see the jaw tip (hidden under the gripper). Files kept but not used: `calib/nudge_calibrate.py`, `nudge_visual.py`, `hover_snap.py`; `geometry.BoardToRobot` still supports an optional 2-D correction (`data/board_correction.json`, none saved). |
| 00:05 | **17-point touch calibration** (`board_to_robot --grid --ring`, RMS 9.2 mm). Hover test on a8, h8, a1, h1, d4: all correct (user, by eye). Board-to-robot DONE. (Roll-lock option added to the touch tool afterwards; not needed.) |
| ~23:50-00:20 | Phone (IP Webcam) as fixed camera: stream works (25-29 fps); `ext_corners.json` fitted automatically on the empty board (35/35 inner corners); robot colour now taken from the board calibration (camera independent); gripper-camera role disabled; voice off by default; `scripts/run_pipeline.ps1`. |
| 00:21 | Pipeline starts in dry-run ("robot white, vision yolo, arm simulated"). |
| ~00:25-00:45 | **Zoom**: `vision/zoom.py` re-reads unsure squares (< 0.6) on a 4000x2256 phone still (photo size set via the app API), own corners `ext_corners_hires.json`, hooked into the server's `_fused()`. YOLO on the still: 0.15 s on the GPU, 0.5 GB. Pretrained YOLO downloaded: start position read 52/64 (stream), 55/64 (still), 54/64 merged. |
| ~00:50-01:15 | New pieces placed. Measured: bases ~19-20 mm, heads 8-15 mm; heights from the product photo (pawn 25, king 50 printed; others by scaling: R 29.5, N 33, B 37, Q 44). Planner and config reworked for a "geometric grasp model" (see below). Found and fixed a hidden crash (`_jaw_shifted` still called `placo`). |
| ~01:15 | Gripper gap measured with a ruler (`calib/gripper_gap.py`): ~1.92 mm per 1 %. |
| ~01:20-01:50 | Grasp tests with `move_piece e2 e4` on a pawn: head grip (3 runs closed on air), base grip with neighbours removed (bugs: no `settle` at the bottom; "empty" check inside the tracking error; re-opened real grasps), then a **successful grasp and place** (base grip, ~01:35). Added: firm close command, hold squeeze, slower lift/carry/lower, measured-position logging, debug photos from the gripper camera. |
| 01:28 | `teach/record_poses.py` (teleop recorder in the original demo format) written and tested on a synthetic demo; the user found teleoperating the grasp too hard, so it was not used. |
| ~01:55 | Read the original `PLAN.md`: it grasped at 50-60 % of the piece height ("wide base and graspable neck"). Width profiles measured from the listing silhouette (`calib/piece_profile.py`, `data/piece_profiles.json`): thin stems 4-8 mm. New default grip = **neck** (`GRIP_ON="neck"`); planner plans all 16 pawn moves in the full start row. |
| 01:44-01:45 | Neck-grip run: 3 attempts closed on air (0.9 %). Gripper-camera photos saved; phone photo shows the pawn ~4-11 mm off the centre of e2. Fix applied (wider opening when nothing is near; `hover_test --hold` to place a piece under the tip) but **not run**. |
| 01:56-02:11 | More neck-grip runs by the user (outcomes not reported to me). The debug photos show the code reached "holding" at 01:56:31, 01:59:03, 02:10:23 and 02:11:06, i.e. verified grasps happened; the photos from this period are black (the gripper webcam's DirectShow index changed). |
| ~02:00 | Phone moved closer: stored camera corners are stale. |
| 02:14 | User asked to stop everything and write notes. No processes running. Follower USB (COM5) not detected afterwards (unplugged / powered off). |

## Code changes (all uncommitted)
| File | Change |
|---|---|
| `src/chess_player/config.py` | Windows ports/ids/camera/Stockfish; phone URL; board 25 mm + new piece table; grasp model (`GRASP_MODE`, `GRIP_ON`, gap table, per-piece grip height/open/hold/shift/held, thresholds rescaled for 2 mm per %); zoom settings; carry timings; `ARM_SPEED 0.6`; old-rig trims reset; `CAMERA_ON_ROBOT_SIDE`; finger sizes from the URDF meshes; `max_width()` and profile loading |
| `motion/ik.py` | Removed `placo`; own FK + SciPy IK; `jaw_open_direction()`; `fk_to()` |
| `motion/pick_place.py` | `settle()` at the bottom of the descent and before release; collision model with profile-based neighbour widths, head/neck grasp geometry, per-attempt room (8 -> 1.2 mm); firm hold squeeze; slow first step after the grasp; measured jaw-position logging; debug photos; `_jaw_shifted` without placo; travel height/arc for a high-held piece |
| `geometry.py` | Optional 2-D correction (`board_correction.json`) in `BoardToRobot` |
| `calib/board_to_robot.py` | `--grid`, `--ring`, tilt readout, camera-agnostic corner descriptions, roll lock while touching |
| `calib/hover_test.py` | settle after descent, measured vs commanded printout, long hold to place a piece |
| `vision/` | `camera.py` (DirectShow/Windows capture), `ext_camera.py` (index/URL sources, robot colour from the calibration), `zoom.py` (new), `train_yolo.py` (copies instead of symlinks) |
| `ui/server.py` | gripper camera role off by default, voice opt-in (`--voice`), zoom hook |
| `teach/` | `record_poses.py` (new), `analyze_demo.py` (holding thresholds for small heads) |
| new tools | `calib/gripper_gap.py`, `calib/piece_profile.py`, `calib/fit_phone_corners.py`, `calib/nudge_*.py`, `calib/hover_snap.py`, `tools/torque.py`, `tools/safe_lift.py`, `tools/read_pose.py`, `scripts/run_pipeline.ps1` |
| data | new `board_to_robot.json`, `look_pose.json`, `home_pose.json`, `ext_corners*.json`, `gripper_gap.json`, `piece_profiles.json`, `listing_pieces.png`, backups of the old/earlier files |

## Lessons (so we do not repeat them)
1. **Check what the camera can actually see before building a tool on it**: the overhead phone cannot see the jaw tip; three tools were wasted on that.
2. **Report measured, not commanded, positions**: `arm.joints()` returns the last command; use `arm.present()`.
3. **The arm sags 2-4 degrees at full reach**: always run `settle()` where accuracy matters (bottom of the descent, before release).
4. **Hand-touching a calibration is noisy (±8 mm)**: average many points (17 worked); never hover below 12 mm; tape the board.
5. **Gripper thresholds must be scaled to the gripper**: ~2 mm of gap per 1 %; a 1.5 % squeeze was inside the tracking error.
6. **A piece placed by hand is off by several mm**: tight grasps need the piece position measured (vision) or an opening wide enough; the planner tightens the opening only where neighbours force it.
7. Windows: `placo` has no wheel (replaced); DirectShow camera indexes can change; the phone address changes with the network; serial ports are exclusive (close the pipeline before arm scripts).

## Ideas / decisions pending
- Closed-loop pick (the user's idea at the end of the session): see STATUS.md "Next steps" item 2.
- Whether to keep the base grip as the fallback when a square has no neighbours (it worked once on a lone pawn).
- YOLO model choice (YOLOv8m vs newer) after collecting data from the phone view.
- Commit/push: nothing committed yet.
