# Session log 2026-11-06: fixed camera, YOLO dataset and a fine-tuned model (Ubuntu)

Companion files: [../STATUS.md](../STATUS.md) (current state), [GRID_ROBUSTNESS.md](GRID_ROBUSTNESS.md) (idea for a self-locking grid).
Times are not logged; the order below is the order things happened.

## Where things stand (one paragraph)
The fixed camera (Ausha webcam, 1280x720) now looks at the board from the human's side, raised and tilted down. The board grid
for that view is fitted and saved (`data/ext_corners.json`). 80 labelled images of 12 positions were captured with the new pieces and
a YOLO model was fine-tuned on them: on held-out positions mAP50 = 0.965, and it read 4 fresh positions in a row with every piece
right. **Not done:** picking pieces up (see STATUS.md), re-checking `board_to_robot.json` after this session's moves, and testing the
full pipeline with the new model.

## Camera set-up
- Camera: `/dev/v4l/by-id/usb-Ausha_Webcam_Ausha_Webcam-video-index0`, only `brightness` and `power_line_frequency` are controllable
  (no exposure / white balance). **Brightness is set to 95** (default 128 blew out the white pieces under the strong yellow light:
  ~8 % of pixels clipped vs 0.07 %). It resets when the camera is unplugged: `v4l2-ctl -d <dev> --set-ctrl brightness=95`.
- View: human's side, rank 1 (robot's side, white) at the TOP of the picture, file h on the left, a on the right, a1 dark at the top
  right. Robot colour reads "white".
- Grid: `data/ext_corners.json` = a1, h1, h8, a8 in image pixels. `data/ext_corners.locked-oblique.json` is a read-only copy,
  `data/ext_ref_frame.jpg` is the reference frame the drift check compares against. Older fits are kept as
  `data/ext_corners.before-*.json`. The camera moved or was bumped about six times; each time the grid was refitted.
- How a grid is fitted: board EMPTY, then rough corners by eye into `E.refine_corners` (snaps to the 35 inner corners). A fit with
  pieces on the board is only good to ~15 px: do not trust it. For a small camera shift the grid can instead be translated by the
  ECC shift of the piece-free board edges (`capture_safe.grid_drift`).
- A viewer window with the live feed and the locked grid drawn on it, which also re-serves the stream at
  `http://127.0.0.1:8771/video` (and `/shot` for one frame) so the capture tools can read the camera while the window is open, was a
  scratch script (not in the repo): a few lines of OpenCV + matplotlib/Tk. The venv's OpenCV is headless, so `cv2.imshow` does not work.

## Dataset (`data/dataset/`)
- 12 positions, 80 images (images/, labels/ YOLO txt, previews/ with boxes drawn, meta/ with the FEN and brightness):
  `pos01` (start, 13 images), `pos02` (game position), `pos10` (start again, new view), `pos11` (e4 e5), `pos13` (to move 3),
  `pos14`, `pos15`, `pos16` (opening moves), `pos17` (both castled), `pos18` (a capture), `pos19` (the trade), `pos20` (queens behind pawns), `pos21` (pawn behind a queen on both sides).
- Every shot is labelled from a known FEN: each box is the piece's 3-D cylinder projected through the camera pose that
  `BoardView` solves from the grid (focal length estimated from the board's homography).
- The old-rig dataset (33 mm board, other pieces) was moved to `data/dataset_old_rig/` (not committed; it is in git history).
- **King/queen swap, fixed.** The pieces were physically set with the king on the d-file and the queen on the e-file for both colours
  (non-standard), while the labels assumed the standard layout. All K/Q/k/q class ids were swapped in the 52 labels existing at that time
  (`data/dataset_labels_before_kq_swap/` and `data/dataset_meta_before_kq_swap/` hold the originals). Later captures use the standard
  arrangement (queen on d1/d8). Check with the preview images if anything looks off.
  **Exception: `pos02`**: white was set king-on-d1 but black was already standard, so the blanket swap broke its black labels. Found because the
  validation confusion matrix showed black king/queen mixed 44 %; fixed by restoring its original labels and swapping only the white K/Q
  (`meta.kq_swapped = "white only"`). Lesson: always look at the validation confusion matrix, one mislabelled position shows up there.
- Two positions were relabelled after a placement error: `pos13` (bishop was one square off, recaptured) and `pos20` (a pawn was on d5, not d6).
- Held-out split is by position name (`meta.position`), so one position is never in both train and val.

## Capture tools (new, `src/chess_player/vision/`)
- `capture_safe.py`: countdown (hands away), N shots with the camera brightness cycled (default 60,95,128,170,205 around the base 95), optional
  software contrast/gamma/tint jitter, and a **drift check** per shot (ECC translation of the piece-free board edges against
  `data/ext_ref_frame.jpg`; a shot is rejected above 2.5 px or a match below 0.7). Exits non-zero with 0 shots kept.
- `capture_chain.py`: you make moves on the board and tell it (SAN like `Nf3`, UCI like `e2e5`, `xe5` to remove a piece, illegal moves are fine);
  it keeps the board in `data/dataset_chain_state.json`, applies the moves and captures. This is what made it possible to build
  positions in ~20 s each instead of re-setting a full game.
  Example: `CHESS_CAMERA_URL=http://127.0.0.1:8771/video python -m chess_player.vision.capture_chain "e4 e5" --shots 5`
- `grid_lock.py`: a prototype (not wired in) for keeping the grid locked with markers / features, see GRID_ROBUSTNESS.md.

## Model
- Fine-tuned from the public pretrained chess model: `python -m chess_player.vision.train_yolo --epochs 60 --batch 8`. About 100 s on the RTX 4090 Laptop
  (16 GB). Output: `data/yolo/best.pt` (gitignored; the detector prefers it over the pretrained model).
- Final run (12 positions, 80 images): trained on 10 positions, validated on `pos02`, `pos18`, `pos19` (18 images, 553 pieces): precision 0.996, recall 0.997,
  mAP50 0.995, mAP50-95 0.937. (An earlier run with the wrong `pos02` black labels scored 0.923 / 0.924 / 0.961 / 0.912.)
- Fresh-position checks before training on them (score = squares read right out of 64): 61/64 (first model), then 64/64 on the standard
  start-up arrangement, 64/64 castled, 64/64 after a capture, 64/64 after the trade. The pawn-behind-queen test was partly a miss (a hidden queen).
- `config.KQ_TRUST_CONF` lowered 0.85 -> 0.5: the box-height guess for king vs queen used to overrule a correct "K 0.80" from the fine-tuned model.
- Earlier weights saved locally (not in git) as `data/yolo/best.v1.pt`, `best.v2-kq-swapped.pt`, `best.v3.pt`.

## Rebuilding from a clone (if something breaks while travelling)
```bash
bash setup/install_ubuntu.sh          # or: bash setup/download_models.sh   (Stockfish + the pretrained chess model)
source scripts/env.sh
python -m chess_player.vision.train_yolo --epochs 60 --batch 8   # ~2 min on a good GPU; writes data/yolo/best.pt
```
The dataset is in `data/dataset/`. New camera position = refit the grid on the empty board and recapture (the labels depend on the grid).

## Known problems / not done
- Pieces standing directly behind others (a queen right behind a pawn) can be missed or read as the front piece. A higher camera angle would help;
  more positions like pos20 would help.
- The camera and board were bumped repeatedly; the grid needed several refits. Consider the marker idea in GRID_ROBUSTNESS.md and a firmer mount.
- `data/board_to_robot.json` was NOT re-verified after the camera/board changes of this session: run `calib/hover_test.py` before picking.
- The picking problem in STATUS.md is untouched by this session.
- Setups for real games must be standard (queen on d1/d8), otherwise the engine and the board disagree.
- Two 17 MB debug videos (`data/pick_attempt*.mp4`) from the earlier session are left uncommitted.

## Board-to-robot calibration, later the same day (the camera shifted nothing; the BOARD had moved relative to the robot)
- `hover_test` on the empty board showed the tip landing about 5 cm (toward the a-file and rank 8) from the target; measured from the picture with the
  camera pose, fingertip vs where the maths put it (48.8, 49.0 mm). A 2-D shift (+48.8, -49.0) brought it to ~1 cm (`data/hover_check2/montage_after.jpg`).
- The marker-based hand-eye calibration (`calib/handeye_marker.py`, self-test recovers a 64 mm error to 1.6 mm) was written but NOT used: no sticker at hand.
- A fresh 6-point corner touch (`calib/board_to_robot --centers`) fitted badly by its own checks (RMS 16.5 mm, side mismatch 44 mm, tool tilt 19-27 deg at the touches),
  but a hover test with it, watched by eye, was accurate. It is the ACTIVE `data/board_to_robot.json` now. Saved aside: `board_to_robot.touch-corners-2026-11-06.json`
  (same), `board_to_robot.baked-baseline-2026-11-06.json` (the 2026-11-05 17-point fit with the shift baked in: the fallback),
  `board_to_robot.before-touch-2026-11-06.json` (the original 17-point fit).
- Then `calib/nudge_calibrate --height 20` over 10 squares (a8 h8 a4 h4 a1 h1 d1 d8 d4 e5): every square was accepted with NO nudge (h8 could have gone 2-3 mm),
  so `data/board_correction.json` is the identity. Calibration is good to ~2-3 mm by eye over the board, judged at 20 mm height.
- Gripper: the widest opening any grasp approach may use is 16.1 % (~33 mm), `data/gripper_max_open.json` -> `config.GRIP_OPEN_MAX_PCT`.
- New tools for the next step (the grasp): `calib/locate_piece.py` (YOLO -> board mm -> robot -> straight-down hover over a piece),
  `calib/compare_pose.py` and `calib/handeye_sample.py` (read the arm held by hand and compare it with the camera/maths), `scripts/teach_demo.sh`
  (teleop recorder: all joints + video). `ui/server.py`: click a highlighted move and the arm plays it (confirmation first), dry-run shows what to do by hand.

## Grasping, later the same day (READ THIS FIRST in a new chat)
**State at the end:** every piece type has a VERIFIED pick-and-place on an EMPTY board (c2 -> c3, landed upright within 1-3 mm); crowded squares are NOT solved.
The arm is home (roll -96.8), camera window running (scratchpad script, http://127.0.0.1:8771), pipeline (dry-run) on :8770.

### The recipe that works (all six types: p r n b q k)
Vertical tool (tilt 0), wrist roll -96.9 (jaws open along board x, the FIXED finger on the a-file side), safe height then straight down to
**15 mm** tip height (the pawn: 1 mm, base grip), jaws open 16.1 % (~33 mm), tip **5 mm toward the a-file** from the piece centre
(pawn 7.7 mm), then a **SLOW contact-sensing close** (`motion/slow_grip.py`: 0.5 % steps, contact = measured jaw lags the command by > 1 %, light squeeze),
lift 30 mm, carry, lower, **slow release**, lift, home. Contact readings: p 6.7, r 8.6, n 5.6, b 7.1, q 8.7, k 8.7 % (2.2-2.6 % = the jaws closed on AIR).
Per-piece numbers, YOLO bias and place corrections: `data/vision_grasps.json`. Pick positions come from YOLO (bias removed): `calib/teach_piece measure`.

### Tools (all in src/chess_player/calib unless noted)
- `locate_piece.py`: hover over a piece from YOLO with LIVE nudge commands (`z -5`, `x 3`, `y -2`, `open 12`, `close` (slow), `home`); `--height --shift-x/-y --open --roll --go`.
  Run it with stdin from a file (`tail -n 0 -f cmds | python -u -m ... --go`) to drive it from a script; the first Enter starts the arm.
- `grasp_vision.py`: one command pick and place from YOLO using the saved grasp (slow close/release). `grasp_joint.py`: replay a hand-taught joint pose (needs the piece where it stood when taught).
- `teach_piece.py measure|record`, `compare_pose.py`, `handeye_sample.py`, `handeye_marker.py` (needs a coloured sticker on the gripper; not used), `src/tools/torque.py hand` (arm limp, wrist roll held), `src/tools/safe_lift.py` (home).
- `board_correction.json` is the identity; `board_to_robot.json` is the 6-point corner touch (judged accurate by eye); backups `board_to_robot.*.json`.

### Lessons (mistakes that cost hours; do not repeat)
1. The jaw TIP frame is the FIXED finger's inner face (10 mm thick toward the a-file); the middle of the opening is gap/2 toward the h-file. A piece must have the fixed pad nearly touching BEFORE closing, or the moving jaw pushes it away.
2. A FAST close pushes narrow pieces over; use `slow_close`. Do not compare the measured jaw with a target built from the measured jaw (contact check bug).
3. Never close the gripper while lifting out of a grasp: it carries the piece. Open first.
4. YOLO is fooled by the arm in the picture: measure the piece with the arm HOME, or pass `--xy`. Tall pieces can flip between c1/c2.
5. YOLO's base point is a few mm off the true centre per piece type (`yolo_base_point_bias_mm`).
6. A hand-taught pose is only valid relative to where the piece really was; the camera cannot see the piece under the gripper. A replay of a pose with a tilted tool (18-23 deg) did not work; the VERTICAL recipe did.
7. The wrist roll range is +-180: asking for +83.1 (jaws the other way) made the wrist turn the long way round. Do not use roll flips without checking the path.

### NOT solved
- **Crowded squares.** Fingers are 10 (fixed) and 12 mm (moving) thick, neighbouring bases are 5.5 mm apart. The 33 mm opening puts the moving finger in a neighbour on the h-file side. A 5 % opening (10 mm) leaves ~1 mm per side: pushed the pawn. Rolling the wrist 90 deg (jaws along y) also pushed the pawn. Rolling 180 deg (fixed finger on the crowded side) made the wrist spin; untested properly.
  Ideas: move a neighbour aside first; thinner fixed finger (hardware); choose the free side by rule; teach the crowded grasp by hand with the VERTICAL roll -96.9 and the neighbours in place.
- Castling, captures (graveyard), promotion, knight grips from b1/g1: untested with the arm. Full game with the real arm: never run. Voice: untested. Model weights `data/yolo/best.pt` are NOT in git.
