# wooden-chess: SO-101 chess robot, 12-inch board with wooden pieces

Fresh start on the new rig. The code is a copy of the parent repo's `src/`, `scripts/`, `setup/` (branch `nithya`, 2026-11-07);
`data/` holds ONLY the arm-specific files (arm calibration, gripper tables, home pose). Everything that depends on the board or the
camera (grid, board_to_robot, dataset, YOLO, grasp numbers) starts empty and is re-made here. `.venv` and `bin/` are symlinks to the parent's.

Run from this folder: `source scripts/env.sh` (PYTHONPATH then points at THIS folder's src).

## Rig
- Board: 12 in (~305 mm) wooden board, wooden pieces (bigger bases, more room between pieces, easier to grip). MEASURE: square size,
  border, board thickness, piece heights/base diameters -> `src/chess_player/config.py` (still holds the old 25 mm values).
- Parent rigs for reference: original 33.3 mm squares / 340 mm board (pieces 19-23 mm base); 25 mm toy board (25 mm squares, 19 mm bases).
- Fixed camera: Ausha webcam (human's side, raised), brightness 95. Arm reach ~35 cm: check all 64 squares are reachable (reachability_sweep).

## Order of work
1. **Board calibration**: measure board; camera grid on an EMPTY board; `calib/board_to_robot` (touch points), `hover_test`.
2. **Nudge calibration**: `calib/nudge_calibrate` over a spread of squares -> `data/board_correction.json`.
3. **YOLO training**: capture with `vision/capture_chain.py` + `capture_safe.py`, `vision/train_yolo.py`. Check preview overlays and the val confusion matrix.
4. **Piece gripping**: `calib/teach_piece`, `locate_piece`, `grasp_vision`, `motion/slow_grip.py` (slow contact-sensing close).
5. **Collision control**: neighbour clearance with the new fingers/pieces, crowded squares (the unsolved problem of the old rig), captures/graveyard.

## Lessons carried over (details in docs/old-rig/SESSION_LOG_2026-11-06.md)
- Refit the grid only on an EMPTY board; check the preview before trusting a capture; set pieces in the STANDARD layout (queen on d1/d8).
- The jaw tip frame is the FIXED finger's inner face; the moving jaw is gap/2 toward the h-file. Close SLOWLY; never close while lifting out.
- YOLO is fooled by the arm in the picture: measure with the arm home. YOLO's base point has a per-piece bias.
- Wrist roll range is +-180: never command a flip without checking the path.
- Model weights (*.pt) are git-ignored; the machine has no GitHub login (user pushes).

## Status 2026-11-07 (end of session 1)
- **Board calibration + nudge calibration DONE**: `data/board_to_robot.json` (4-corner touch, poor by its own checks: RMS 11.8 mm) +
  `data/board_correction.json` (visual-by-eye nudge over 10 squares, leftover ~3 mm, scale 1.072). Validated by eye at 30 mm hover on
  g2 b7 e4 c6 f3: all on the centres. NOT checked: the true board height (z). Check it with a slow touch-down before any grasp.
- Board is ~25 mm closer to the robot than the first position; far ranks 7-8 are ~420 mm from the base (tool tilt 50-60 deg): vertical grasps are impossible there.
- Camera creeps slowly: `scripts/livecam.py` tracks it (ECC vs the locked reference frame) and rewrites `data/ext_corners.json`. Refit with
  `python -m chess_player.calib.fit_grid --rough "a1 h1 h8 a8 px"` (empty board) only if the window turns red. Keep the window open.
- Nudge tool directions (from the human's seat): d = right (a-file), a = left (h-file), w = away (toward robot), s = toward you; 15 mm = half a square.
- Next: YOLO dataset + training, then piece gripping, then collision control.

## Status 2026-11-07 (late): YOLO dataset captured, training blocked by the GPU
- `data/dataset/`: 12 positions (pos01 start; pos02-pos12 a short game with both castlings and a bishop/queen trade), 65 images, labels checked on previews.
  pos01 was fitted on an empty board; the grid was then REFITTED from the empty middle ranks with pieces on (see `data/ext_corners.locked-wooden.json`;
  the empty-board fit is `ext_corners.locked-wooden.empty-board-1.json`). The camera creeps, so the drift check mask now uses the floor + front rim.
- Chain state is in `data/dataset_chain_state.json` (next position = pos13). Board currently holds the pos12 position (c3, a6 played).
- Train with: `python -m chess_player.vision.train_yolo --epochs 60 --batch 8` (needs CUDA). On 2026-11-07 CUDA broke: kernel log shows
  `Xid 154 ... Node Reboot Required` after MMU faults following a suspend/resume. A REBOOT is needed; then restart `scripts/livecam.py`
  (camera window + tracking) and check the drift (`capture_safe.grid_shift`) before capturing more.

## Status 2026-11-07 (night): YOLO DONE
- GPU fixed by a reboot (Xid 154). Dataset: 13 positions, 70 images (pos01-pos13). Model `data/yolo/best.pt` (git-ignored; retrain:
  `python -m chess_player.vision.train_yolo --epochs 60 --batch 8`, ~100 s): val (3 held-out positions, 460 pieces) P 0.997, R 1.0, mAP50 0.995, mAP50-95 0.933.
- Fresh-position check before training on it: 64/64 squares read correctly (`python -m chess_player.vision.read_board`).
- Drift check gotcha: the reference frame (`data/ext_ref_frame.jpg`) goes stale when sunlight on the floor changes (match drops under 0.7 and shots are
  rejected). Fix = take a new reference frame and shift the locked grid by `capture_safe.grid_shift` (see the git history of this README / the session).
- Next: true board height (slow touch-down on an empty square), then piece gripping (`calib/teach_piece`, `locate_piece`, `grasp_vision`), then collision control.

## Status 2026-11-07 (night, later): board HEIGHT fixed and verified
- The touch calibration put the board ~27.5 mm too low (the tip was 2-3 mm above the board at a "30 mm" hover). Fixed in `data/board_to_robot.json`
  (t raised 27.5 mm, backup `board_to_robot.before-z-fix.json`) plus `z_plane` (extra height c=2.1 mm at d4, slopes 0.0162 /mm in x, 0.0295 /mm in y;
  `geometry.BoardToRobot` applies it). Checked by eye: a 12 mm hover is ~12 mm over d4, b2, g2, a1, b7, g7, h8.
- `calib/find_board_z.py` (slow touch-down, contact = the tip stops following the command) is only reliable on some squares: use it as a hint, not truth.
- The arm hovers measurably LOWER than the kinematics say when stretched out with a tilted tool (far ranks tool tilt 40-60 deg): keep checking heights at far squares.
- Next: piece gripping. Start with a pawn on d4 (middle of the board, tool near vertical).

## Pawn pick and place: WORKS (2026-11-07), e7 -> e5, placed within ~1 mm of the centre
Driven with `calib/locate_piece --square e7 --height 55 --open 16.1 --shift-x -11.5 --shift-y -1.9 --go` and its live commands (stdin from a file):
`z -15`, `z -15`, `z -10` (tip ~15 mm target = 19.8 mm measured), `close` (slow: contact at 8.3 % = 17 mm gap, the pawn body),
`z 30`, `y -30`, `y -30`, **`x 5`** (the place-centring correction: the pawn lands ~5 mm toward the a-file otherwise), `z -15`, `z -13`
(tip measured ~= the pick height, 19.6 mm), `open 16.1` (slow release), `z 30`, home.
- A grip at tip 25 mm only reaches the head (contact 5.5 %, 11.6 mm): too high. 13-20 mm measured tip height grips the body.
- Far ranks: tool tilt 31-37 deg at e7, 10-17 deg at e5. The arm overshoots upward when stretched out (lift came out ~8 mm too high).
- YOLO base point reads ~+2.4..4.6 mm high in y (toward rank 8) on hand-placed pawns: do not use it as the true piece centre.
- Next: automate this as a script (grasp_vision + data/vision_grasps.json for the wooden set), then rook, knight, bishop, queen, king; then crowded squares.

## `calib/pick_place.py` (2026-11-07): the pawn recipe as one command
`python -m chess_player.calib.pick_place e7 e5 --piece p [--yes] [--dry-run] [--save]` (pawn verified for real both ways, e7->e5 and e5->e7).
Flags (all optional, mm unless noted): `--open-pct --shift-x --shift-y --grip-tip --lift --place-dx --place-dy --place-extra`; `--save` stores them for that
piece in `data/wooden_grasps.json`. Other piece types start from the pawn numbers with `shift_x = -base_diameter/2` (a GUESS): run them one at a time,
read the printed contact gap (the width of the part gripped), and tune. The safe travel height adapts per square (far corners need it lower than 130 mm).
