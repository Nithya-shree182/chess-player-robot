# STATUS: SO-101 chess robot, WOODEN rig (read this first in a new session)

Last updated: **2026-11-07 (night)**. Folder `wooden-chess/` on branch `nithya` (commit a57d8bb and later). The parent folder's code/notes are the OLD rigs
(33.3 mm and 25 mm toy board): `../STATUS.md`, `docs/old-rig/`. `README.md` here has the long session notes (recipes, numbers, lessons).

## The rig
- 12 in folding wooden board, **30 mm squares** (playing area 240 mm), thin line 2 mm; wooden pieces, heights p30 r35 n39 b49 q55 k60 mm
  (base diameters in `config.PIECES` are ESTIMATES 22-26 mm). **Board thickness 25 mm is an ESTIMATE: measure it** (sets the graveyard height).
- Robot plays WHITE (rank 1 at the top of the camera picture = robot side; h-file on the left, a-file on the right of the picture).
- The board spans ~165-425 mm from the robot base: the tool is tilted 30-60 deg on the far ranks (tilted, not vertical, grasps there).
- Fixed camera: Ausha webcam, human's side, raised. The room is dark (no more light possible): noisy picture, so `scripts/livecam.py` averages still pixels over time.

## What works (all verified on the real robot unless noted)
1. **Camera window** `scripts/livecam.py` (KEEP IT RUNNING): live picture with the locked grid + square names, tracks slow camera creep (ECC against the
   reference frame, rewrites `data/ext_corners.json`), serves `http://127.0.0.1:8771/video|shot` for every tool (`config.EXT_CAMERA_URL` points there).
   Start: `source scripts/env.sh && python scripts/livecam.py`. Locked grid: `data/ext_corners.locked-wooden.json` (read-only); refit on an EMPTY board with
   `python -m chess_player.calib.fit_grid --rough "a1x,y h1x,y h8x,y a8x,y"` (median of 7 frames; restart the window after).
   The drift check (`vision/capture_safe.py`, mask = wall panel edges + robot base) rejects capture shots if the scene moved > 2.5 px or the match is < 0.7.
2. **Board calibration**: `data/board_to_robot.json` (4-corner touch, poor by its own checks, RMS ~12 mm) + `data/board_correction.json` (nudge by eye over 10
   squares, ~3 mm leftover) + the **height fix** (`t` raised 27.5 mm and `z_plane` in the json: the tip was 2-3 mm above the board at a "30 mm" hover).
   Validated by eye: 12 mm hover is ~12 mm over d4 b2 g2 a1 b7 g7 h8. The camera cannot measure tip height (one low camera): your eye is the reference.
3. **YOLO**: 13 positions / 70 images in `data/dataset/`, model `data/yolo/best.pt` (git-ignored; `python -m chess_player.vision.train_yolo --epochs 60 --batch 8`,
   ~100 s). Held-out P 0.997 R 1.0 mAP50 0.995; a fresh position read 64/64. Read the board: `python -m chess_player.vision.read_board`.
4. **Pawn pick and place** (the recipe): `python -m chess_player.calib.pick_place e7 e5 --piece p --yes` (verified both ways, lands within ~1 mm of the centre).
5. **Game wiring**: `motion/wooden_pick_place.py` (class `WoodenPickPlace`, a drop-in for `PickPlace`) is used by `game/session.py` (`config.USE_RECIPE_GRASP`).
   It does captures (to graveyard slots on the table beside the board, 14 slots), castling, en passant, promotion (spare queens at `config.SPARES`: put one
   at each spot before a game), picks the carry height from the tallest piece near the path, opens the jaws only a little in a crowded spot, retries an empty
   hand and detects a slipped piece. Per-piece numbers: `data/wooden_grasps.json` (defaults = the pawn). Run a game: `python -m chess_player.main`
   or `scripts/run_pipeline.sh` (UI on :8770; add `--dry-run` for a simulated arm). NOT yet tested end to end (see below).

## Open / next (in this order)
1. **KNIGHT grasp** (the user reports the knight slips: its head is off-centre so the pads close on a changing shape and it falls). Plan: one hands-on run
   with `calib/locate_piece` (live commands) to find the grip height (start low, ~9 mm tip, on the round base), the shift, the contact gap and the place
   correction; then save with `calib/pick_place ... --piece n --save` so `WoodenPickPlace` always uses them. Needs a knight on e7 and its facing noted.
2. The other piece types (r b q k): each needs its numbers (`pick_place.py ... --piece X`, read the printed contact gap; tune `--shift-x`, `--grip-tip`).
3. Full game test: a simulated game (`/tmp` script idea: `Session(dry_run=True)`, engine vs engine, `s.check(move)` + `s.execute(move)`) was too slow (IK in Python)
   and timed out: rerun with fewer plies; then the real arm on a few engine moves; then the UI with the camera watching human moves.
4. Board thickness (graveyard height), graveyard/spare slots reachability on the real arm (`config.GRAVEYARD`, `SPARES`).
5. Collision control in crowded squares (the h-file side neighbour vs the moving finger) and verifying each move with the camera.

## Lessons / gotchas (cost hours)
- Never kill by `pkill -f` / `pgrep -f` patterns that match your own command line (it kills the shell): kill by pid.
- The camera creeps and gets bumped: refit on an EMPTY board, or from the empty middle ranks with pieces on. A stale reference frame (sunlight on the floor
  changes) makes the drift check reject shots: re-take the reference and shift the grid by `capture_safe.grid_shift`.
- Hand-touched calibrations are good to ~10-15 mm only; the nudge-by-eye tool (`calib/nudge_calibrate --height 30`, directions from the human's seat:
  d right, a left, w away, s toward you; half a square = 15 mm) got to ~3 mm.
- A grip at tip height 25 mm only reaches a pawn's head (contact 5.5 %); 13-20 mm grips the body (8.3 %, 17 mm gap). The held piece sits against the FIXED
  finger: place the tip at centre - gap/2 + place_dx (pawn place_dx = +3 mm).
- YOLO's base point reads 2-5 mm high in y: do not use it as the true piece centre.
- GPU: after suspend/resume CUDA may break (`Xid 154 Node Reboot Required`): reboot.
- The user wants simple instructions (one command, directions from their seat); do not ask them to click on pictures. No Claude/AI trace in commits.
- The machine has no GitHub login: the user pushes (`cd ~/Desktop/chess && git push origin nithya`).
