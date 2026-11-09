# STATUS: SO-101 chess robot, WOODEN rig (read this first in a new session)

Last updated: **2026-11-08 (night)**. Folder `wooden-chess/` on branch `nithya`. START WITH THE SECTION "PLAN FOR 2026-11-09" AT THE END. The parent folder's code/notes are the OLD rigs
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
   or `scripts/run_pipeline.sh` (UI on :8770; add `--dry-run` for a simulated arm). Tested end to end in SIMULATION (normal moves, capture, castling, en passant,
   promotion) and a few real moves on 2026-11-08; real games still miss grasps (see the plan at the end).

## Update 2026-11-07 (late night): per-piece grasps + always-high carry (code done, NOT yet run on the arm)
User's findings with a full board: every non-pawn piece needs its own grip; tall pieces (n q k) tip over horizontally when placed (grip too low/loose);
pulling from the back ranks swept neighbours because the carry was only grip+30 mm. Changes (`motion/wooden_pick_place.py`):
- With other pieces on the board the piece is ALWAYS lifted straight up until its bottom is >= 70 mm above the board (`SAFE_CARRY_BOTTOM_MM`) before any
  sideways move (tip ~85-95 mm); lowered only where the arm cannot reach (far ranks), and then only if no tall piece is near the path (else `Blocked`).
- Per-piece numbers now include `squeeze` (hold harder) and `vert_speed` (lift/lower mm/s). `START` holds UNVERIFIED starting values for n b q k (higher grip,
  harder squeeze, slower lift); a piece is "taught" only once it is in `data/wooden_grasps.json` (the game logs a WARNING for untaught types).
- `calib/pick_place.py` now runs the SAME code as the game (one path: what you teach is what is played) and has `--others "d6:P e6:K"` to run/plan with neighbours.
- Dry runs of p n q k from e7 to e5 with neighbours pass (carry 85-95 mm, tilt 51 deg at e7).
**KNIGHT TAUGHT 2026-11-07 (staged tool `calib/teach_stages.py`, user validated every stage):** grip tip 17.55 mm (45 % of 39 mm), fixed finger -7.0/-0.3 mm
from the centre, contact 6.1 % (12.9 mm, the neck), squeeze 1.8, lift/lower 18 mm/s, safe carry 88 mm, release 4 % past contact (~10 %), place dx 3.0 dy 0 extra 2.
Knight faces the HUMAN (away from the robot) = the robot's own white knights; black knights (captures) face the robot: shift_y probably flips, not taught yet.
`teach_stages.py` usage: `python -m chess_player.calib.teach_stages --piece X --square e2 --dst e4 --keys` (stage 1 nudge from your seat with w a s d r f, Enter=ok;
2 `close`; 3 `lift`; 4 `carry`, `place`, nudge, `open` or `:release`, `leave`, Enter). Never run a dry run into the real folder (dry runs write data/teach/<piece>_dry).
The board moved ~8 mm vs the robot and the camera slid 14 px during the session: TAPE THE BOARD, refit the grid (`fit_grid`, camera window must be running) if the drift check shows > 3 px.
**ALL SIX TYPES TAUGHT 2026-11-08 (`data/wooden_grasps.json`, each stage validated by the user, nudges followed exactly; backups in `data/teach/backups/`, pictures + joint poses per piece in `data/teach/<piece>/`):**

| piece | height | grip tip mm | fixed finger x / y mm | contact % (gap mm) | squeeze | lift mm/s |
|---|---|---|---|---|---|---|
| p | 30 | 11.55 | -5.0 / +0.7 | 8.4 (17.5) | 1.8 | 18 |
| r | 35 | 11.75 | -10.0 / +0.7 | 9.5 (19.6) | 1.2 | 25 |
| n | 39 | 17.55 | -6.0 / -0.3 | 6.1 (12.9, the neck) | 1.8 | 18 |
| b | 49 | 13.55 | -10.0 / +0.7 | 9.4 (19.5) | 1.8 | 18 |
| q | 55 | 24.75 | -14.0 / -1.3 | 9.5 (19.6) | 2.2 | 14 |
| k | 60 | 25.00 | -12.0 / -2.3 | 11.0 (22.6) | 2.2 | 14 |

All: place dx +3.0, dy 0, extra 2.0 (defaults held: no nudge needed at e4), release 4 % past contact (cap 16.1 %), carry >= 70 mm piece-bottom.
Taught with the piece facing the HUMAN (white's side). Black pieces (captures to the graveyard) face the robot: shift_y probably flips, NOT taught.
Planning check with all 32 pieces on the board: all 20 legal opening moves plan (carry 82-88 mm, tilt <= 34 deg, jaws 8.6-16.1 %).
Taught at e2 -> e4 only and the board/camera crept during the session (shift_y drifts +0.7 .. -2.3 across pieces, maybe real, maybe drift): verify on far ranks (e7 -> e5) and a corner.
NEXT: real-arm checks with neighbours (`pick_place.py ... --others`), a far-rank pick, captures + graveyard, then a few engine moves in a real game (`scripts/run_pipeline.sh`).
**(old line) Teaching session: r, b, q, k, then re-check p (knight done) - DONE, see above.** Per piece: run
`python -m chess_player.calib.pick_place e2 e4 --piece q --others "d3:p f3:P" ` (middle of the board first), watch the lift/hold/place, tune `--grip-tip`,
`--squeeze`, `--vert-speed`, `--place-dx`, then add `--save`. Finish with one far-rank pick (e7) to confirm the safe height.

## Open / next: superseded by "PLAN FOR 2026-11-09" at the end of this file.

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
- The user wants simple instructions (one command, directions from their seat); do not ask them to click on pictures.
- The machine has no GitHub login: the user pushes (`cd ~/Desktop/chess && git push origin nithya`).

## 2026-11-08 (night): first real game attempts, findings, PLAN FOR TOMORROW
Done tonight: all six types taught (table above); pipeline wired (UI `scripts/run_pipeline.sh` and terminal `python -m chess_player.main --keyboard --robot-color white`);
UI fixes (live sync OFF by default; camera can no longer overwrite the game with an illegal reading; Recheck snaps to the start by chess logic: White's half standard +
ranks 3-5 empty => start, 8/8 correct); Ctrl+C in main.py freezes the arm (no swing home with a piece in the hand); `teach_stages.py` guards (type the piece name).
User's pointers from the real run (to fix next, in this order):
1. **Knight misses often; queen struggles to grip.** Taught only at e2->e4 while the board/camera crept; calibration error (5-10 mm) varies over the board, so offsets that
   worked at e2 miss elsewhere. Fix ideas: teach each type at 3 squares (near/mid/far) and interpolate; visual servoing at hover; V-groove self-centring pads.
2. **Camera validation must always run (ReAct: observe -> act -> verify -> recover).** `--keyboard` mode has NO camera verification. Verify per transfer: source empty,
   destination holds the right colour upright, nothing else changed; on failure re-measure with YOLO and retry (`play(recover=...)` hook exists). Camera must be READ-ONLY for
   the game state (verify, never rewrite).
3. **Graveyard/capture release goes too deep:** the table height is `-BOARD_THICKNESS_MM` (25 mm ESTIMATE; real board probably thinner). Measure it, or ground-sense: descend slowly,
   stop when the arm lags the command (like `calib/find_board_z.py`), release just above contact.
4. Speed is too slow (~35-45 s per move): only AFTER the game runs without grasp misses (vert_speed, CARRY_MM_PER_S, settle, slow_close steps).
5. **Portability (Jammu demo):** every setup will differ. Plan: rigid base plate (arm + board stops + camera mast = fixed transforms), ArUco markers on the board corners AND the
   robot base so camera->board->robot is computed automatically at start, wrist-camera visual servoing for the final approach (kills residual calibration error), a
   5-minute startup self-check (hover 3 squares, grid check), portable diffuse lamp + clamps. Calibration files become per-setup profiles.
6. Lighting: the room is dark, the picture grainy, black pieces on the board's near edge are misread. A lamp over the board + the camera higher/farther back fixes most of it.
Open question from the real run: why the user pressed STOP at 00:46:18 at the start of the first capture (arm was at its home pose, nothing dropped).

## Problem log 2026-11-08 (context: what went wrong, why, and what was done)
- **Camera/board drift all evening.** The Ausha webcam slid ~14 px and the board moved ~6-9 mm vs the robot between sessions; the grid was refitted 4 times
  (`calib/fit_grid`, backups `data/ext_corners.before-regrid-*.json`). Two refits were wrong because the rough corners were guessed under tall white pieces: take them
  from a zoomed picture. After every refit RESTART `scripts/livecam.py` (kill by pid; its tracker otherwise drags the grid back to the old reference).
- **Taught offsets drift:** the same pawn gave shift_x -10.6 (earlier) vs -5.0 (tonight); the knight crept -9 to -6. Cause = board/camera/calibration creep, so every
  taught number is only valid for the setup it was taught in (see the portability plan).
- **Teaching-tool accidents:** a piece taught with the wrong `--piece` letter overwrote another piece's entry twice (restored from the logs). `teach_stages.py` now asks you to
  TYPE the piece name, and backs up `wooden_grasps.json` to `data/teach/backups/` before every save. Dry runs write to `data/teach/<piece>_dry`.
- **UI / camera sync problems (user: "the simulator keeps shifting detections, the arm never plays"):** Black's pieces stand at the camera's near edge in a dark, grainy
  picture: back-rank types are misread with high confidence (k/q/b/n swapped), the e7 pawn is seen on e6, a8/b8/g8 are missed. Live sync then REPLACED the game with
  illegal or wrong boards, Recheck refused or invented positions, and one misread board was saved as YOLO training data (deleted). Fixes (code, all in git):
  `BoardWatcher.stable()` majority vote (a few flickering squares no longer block the board), live sync OFF by default (`config.LIVE_SYNC_DEFAULT`), illegal camera readings are
  never applied or saved, impossible readings pruned (`state.prune_impossible`: max 8 pawns / 16 pieces per side), start detection by chess logic (`state.start_mismatch`:
  White's half standard + ranks 3-5 empty = start, ignoring Black's side; 8/8 Recheck tests right), Recheck logs why it rejects. `observe`/`recheck`/`read_board` never trust Black's piece types.
- **Sim arm bug:** `WoodenPickPlace.execute` treated a FakeArm as real, so the contact check failed ("closed on nothing"); fixed (`real = not FakeArm`).
- **A dry run polluted `data/board_state.json`** (shared with the real pipeline) with a lone-queen test position; the pipeline loaded it. After any dry run/test, reset the position (`Start`).
- **Terminal mode** `python -m chess_player.main --keyboard --robot-color white` plays with no UI/camera sync (types your moves); Ctrl+C freezes the arm (no swing home).
  It has NO camera verification (see plan item 1).
- **Arm connection scare:** `scan_motors.py` found no servos until the USB board was re-plugged (device re-created); after that all 6 answered. If "Missing motor IDs": replug USB, re-scan.
- **Release opening:** `release_pct` (4 % past the contact reading, cap 16.1 % = `data/gripper_max_open.json`); earlier releases opened to ~15 %.
- **STOP at 00:46:18** by the user at the start of the first capture: the reason was never given (arm at home, nothing dropped). Ask.
- Unverified on the real arm: far ranks (e7), corners, captures/graveyard, castling, promotion, black pieces (they face the robot), neighbours' clearance at crowded squares.

## PLAN FOR 2026-11-09 (the user's pointers + proposals; nothing here is built yet unless stated)
User's pointers after the first real game: (a) knight still missed often, queen struggled to grip; (b) camera validation must ALWAYS run, like a ReAct policy, it must not
be redundant; (c) a capture puts the piece outside the board but goes far too deep, it should release when it feels the ground; (d) speed is far too slow but touch it only
AFTER the whole game runs with no grasp misses; (e) more grasp fine-tuning is needed but the user cannot teach every case by hand: suggest how; (f) everything (board, arm,
camera) will be placed differently tomorrow and at the Jammu demo: no "recalibrate every time" answer; (g) UI work is for the morning.
1. **Camera ReAct loop (observe -> act -> verify -> recover), read-only for the game state.** Per transfer: YOLO (arm at home) measures the real piece position; act;
   verify source empty + destination holds the right colour upright + nothing else changed (Black's types NOT trusted, only occupancy/colour); on failure re-measure and retry
   from where the piece really is (`PickPlace.play(recover=...)` exists). Add a no-camera check too: measured gripper contact vs the taught contact per type (knight 6.1 %; 8-9 % = head/edge):
   release, shift ~2 mm, retry within a bounded search. Wire it into BOTH the UI and `main.py --keyboard`.
2. **Grasp reliability without hand-teaching every case:** (i) teach each type at 3 squares (near/mid/far) and interpolate; (ii) an overnight self-test: pick/place each type 50-100x
   across the board, log contact + success, auto-tune shift/height/squeeze; (iii) wrist-camera visual servoing at the hover (gripper camera is currently NOT plugged in) so the board
   calibration error stops mattering; (iv) self-centring pads (3D-printed V-groove / silicone) make a few mm of error irrelevant (cheap, big win); (v) teach black pieces (facing the robot) separately.
3. **Capture/graveyard depth:** `config.BOARD_THICKNESS_MM = 25` is an ESTIMATE and sets the table height used for graveyard + spare slots. Measure it once, or ground-sense: descend
   slowly, stop when the arm lags the command (as `calib/find_board_z.py`), release just above contact. Also check slot reachability (`config.GRAVEYARD`, `SPARES`).
4. **Speed** (~35-45 s/move): only after zero misses. Knobs: `vert_speed` per piece, `CARRY_MM_PER_S`, `arm.settle`, `slow_close` step/wait.
5. **Portability (Jammu): remove the need for manual calibration.** (i) rigid base plate: arm base + board stops + camera mast on one plate, calibrate once, it travels as a unit;
   (ii) ArUco markers on the board corners and the robot base (`vision/aruco_homography.py`, `vision/make_markers.py`, `data/aruco_markers_A4.png` exist) so camera->board->robot is computed
   automatically at start, no touching; (iii) wrist-camera servoing for the final approach; (iv) a 5-minute startup self-check (hover 3 squares + grid check, clear pass/fail); (v) calibration
   files become per-setup profiles; (vi) a portable diffuse lamp + clamps (the dark grainy room caused most vision trouble); camera higher/farther so Black's near edge is inside the frame.
6. **Tomorrow's first steps (setup will differ):** place board/arm/camera, tape the board; start `scripts/livecam.py`; refit the grid on an EMPTY board (`fit_grid`) and restart livecam;
   redo board->robot (`calib/board_to_robot`) + nudge by eye (`calib/nudge_calibrate --height 30`) + the height check; check Z with a slow touch-down; then a one-piece pick
   (`calib/pick_place e2 e4 --piece p`) before any game. The taught grasp numbers may need re-tuning because they depend on the calibration (item 5 would fix that).
   Suggested order of work: 1 (camera loop) -> 3 (graveyard) -> 5 (base plate + markers) -> 2 (pads / servoing) -> 4 (speed).
7. **Git:** the machine has no GitHub login. The user pushes: `cd ~/Desktop/chess && git push origin nithya`.

## 2026-11-09 (end of day): what changed, what is verified, what to do first tomorrow
**Layout/camera:** board moved ~29 mm toward the robot (ranks 1-4 grip vertical, rank 8 still 31-39 deg). A torch over the board gives a clean picture (the room is otherwise dark).
`scripts/regrid.sh` re-makes the camera grid (EMPTY board, no clicking); `scripts/recalibrate.sh` redoes the arm calibration (4-corner touch + nudge by eye + hover check);
`calib/auto_grid.py --pieces` shifts the grid with pieces on (checkerboard search on the empty middle). The camera window follows drift only under 8 px: refit after any bump.
**Grip logic (all in `motion/wooden_pick_place.py`, used by the game, the UI and `calib/pick_place.py`):** taught numbers + camera centring (3 mm limit) + shift search after a miss
(x/y steps, up to 4 retries; rook/bishop/queen/king only 1 retry: they stand 4 mm from neighbours) + contact check (1.5 % vs taught width; < 4 % = nothing held) + safe hover/carry height.
All six types re-taught on 2026-11-09 on their HOME squares (`scripts/teach_all.sh`; knight b1, queen d1, king e1, bishop c1, rook a1, pawn e2). Batch results: pawn 8/8, knight 2/2, queen/king/rook 2/2 first try;
the bishop moved properly (the camera check disagreed; it had no camera reference) - re-teach it once for a `cam_ref`. Learned offsets are RECORDED but NOT applied (`CHESS_USE_LEARNED=1` applies them): a square needed +5 mm in one run and +2.5 in the next.
**Per-move camera verification (`game/verify.py`, `Session._verify_and_fix`):** looks before/after every robot move (occupancy + colour only, differential, read-only), plays a simple move again or picks up a fallen piece,
else raises `MoveNotVerified` (a GraspFailed: the human finishes the move). `VERIFY_MOVES` in config. Tested in logic + camera; NOT yet with a real deliberate failure.
**Test tools:** `calib/batch_test.py` (moves, camera check, before/after pictures in `data/batch_logs/`), `vision/eval_board.py` (YOLO vs a known position), `scripts/teach_all.sh`.
**Stockfish:** was deterministic (same opening every game); now picks randomly among the best 4 lines within 35 cp (`ENGINE_VARIETY_CP`, 0 = off). Restart the UI to get it.
**Board map (autocal) - UNVERIFIED, check first tomorrow:** `calib/autocal.py` carried one pawn over squares and measured its landing error with the camera: the error is position-dependent (x about -1 mm at a1 to +6 mm at g1/h1,
~1.4 mm scatter). `data/board_residual.json` = affine fit from 27 squares (ranks 1-6 + h8), **x correction only** (y zeroed: it is mixed up with camera perspective bias; the original is `board_residual.xy-original.json`).
It moves every pick/place by -(E(square) - E(the square that type was taught on)). Switch off: `CHESS_NO_RESIDUAL=1`. A rejected over-extrapolated fit is `board_residual.REJECTED-*.json`.
**FIRST THING TOMORROW:** put the 32 pieces on the start position (the lone autocal pawn is on h8 -> back to e2), then `python -m chess_player.calib.batch_test` (8 pawn moves). Before the map it was 5/8 first try with 5 shifted retries.
If it is not clearly better, `export CHESS_NO_RESIDUAL=1` (or delete data/board_residual.json) and move on.
**Open:** bishop re-teach; YOLO retrain under the torch light (g2/b2 pawns read as knights, Black near edge, king as queen); a lone pawn is detected badly by YOLO (autocal needed conf 0.15);
real game with the verification on; far ranks/captures/graveyard depth (`BOARD_THICKNESS_MM` is still an estimate); ArUco markers (sheet `data/aruco_markers_A4_apriltag36h11.pdf` printed, not wired); self-centring pads (no 3D printer).
Camera: after a replug `/dev/video4` loses the user ACL: `sudo setfacl -m u:$USER:rw /dev/video4 /dev/video5` (or `sudo usermod -aG video $USER` + re-login).
Committed locally on 2026-11-09 (the user pushes: `cd ~/Desktop/chess && git push origin nithya`).

## 2026-11-09 (later): calibration without hands/eyes, session trim, self-test (BUILT; camcal NOT yet run on the real arm)
- `calib/camcal.py` (the arm touches 8 squares, ground-sense; the camera finds the tip = lowest changed pixel vs an arm-at-home picture; board px->mm from the grid + FK of the
  measured joints -> rigid fit; then a check on 3 fresh squares). Needs scripts/livecam.py running. Pictures with the tip marked: `data/camcal/`. Math tested on a simulation (0.4 mm
  from truth with the tip bias known, 5 mm without). **Run ONCE first: `python -m chess_player.calib.camcal --learn-bias`** (against the current good nudged calibration; saves
  `data/camcal_bias.json`), look at `data/camcal/bias_*.jpg` (red dot must be on the jaw tip), then `python -m chess_player.calib.camcal`. It moves the old board_correction/board_residual aside.
- `WoodenPickPlace.note_landing` + session hook: after each verified move the piece's landing is measured against its `cam_ref`; a bounded session trim (<= 4 mm, never saved) is added to
  every pick/place. Logic tested with a stub locator (converges to the injected drift).
- `calib/selftest.py` / `scripts/selftest.sh`: camera, grid, YOLO, reach, arm, 3 touch checks (e3 d6 g4 empty). `scripts/setup_location.sh` = regrid + camcal + selftest in one go.
