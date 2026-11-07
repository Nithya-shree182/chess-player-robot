# Keeping the grid locked (no more "remove all pieces and re-fit")

Written 2026-11-06. Prototype: [src/chess_player/vision/grid_lock.py](../src/chess_player/vision/grid_lock.py) (not wired in yet).

## Why the grid keeps breaking
`ext_corners.json` is four fixed pixel positions fitted once, from an empty board (`fit_phone_corners`) or from the start
position (`auto_corners`). Anything that moves the camera or the board by a few pixels makes all 64 squares wrong, and the only
way to refit is to clear the board. The stored dataset shows it: the camera pose differs by ~70 px between sessions, with
lighting ranging from washed-out to orange.

Two different things can move, and they need different fixes:

| What moved | Breaks | Fix |
|---|---|---|
| Camera | the pixel grid (vision) | re-find the grid every look (this doc) |
| Board relative to the camera only | the pixel grid | same, if the markers are ON the board |
| Board relative to the ROBOT | `board_to_robot.json` (picking/placing) | mechanical stops for the board, or markers on the robot base (see below) |

## Options tested (on the real frames in `data/dataset` + the live frame, 2026-11-06)
1. **ArUco/AprilTag markers on the board's border** (recommended). Detection is on grayscale, so the yellow cast does not matter.
   Synthetic markers pasted at the true board positions, then degraded: found 4/4 and corner error 0.4-0.7 px (~0.15 mm) with
   washed-out, dark, yellow, yellow+washed-out, noisy, low-res (640x360). Failed on strong blur (sigma 2 px) and hard shadow
   (2/4, 3/4). Camera-move test with 6 markers: grid error 0.1-0.3 px (<0.1 mm); with 2 of 6 markers visible 1.1 px; all
   markers hidden -> falls back to features.
2. **Feature re-lock** (SIFT on the board's permanent features: frame, letters, seam, dots, against a saved reference frame).
   No stickers. On real frames with pieces on the board and the camera ~70 px off, corner error 1.7-9.7 px (~0.5-3 mm) against a
   sub-pixel snap to the printed squares. Good as a fallback, not as the main method.
3. **Current method** (fit once on an empty board). Sub-pixel when fresh, but no way to follow a move. Keep it for the first
   reference only.
4. Not tried: ChArUco (needs a printed board area; the board itself is already a checkerboard), tracking the white line around
   the playing area (the line is thin: ~4 px at this distance).

## Findings that shape the design
- The Ausha webcam exposes only `brightness` and `power_line_frequency` (v4l2): **no manual exposure or white balance**. The
  colour cast and auto-exposure cannot be locked, so the method must not depend on them. Markers on grayscale do not.
- Use **DICT_APRILTAG_36h11** (or 6x6), not the repo's current `DICT_4X4_50`: on 204 real frames without markers 4x4_50 gave 4
  false detections; 5x5, 6x6 and AprilTag gave 0.
- The camera also does 1920x1080 MJPG: use it for the reference registration (more pixels per marker).
- At 1280x720 the far border is ~3 px/mm, the near one ~4.5 px/mm. A 16 mm marker is ~50-75 px wide: plenty. 10 mm also worked in
  the synthetic test (less margin under blur).
- **Tall pieces overlap the far border** (the white rook at the far corner hides a far-corner marker in the live frame). Put
  markers on the near corners and the left/right sides (mid-board), not on the far border.

## How to do it
1. Print 6 markers (AprilTag 36h11, ids 0-5; `vision/make_markers.py` needs its dictionary changed and 6 ids), each with a 2 mm
   WHITE margin (the black frame needs a light surround; the board border is dark). Matte paper or matte laminate: glossy
   stickers glare. Size 12-16 mm: measure the flat part of the border first (the letters take some of it; covering them is fine).
2. Stick them flat on the board border: near-left, near-right, left-middle, right-middle (+ far corners if they stay visible).
   Their exact positions do NOT need measuring.
3. With the board empty and the camera where it should be: `fit_phone_corners` as today, then
   `grid_lock.register(frame, grid_px)` (saves the marker positions and a reference frame).
4. In the pipeline replace the fixed `corners_view(frame, C.EXT_CORNERS)` by `GridLock().lock(frame)` every N frames (about
   2 Hz is plenty). Keep the last good grid when it returns None, and show "grid lost" in the UI.
5. Safeguards before using a new grid: markers agree with one homography (RMS < 1.5 px, already in the code), the grid is a
   convex quadrilateral of plausible area (in the code), a1 still reads as a dark square (`colour_check`), and a big jump (> ~15
   px) must persist for 2-3 looks before it is accepted (smoothing/hysteresis, not written yet).

## The robot side (not solved by the above)
`board_to_robot.json` (17 touch points) is only valid while the board stays put relative to the robot base. Cheapest fix:
mechanical stops (two L-shaped blocks or tape stops fixed to the robot's table that the board's corner is pushed into). Software
fix: a few markers on a plate fixed to the robot, then `T_robot<-board = T_robot<-camera * T_camera<-board` through the camera
(needs camera intrinsics: there is no `camera_intrinsics.npz` yet, and then `solvePnP`).

## Limits of these tests
Markers were rendered into real frames, not printed and photographed; real stickers add glare, curl and a few-mm placement
error (the reference-based design absorbs the placement error, not the glare or curl). Marker and feature tests used one camera
and one lighting family. Re-run the check with printed markers before relying on it.
