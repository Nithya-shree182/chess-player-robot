#!/usr/bin/env bash
# Re-calibrate the ARM to the board after the board, the arm or the camera was moved (tape the board down first).
#     bash ~/Desktop/chess/wooden-chess/scripts/recalibrate.sh            # arm calibration + hover check
#     bash ~/Desktop/chess/wooden-chess/scripts/recalibrate.sh --grid     # first re-make the camera grid (EMPTY board)
#     bash ~/Desktop/chess/wooden-chess/scripts/recalibrate.sh --check    # only the hover check, nothing changes
#
# Steps (each one asks before it moves the arm):
#   0. [--grid] camera grid: scripts/regrid.sh
#   1. back up data/board_to_robot.json + data/board_correction.json (data/*.before-recal-<time>.json)
#   2. 4-corner touch with the closed jaw tip (torque off, you guide the arm)        -> board_to_robot.json
#   3. nudge by eye over 10 squares at 30 mm. Keys from YOUR seat: a/d = left/right, w/s = away/toward you,
#      1 mm per press, Enter = this square is right, k = skip, q = finish           -> board_correction.json
#   4. hover check: tip 30 mm over corners, centre and far ranks, 6 s each; you judge by eye.
#      If something is off, run step 3 again:  python -m chess_player.calib.nudge_calibrate --height 30 --keys
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source scripts/env.sh
D=data
STAMP=$(date +%m%d-%H%M)
MODE=all
for a in "$@"; do
  case "$a" in --grid) MODE=grid ;; --check) MODE=check ;; esac
done

ask() { read -r -p "$1 [Enter = yes, n = stop] " r; [ "$r" != "n" ]; }

hover_check() {
  echo; echo "== Hover check: the arm will move to 12 squares, 30 mm above the board. Watch the tip. =="
  ask "Clear the board of pieces and keep a hand near the power switch. Go?" || exit 1
  python -m chess_player.calib.hover_test --height 30 --hold 6 \
    --squares a1 h1 a8 h8 d4 e4 d5 e5 b7 g7 d8 e8
  echo "Was the tip over the middle of every square? If not: python -m chess_player.calib.nudge_calibrate --height 30 --keys"
}

if [ "$MODE" = check ]; then hover_check; exit 0; fi

if [ "$MODE" = grid ]; then
  echo "== 0. Camera grid (the board must be EMPTY and fully in the picture) =="
  bash scripts/regrid.sh || { echo "grid failed: fix that first (see message above)"; exit 1; }
fi

echo "== 1. Backing up the current arm calibration =="
for f in board_to_robot board_correction; do
  [ -f $D/$f.json ] && cp $D/$f.json $D/$f.before-recal-$STAMP.json && echo "   $D/$f.before-recal-$STAMP.json"
done

echo; echo "== 2. 4-corner touch =="
echo "   Torque goes OFF: guide the arm by hand and touch the closed jaw tip to the corners as asked."
ask "Ready?" || exit 1
python -m chess_player.calib.board_to_robot || { echo "touch calibration failed; the backup is in $D/*.before-recal-$STAMP.json"; exit 1; }

echo; echo "== 3. Nudge by eye =="
echo "   The arm moves to 30 mm over each square. From your seat: a/d left/right, w/s away/toward you, Enter = right, q = finish."
ask "Ready?" || exit 1
python -m chess_player.calib.nudge_calibrate --height 30 --keys || { echo "nudge stopped; step 2 is saved, step 3 is not complete"; exit 1; }

hover_check
