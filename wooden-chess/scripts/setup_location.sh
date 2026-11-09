#!/usr/bin/env bash
# NEW PLACE / board or camera moved: the whole set-up in one command, no hand guiding, no nudging.
#     bash ~/Desktop/chess/wooden-chess/scripts/setup_location.sh
# 1. tape the board; EMPTY board, fully in the picture, lamp on.   2. camera grid (regrid.sh)   3. arm <-> board by camera (camcal)   4. self-test
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source scripts/env.sh
bash scripts/regrid.sh || { echo "STOP: the camera grid failed (see above)"; exit 1; }
echo; read -r -p "Board EMPTY, hand near the power switch. Arm calibration by camera (about 4 min). Enter = go, n = stop: " r
[ "$r" = "n" ] && exit 1
python -m chess_player.calib.camcal || { echo "STOP: camcal failed (see above and data/camcal/*.jpg)"; exit 1; }
echo; echo "Done. Put the 32 pieces on the start squares, then: bash scripts/selftest.sh (3 touch squares e3 d6 g4 must be empty) or --no-touch."
bash scripts/selftest.sh --no-touch
