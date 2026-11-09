#!/usr/bin/env bash
# Start-up self-test: camera, grid, YOLO, reach, arm, and 3 touch-downs checked by the camera. PASS/WARN/FAIL table at the end.
#     bash ~/Desktop/chess/wooden-chess/scripts/selftest.sh [--no-touch]
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source scripts/env.sh
python -m chess_player.calib.selftest "$@"
