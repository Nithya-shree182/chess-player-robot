#!/usr/bin/env bash
# One command to re-make the camera grid. EMPTY board, fully in the picture, board taped. Run from anywhere:
#     bash ~/Desktop/chess/wooden-chess/scripts/regrid.sh
# It (re)starts the camera window, finds the board by itself (no clicking), saves the grid, restarts the camera window so
# its tracker uses the NEW grid, and leaves the check picture in data/ext_corners_check.jpg (open it to confirm).
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source scripts/env.sh
LOG="${TMPDIR:-/tmp}/livecam.log"

stop_livecam() {  # by pid, from ps (never pkill -f with a pattern that matches this shell)
  for p in $(ps -eo pid,args | awk '/[s]cripts\/livecam.py/ {print $1}'); do kill "$p" 2>/dev/null; done
  sleep 2
}
start_livecam() {
  nohup python scripts/livecam.py > "$LOG" 2>&1 &
  for _ in $(seq 1 20); do
    curl -s -m 2 -o /dev/null http://127.0.0.1:8771/shot && return 0
    sleep 1
  done
  echo "camera window did not start: see $LOG"; return 1
}

stop_livecam
start_livecam || exit 1
sleep 5
if python -m chess_player.calib.auto_grid "$@"; then
  stop_livecam
  start_livecam && echo "OK: grid saved, camera window restarted. Check data/ext_corners_check.jpg"
else
  echo "grid NOT changed (see message above and data/ext_corners_check.jpg)"; exit 1
fi
