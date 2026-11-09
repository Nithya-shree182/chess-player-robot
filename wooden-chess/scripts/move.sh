#!/usr/bin/env bash
# Pick a piece from one square and place it on another, going to the HAND-TAUGHT centres exactly (no corrections). Asks for piece, from, to.
# Usage: scripts/move.sh            (asks)   or   scripts/move.sh p e2 e4
cd "$(dirname "$0")/.." && source scripts/env.sh
export CHESS_CAMERA_URL=http://127.0.0.1:8771/video
curl -s -m 3 -o /dev/null http://127.0.0.1:8771/shot || echo "(camera window not running: fine, the camera no longer steers the arm; only the grip-point pictures are skipped)"
if [ $# -ge 3 ]; then P=$1; S=$2; D=$3; else
  read -rp "Piece (p n b r q k): " P
  read -rp "From square (e.g. e2): " S
  read -rp "To square (e.g. e4): " D
fi
exec python -m chess_player.calib.simple_move "$P" "$S" "$D" "${@:4}"
