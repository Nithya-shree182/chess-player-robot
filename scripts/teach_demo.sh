#!/usr/bin/env bash
# Teach a move by teleoperation: the follower arm copies the leader, and every joint of both arms is recorded at 30 fps
# together with the board camera's video (frame k of video.mp4 = row k of poses.csv).
#   scripts/teach_demo.sh "pawn c2 to c3"          # one take (default task: pawn c2 to c3)
#   scripts/teach_demo.sh "pawn c2 to c3" 5        # five takes
# Needs: the camera window running (serving http://127.0.0.1:8771/video), both arms powered. Enter = start / stop a take.
# Saved to data/demos/<time>_<task>/ : poses.csv, video.mp4, info.json
set -e
cd "$(dirname "$0")/.."
source scripts/env.sh
export CHESS_CAMERA_URL="${CHESS_CAMERA_URL:-http://127.0.0.1:8771/video}"
exec python -m chess_player.teach.record_poses --task "${1:-pawn c2 to c3}" --episodes "${2:-1}"
