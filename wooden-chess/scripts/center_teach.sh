#!/usr/bin/env bash
# Teach square centres by teleop in the order h1 g1 .. a1, h2 .. a8 (empty board). Enter saves + next. Output: data/center_teach.json.
# Output: data/center_teach.json. Hold the leader near the follower's pose before starting.
cd "$(dirname "$0")/.." && source scripts/env.sh
exec python -m chess_player.calib.center_teach
