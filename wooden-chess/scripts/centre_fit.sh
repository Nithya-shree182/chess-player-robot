#!/usr/bin/env bash
# Fit + validate the square-centre formula from the taught centres (data/center_teach.json). Output in data/centre_model/.
# In a NEW scene (after regrid/recalibrate), teach 2-3 centres with scripts/center_teach.sh, then:  scripts/centre_fit.sh --check
cd "$(dirname "$0")/.." && source scripts/env.sh
exec python -m chess_player.calib.centre_model "$@"
