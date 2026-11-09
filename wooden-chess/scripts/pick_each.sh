#!/usr/bin/env bash
# Pick up and put back a pawn on every square in turn (h1 g1 .. a1, h2 .. a8), 10 s pause after each. Options: --pause 5  --from d3  --piece p  --yes
cd "$(dirname "$0")/.." && source scripts/env.sh
exec python -m chess_player.calib.pick_each "$@"
