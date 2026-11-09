#!/usr/bin/env bash
# Teach the back-rank pieces ON THEIR OWN HOME SQUARES (crowded, like in a game), one after the other, and test each one right away.
#     bash ~/Desktop/chess/wooden-chess/scripts/teach_all.sh              # knight, queen, king, bishop, rook
#     bash ~/Desktop/chess/wooden-chess/scripts/teach_all.sh q k          # only some
# Needs: the camera window running, the board in the START position (all 32 pieces, centred), the arm connected.
# For each piece the teach tool asks you to type the piece name, then guides you through 4 stages (keys: a/d left-right, w/s away/toward you,
# r/f up/down, capital = 5 mm, Enter = ok, Esc = abort). Each teach is backed up automatically in data/teach/backups/.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source scripts/env.sh
declare -A SQ=( [n]=b1 [q]=d1 [k]=e1 [b]=c1 [r]=a1 )
declare -A DST=( [n]=c3 [q]=d3 [k]=e3 [b]=c3 [r]=a3 )
declare -A NAME=( [n]=knight [q]=queen [k]=king [b]=bishop [r]=rook )
TYPES=("$@"); [ ${#TYPES[@]} -eq 0 ] && TYPES=(n q k b r)
for t in "${TYPES[@]}"; do
  s=${SQ[$t]}; d=${DST[$t]}
  echo; echo "=================== ${NAME[$t]^^}: $s -> $d ==================="
  echo "The white ${NAME[$t]} must stand centred on $s (start position), facing you."
  read -r -p "Ready? [Enter = teach, s = skip this piece] " a; [ "$a" = "s" ] && continue
  python -m chess_player.calib.teach_stages --piece "$t" --square "$s" --dst "$d" --keys || { echo "teach stopped for $t"; continue; }
  echo; echo "Put the ${NAME[$t]} back on $s (centred), then it is tested: $s->$d and back."
  read -r -p "Ready for the test? [Enter = run, s = skip the test] " a; [ "$a" = "s" ] && continue
  python -m chess_player.calib.batch_test --moves "$s-$d $d-$s" --keep-going
done
echo; echo "Done. Paste the SUMMARY lines back."
