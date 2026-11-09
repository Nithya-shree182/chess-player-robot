"""Pick up and put back a pawn on EVERY square, in the order h1 g1 .. a1, h2 .. a8, with a pause after each (default 10 s).

  python -m chess_player.calib.pick_each [--piece p] [--pause 10] [--from h1] [--yes] [--dry-run]

Same movement as scripts/move.sh (motion/simple_pick_place.py: the taught centre of the square exactly, the piece's taught heights/gripper values, the
current temporary offsets in that file): down on the square, grip (fully-closed = empty hand: retry up to 3 tries), lift, put it back on the SAME square,
straight up, wait, next square. A pawn must stand on the square when the arm gets there: use the pause to put the next pawn on the next square. Ctrl+C freezes.
"""
import argparse
import time

import chess

from chess_player.calib.simple_move import square
from chess_player.motion.arm import Arm, FakeArm
from chess_player.motion.simple_pick_place import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece", default="p", choices=list("pnbrqk"))
    ap.add_argument("--pause", type=float, default=10.0)
    ap.add_argument("--from", dest="start")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    order = [chess.square_name(chess.square(f, r)) for r in range(8) for f in range(7, -1, -1)]
    if a.start:
        order = order[order.index(square(a.start)):]
    if a.dry_run:  # plan all squares on a simulated arm
        for s in order:
            run(a.piece, s, s, arm=FakeArm(), log=lambda *x: None)
        print(f"dry run: all {len(order)} squares plan fine; nothing moved")
        return
    if not a.yes:
        input(f"A {a.piece} on {order[0]} (and the next one on each square during the pause). Hand near the power switch. Enter = go: ")
    arm = Arm(use_sag_model=False)
    results = []
    try:
        for k, s in enumerate(order):
            print(f"\n[{k + 1}/{len(order)}] {s}", flush=True)
            _, ok = run(a.piece, s, s, arm=arm, home=(k == 0))
            results.append((s, ok))
            print(f"  {s}: {'picked and put back' if ok else 'NOT gripped'}; waiting {a.pause:g} s" + ("" if k + 1 < len(order) else " (last square)"), flush=True)
            if k + 1 < len(order):
                time.sleep(a.pause)
        from chess_player.motion.arm import look_pose
        from chess_player import config as C

        arm.move_to({**look_pose(), "gripper": C.GRIPPER_LOOK}, 4.0)
    except KeyboardInterrupt:
        print("\nstopped (arm frozen)")
    finally:
        bad = [s for s, ok in results if not ok]
        print(f"\n{len(results)} squares tried, {len(results) - len(bad)} picked and put back; not gripped: {bad if bad else 'none'}")
        arm.close()


if __name__ == "__main__":
    main()
