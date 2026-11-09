"""python -m chess_player.calib.simple_move <piece> <src> <dst> [--yes] [--dry-run]  (see motion/simple_pick_place.py: no corrections of any kind)"""
import argparse

from chess_player.motion.arm import FakeArm
from chess_player.motion.simple_pick_place import run


def square(text):
    """'a7', 'A7' and also '7a' (digit first) -> 'a7'."""
    t = text.strip().lower()
    if len(t) == 2 and t[0].isdigit() and t[1] in "abcdefgh":
        t = t[1] + t[0]
    if not (len(t) == 2 and t[0] in "abcdefgh" and t[1] in "12345678"):
        raise SystemExit(f"'{text}' is not a square (letter a-h then number 1-8, e.g. a7)")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece", choices=list("pnbrqk"))
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.src, a.dst = square(a.src), square(a.dst)
    if a.dry_run:
        run(a.piece, a.src, a.dst, arm=FakeArm())
        print("dry run only: nothing moved")
        return
    if not a.yes:
        input("Only the piece on the source square. Hand near the power switch. Enter = go: ")
    run(a.piece, a.src, a.dst)


if __name__ == "__main__":
    main()
