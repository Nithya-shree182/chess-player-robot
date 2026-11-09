"""Check how well YOLO reads the REAL board against a position you know (default: the standard start).

  python -m chess_player.vision.eval_board                 # start position, 12 frames
  python -m chess_player.vision.eval_board --fen "<FEN>"   # any other known position
  python -m chess_player.vision.eval_board --brightness 60,95,128,170   # repeat at several camera brightness values

Per frame and per square it compares the detection with the truth, then prints: accuracy for White and Black separately
(occupancy, colour, type), every wrong square with what it was read as and how often, and the confidence of the right and
the wrong readings. Read-only: it never saves anything or changes the game state. Needs scripts/livecam.py running.
"""
import argparse
import collections
import subprocess
import time

import chess

from chess_player import config as C


def read_frames(cam, n, gap=0.4):
    from chess_player.vision import detect_board as D
    from chess_player.vision.ext_camera import view

    out = []
    for i in range(n):
        if i:
            time.sleep(gap)
        frame = cam.grab(max_age_s=2.0)
        out.append(D.detect_pieces(frame, view(frame)))
    return out


def evaluate(truth, frames, title=""):
    n = len(frames)
    st = {c: collections.Counter() for c in (chess.WHITE, chess.BLACK)}
    wrong = collections.defaultdict(collections.Counter)  # square -> what was read
    conf_ok, conf_bad = [], []
    for dets in frames:
        for sq in chess.SQUARES:
            want = truth.piece_at(sq)
            got = dets.get(sq)
            if want is None:
                if got:
                    wrong[sq]["ghost " + got[0]] += 1
                continue
            col = want.color
            st[col]["n"] += 1
            if not got:
                wrong[sq]["missed"] += 1
                continue
            sym, conf = got
            read = chess.Piece.from_symbol(sym)
            st[col]["occ"] += 1
            st[col]["colour"] += read.color == want.color
            ok = sym == want.symbol()
            st[col]["type"] += ok
            (conf_ok if ok else conf_bad).append(conf)
            if not ok:
                wrong[sq][sym] += 1
    print(f"\n=== {title}{n} frames ===")
    for col, name in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        s = st[col]
        if s["n"]:
            print(f"{name}: seen {100 * s['occ'] / s['n']:.0f}%   right colour {100 * s['colour'] / s['n']:.0f}%   "
                  f"right type {100 * s['type'] / s['n']:.0f}%   (of {s['n']} piece-readings)")
    if conf_ok:
        print(f"confidence when right: mean {sum(conf_ok) / len(conf_ok):.2f};  "
              + (f"when wrong: mean {sum(conf_bad) / len(conf_bad):.2f} ({len(conf_bad)} readings)" if conf_bad else "never wrong"))
    if wrong:
        print("wrong squares (truth -> what it read, times out of frames):")
        for sq in sorted(wrong, key=lambda s: -sum(wrong[s].values())):
            t = truth.piece_at(sq)
            reads = ", ".join(f"{k} x{v}" for k, v in wrong[sq].most_common())
            print(f"  {chess.square_name(sq)}  {t.symbol() if t else '.'}  ->  {reads}")
    else:
        print("every square right in every frame")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", default=chess.STARTING_FEN)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--brightness", help="comma list of camera brightness values to try, e.g. 60,95,128,170")
    a = ap.parse_args()
    from chess_player.vision.ext_camera import ExtCamera
    from chess_player.vision.capture_safe import BASE_BRIGHTNESS, DEV, set_brightness

    truth = chess.Board(a.fen)
    cam = ExtCamera()
    try:
        if a.brightness:
            print("note: this talks to the camera device directly; stop the camera window first if it fails")
            for b in [int(v) for v in a.brightness.split(",")]:
                set_brightness(b)
                time.sleep(2.5)
                evaluate(truth, read_frames(cam, a.frames), f"brightness {b}: ")
            set_brightness(BASE_BRIGHTNESS)
        else:
            evaluate(truth, read_frames(cam, a.frames))
    finally:
        cam.close()


if __name__ == "__main__":
    main()
