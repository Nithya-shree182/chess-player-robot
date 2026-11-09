"""Reliability test: play a list of single-piece moves with the real arm, check each one with the camera, print a compact table.

  python -m chess_player.calib.batch_test                                   # the pawn test (8 moves), start position
  python -m chess_player.calib.batch_test --moves "b1-c3 c3-b1 g1-f3 f3-g1"  # knights (the piece type is read from the start position)
  python -m chess_player.calib.batch_test --moves "d2-d4 d4-d2" --repeat 3   # the same moves 3 times

Needs: scripts/livecam.py running, the board in the START position (every piece centred), the arm connected. Each move runs
calib/pick_place.py (taught numbers + camera centring + the shift search + learned offsets), waits for the arm to leave the picture, reads
the board 6 times and checks the source and destination squares (a piece still on the source, or nothing on the destination, in more than half
of the frames = the move did NOT happen; "nothing" is ignored on ranks 1-2, where pieces behind hide the pawn). Piece TYPES are not judged (the camera confuses king/queen). Stops at the first failure unless
--keep-going. The full output of every move is saved in data/batch_logs/. Before/after pictures of every move are saved next to the log.
Paste only the SUMMARY lines back.
"""
import argparse
import os
import re
import subprocess
import sys
import time

import chess

from chess_player import config as C

SHOT = "http://127.0.0.1:8771/shot"


def grab_bytes():
    import urllib.request

    return urllib.request.urlopen(SHOT, timeout=5).read()


def feed_frozen(n=4, gap=0.6):
    """True when the camera window serves the SAME picture every time: the webcam stopped delivering (the window keeps showing its
    last frame forever). A live feed always differs a little (sensor noise)."""
    try:
        shots = []
        for _ in range(n):
            shots.append(grab_bytes())
            time.sleep(gap)
    except OSError:
        return True
    return len(set(shots)) == 1


DEFAULT_MOVES = "d2-d4 d4-d2 g2-g4 g4-g2 a2-a4 a4-a2 e2-e4 e4-e2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", default=DEFAULT_MOVES, help='e.g. "d2-d4 d4-d2"; pieces are moved square to square (no chess rules)')
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--keep-going", action="store_true", help="do not stop at the first failure")
    ap.add_argument("--no-verify", action="store_true", help="skip the camera check")
    a = ap.parse_args()

    env = dict(os.environ, CHESS_CAMERA_URL="http://127.0.0.1:8771/video")
    log_dir = C.DATA / "batch_logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / time.strftime("%Y%m%d-%H%M%S.log")
    board = chess.Board()
    moves = [m.split("-") for m in a.moves.split()] * a.repeat
    rows, clean = [], 0

    def say(s):
        print(s, flush=True)
        with open(log_file, "a") as f:
            f.write(s + "\n")

    say(f"batch of {len(moves)} moves; details in {log_file}")
    if not a.no_verify and feed_frozen():
        say("STOPPED before moving: the camera picture is FROZEN or unreachable (identical frames). Check that scripts/livecam.py runs, the webcam "
            "is plugged in and readable (ls -l /dev/video4; sudo setfacl -m u:$USER:rw /dev/video4 /dev/video5), then run again.")
        return
    for i, (src, dst) in enumerate(moves, 1):
        s_sq, d_sq = chess.parse_square(src), chess.parse_square(dst)
        piece = board.piece_at(s_sq)
        if piece is None or board.piece_at(d_sq) is not None:
            say(f"{i:2d} {src}->{dst}: SKIPPED (the test board has no piece on {src}, or {dst} is not empty)")
            continue
        rest = board.copy()
        rest.remove_piece_at(s_sq)
        others = [f"{chess.square_name(q)}:{p.symbol()}" for q, p in rest.piece_map().items()]
        t0 = time.time()
        stem = log_dir / f"{log_file.stem}_{i:02d}_{src}{dst}"
        try:
            stem.with_name(stem.name + "_before.jpg").write_bytes(grab_bytes())
        except OSError:
            pass
        r = subprocess.run([sys.executable, "-m", "chess_player.calib.pick_place", src, dst, "--piece", piece.symbol().lower(),
                            "--others", *others, "--yes"], env=env, capture_output=True, text=True, timeout=400)
        out = r.stdout + r.stderr
        with open(log_file, "a") as f:
            f.write(f"\n===== {src}->{dst} =====\n{out}\n")
        shifts = len(re.findall(r"shifting the pick point", out))
        failed = "GraspFailed" in out or "Traceback" in out
        verdict = "ARM FAILED (nothing gripped)" if failed else "arm ok"
        if not failed and not a.no_verify:
            time.sleep(7)
            try:
                stem.with_name(stem.name + "_after.jpg").write_bytes(grab_bytes())
            except OSError:
                pass
            if feed_frozen(n=3, gap=0.5):
                say(f"{i:2d} {piece.symbol().upper()} {src}->{dst}: arm ok | camera FROZEN, cannot verify | {time.time() - t0:.0f}s")
                rows.append((src, dst, False, shifts))
                if not a.keep_going:
                    say("STOPPED: the camera feed froze during the test (replug the webcam / restart livecam).")
                    break
                continue
            after = board.copy()
            after.remove_piece_at(s_sq)
            after.set_piece_at(d_sq, piece)
            v = subprocess.run([sys.executable, "-m", "chess_player.vision.eval_board", "--fen", after.fen(), "--frames", "6"], env=env,
                               capture_output=True, text=True, timeout=180).stdout
            bad = []
            for line in v.splitlines():
                m = re.match(rf"\s+({src}|{dst})\s+\S+\s+->\s+(.*)", line)
                if m:  # only "ghost" on the source or "missed" on the destination, in more than half of the 6 frames
                    kind = "ghost" if m.group(1) == src else "missed"
                    if kind == "missed" and chess.square_rank(d_sq) <= 1:
                        continue  # ranks 1-2: the tall piece behind hides the pawn from the low camera, a miss there is not evidence
                    n = sum(int(x) for x in re.findall(rf"{kind}(?: \w)? x(\d+)", m.group(2)))
                    if n > 3:
                        bad.append(f"{m.group(1)} {kind} x{n}")
            verdict += " | camera: " + ("MOVE NOT SEEN " + str(bad) if bad else "ok")
            failed = failed or bool(bad)
        if not failed:
            board = after if not a.no_verify else board
            if a.no_verify:
                board.remove_piece_at(s_sq)
                board.set_piece_at(d_sq, piece)
            clean += 1
        note = f"{shifts} shift(s)" if shifts else "first try"
        say(f"{i:2d} {piece.symbol().upper()} {src}->{dst}: {verdict} | {note} | {time.time() - t0:.0f}s")
        rows.append((src, dst, not failed, shifts))
        if failed and not a.keep_going:
            say("STOPPED at the first failure. Last lines of the arm output:\n  " + "\n  ".join(out.strip().splitlines()[-6:]))
            break
    first = sum(1 for _, _, ok, sh in rows if ok and sh == 0)
    say(f"SUMMARY: {clean}/{len(rows)} moves clean, {first} on the first try, {sum(sh for *_, sh in rows)} shifted retries in total")
    try:
        import json

        learned = json.loads((C.DATA / "pick_offsets.json").read_text())
        say("learned offsets (mm, x/y): " + "; ".join(f"{k}:{s} {v[0]:+.1f}/{v[1]:+.1f}" for k, d in learned.items() for s, v in d.items()))
    except (OSError, ValueError):
        pass


if __name__ == "__main__":
    main()
