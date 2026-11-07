"""Move-chain capture: tell it the moves you just made on the board, it updates its board and captures labelled shots.

  python -m chess_player.vision.capture_chain --reset            # board is at the start position
  python -m chess_player.vision.capture_chain "e4 e5"            # you moved e2-e4 and e7-e5 (SAN or UCI, e.g. e2e4)
  python -m chess_player.vision.capture_chain "e4" --shots 5

Moves are applied to the stored board (data/dataset_chain_state.json). A move that is not legal chess still works as a
plain 'piece from-square to to-square' (the target is captured), so you may rearrange freely: e.g. "e2e5" or "xe2" (remove).
The turn does not matter. After a capture the position is saved as pos<N> for the training split.
"""
import argparse
import json
import subprocess
import sys

import chess

from chess_player import config as C

STATE = C.DATA / "dataset_chain_state.json"


def load():
    d = json.loads(STATE.read_text()) if STATE.exists() else {"fen": chess.STARTING_FEN, "n": 10}
    return chess.Board(d["fen"]), d["n"]


def apply(board, tok):
    tok = tok.strip()
    if not tok:
        return
    if tok.startswith("x") and len(tok) == 3:  # remove the piece on a square
        board.remove_piece_at(chess.parse_square(tok[1:]))
        return
    try:
        board.push_san(tok)
        return
    except ValueError:
        pass
    try:
        m = chess.Move.from_uci(tok)
        p = board.piece_at(m.from_square)
        if p is None:
            raise ValueError(f"no piece on {chess.square_name(m.from_square)}")
        board.remove_piece_at(m.from_square)
        board.set_piece_at(m.to_square, p)  # replaces whatever stood there
        board.turn = not board.turn
        board.castling_rights = 0
        board.ep_square = None
    except ValueError as e:
        raise SystemExit(f"cannot apply '{tok}': {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("moves", nargs="?", default="")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--brightness", default="60,95,128,170,205")
    a = ap.parse_args()
    if a.reset:
        STATE.write_text(json.dumps({"fen": chess.STARTING_FEN, "n": 10}))
        print("board reset to the start position")
        if not a.moves:
            return
    board, n = load()
    for tok in a.moves.replace(",", " ").split():
        apply(board, tok)
    name = f"pos{n:02d}"
    print(board, "\n", board.fen(), f"\n{len(board.piece_map())} pieces on the board -> {name}")
    r = subprocess.run([sys.executable, "-m", "chess_player.vision.capture_safe", "--name", name, "--fen", board.fen(),
                        "--shots", str(a.shots), "--gap", "1.5", "--countdown", "4", "--brightness", a.brightness, "--jitter"])
    from chess_player.vision.capture_dataset import DS

    if list((DS / "meta").glob(f"{name}_*.json")):  # (the helper can exit non-zero at teardown even after saving shots)
        STATE.write_text(json.dumps({"fen": board.fen(), "n": n + 1}))


if __name__ == "__main__":
    main()
