"""Per-move camera verification (the "verify" of observe -> act -> verify -> recover). READ-ONLY: it never changes the game state.

Before the robot moves (arm at home, out of the picture) and after it is home again, the board is read several times by YOLO and every
square gets a majority vote of what stands there: nothing, a white piece or a black piece. Only OCCUPANCY and COLOUR are used (the camera
confuses piece types), and only DIFFERENCES between the two looks are judged, so a piece YOLO always misses (e7) cancels out.

  expected change  (the move's source and destination, captures, castling ...) must show up: the source empty, the destination holding the
                   mover's colour. A destination on ranks 1-2 that reads "nothing" is not counted: tall pieces behind a pawn hide it.
  nothing expected on every other square: a square whose reading changed was disturbed (pushed, knocked over).
"""
import collections
import time

import chess

UNSURE = "?"


def _colour(sym):
    return "w" if sym.isupper() else "b"


def occupancy(board: chess.Board):
    return {sq: (None if board.piece_at(sq) is None else ("w" if board.piece_at(sq).color else "b")) for sq in chess.SQUARES}


class MoveVerifier:
    def __init__(self, cam, log=print, frames=6):
        self.cam = cam
        self.log = log
        self.frames = frames

    def observe(self, gap_s=0.35):
        """{square: None | 'w' | 'b' | '?'} by majority over several frames ('?' = no clear majority)."""
        from chess_player.vision import detect_board as D
        from chess_player.vision.ext_camera import view

        votes = collections.defaultdict(collections.Counter)
        for i in range(self.frames):
            if i:
                time.sleep(gap_s)
            frame = self.cam.grab(max_age_s=2.0)
            dets = D.detect_pieces(frame, view(frame), kq_height=False)
            for sq in chess.SQUARES:
                votes[sq][_colour(dets[sq][0]) if sq in dets else None] += 1
        out = {}
        for sq in chess.SQUARES:
            val, k = votes[sq].most_common(1)[0]
            out[sq] = val if k >= 0.6 * self.frames else UNSURE
        return out

    @staticmethod
    def judge(before, after, board_before: chess.Board, move: chess.Move):
        """-> (problems, disturbed). problems = [(square, expected, seen)] where an expected change did not happen;
        disturbed = [(square, seen_before, seen_after)] where something changed that should not have."""
        b2 = board_before.copy()
        b2.push(move)
        e_b, e_a = occupancy(board_before), occupancy(b2)
        problems, disturbed = [], []
        for sq in chess.SQUARES:
            ob, oa, hidden_rank = before[sq], after[sq], chess.square_rank(sq) <= 1
            if e_b[sq] != e_a[sq]:  # this square should have changed
                if oa == UNSURE or oa == e_a[sq]:
                    continue
                if e_a[sq] is not None and oa is None and hidden_rank:
                    continue  # a pawn on rank 1-2 hidden by the tall piece behind it: cannot judge
                problems.append((sq, e_a[sq], oa))
            elif UNSURE not in (ob, oa) and ob != oa:
                if (ob is None or oa is None) and hidden_rank:
                    continue  # flicker from occlusion on ranks 1-2
                if "w" not in (ob, oa):
                    continue  # only WHITE pieces are judged here: Black's near-edge readings flicker on a static board (ghosts 1-2 ranks off)
                disturbed.append((sq, ob, oa))
        return problems, disturbed

    @staticmethod
    def strays(before, after, board_before, move):
        """Squares where a piece of the mover's colour APPEARED although none is expected there (a piece that fell or was put down wrong)."""
        mover = "w" if board_before.turn else "b"
        b2 = board_before.copy()
        b2.push(move)
        e_a = occupancy(b2)
        return [sq for sq in chess.SQUARES if e_a[sq] is None and before[sq] is None and after[sq] == mover]


def describe(items, what):
    names = {"w": "a white piece", "b": "a black piece", None: "nothing", UNSURE: "an unclear reading"}
    out = []
    for it in items:
        if what == "problem":
            sq, exp, seen = it
            out.append(f"{chess.square_name(sq)}: expected {names[exp]}, sees {names[seen]}")
        else:
            sq, b, a = it
            out.append(f"{chess.square_name(sq)}: was {names[b]}, now {names[a]}")
    return "; ".join(out)
