"""Game state and move inference. python-chess is the source of truth; vision is a noisy sensor.

infer_move() scores every legal move against the evidence and returns the best one only when it clearly
beats the runner-up, otherwise None (the caller re-looks or asks the human).
"""
from dataclasses import dataclass

import chess


@dataclass
class Inference:
    move: chess.Move | None
    score: float
    runner_up: float
    ranked: list  # [(move, score)] best first
    reason: str = ""


def touched_squares(board: chess.Board, move: chess.Move) -> set:
    """Squares whose contents change when `move` is played."""
    sq = {move.from_square, move.to_square}
    if board.is_castling(move):
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rank = chess.square_rank(move.from_square)
        sq |= {chess.square(7 if kingside else 0, rank), chess.square(5 if kingside else 3, rank)}
    if board.is_en_passant(move):
        sq.add(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)))
    return sq


def score_position(board: chess.Board, dets: dict) -> float:
    """Agreement between one position and YOLO detections {square: (symbol, conf)} (see score_by_detections)."""
    s = 0.0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        d = dets.get(sq)
        if p is None:
            s += 0.5 if d is None else -d[1]
        elif d is None:
            s -= 0.5
        else:
            s += d[1] if d[0] == p.symbol() else (0.2 * d[1] if d[0].isupper() == p.color else -d[1])
    return s


def score_by_detections(board: chess.Board, dets: dict) -> list:
    """Agreement between each candidate position and YOLO detections {square: (symbol, conf)}.

    +conf for a square that matches the candidate (right piece or correctly empty), -conf for a mismatch.
    Empty squares with no detection count +0.5. Only squares some candidate touches matter for the ranking,
    but scoring the whole board keeps it simple and also reports the overall mismatch.
    """
    out = []
    for m in board.legal_moves:
        b = board.copy(stack=False)
        b.push(m)
        out.append((m, score_position(b, dets)))
    return sorted(out, key=lambda x: -x[1])


def illegal_move_seen(board: chess.Board, dets: dict, margin: float = 1.0):
    """When the camera shows a move that the rules don't allow (e.g. ignoring a check): (san-ish text, reason),
    else None. Only for telling the human; the game still only accepts legal moves."""
    stay = score_position(board, dets)
    best_legal = max((s for _, s in score_by_detections(board, dets)), default=float("-inf"))
    best = None
    for m in board.pseudo_legal_moves:
        if board.is_legal(m):
            continue
        b = board.copy(stack=False)
        b.push(m)
        s = score_position(b, dets)
        if best is None or s > best[1]:
            best = (m, s)
    if best is None or best[1] < max(stay, best_legal) + margin:
        return None
    m = best[0]
    p = board.piece_at(m.from_square)
    name = chess.piece_name(p.piece_type)
    text = f"{name} {chess.square_name(m.from_square)} to {chess.square_name(m.to_square)}"
    if board.is_check():
        why = "your king is in check from the " + " and ".join(
            f"{chess.piece_name(board.piece_at(c).piece_type)} on {chess.square_name(c)}" for c in board.checkers())
    else:
        why = "it would leave your king in check"
    legal = " ".join(board.san(x) for x in board.legal_moves)
    return text, f"{why}. Legal moves: {legal}"


def score_by_change(board: chess.Board, change: dict, noise: float | None = None) -> list:
    """Each move's touched squares should have changed and all others not.

    `change` is {square: score} from vision.detect_board.square_change. Scores are normalized by a noise
    level (median change over all squares) so the threshold adapts to lighting.
    """
    vals = sorted(change.values())
    noise = noise or max(1.0, vals[len(vals) // 2])
    z = {sq: v / noise for sq, v in change.items()}
    out = []
    for m in board.legal_moves:
        t = touched_squares(board, m)
        s = sum(min(z[sq], 6.0) for sq in t) / len(t) * 2 - sum(max(0.0, z[sq] - 2.0) for sq in z if sq not in t)
        out.append((m, s))
    return sorted(out, key=lambda x: -x[1])


def infer_move(board: chess.Board, ranked: list, margin: float) -> Inference:
    if not ranked:
        return Inference(None, 0, 0, [], "no legal moves")
    best, second = ranked[0], ranked[1] if len(ranked) > 1 else (None, float("-inf"))
    if best[1] - second[1] < margin:
        return Inference(None, best[1], second[1], ranked[:5],
                         f"ambiguous: {board.san(best[0])} vs {board.san(second[0])}")
    return Inference(best[0], best[1], second[1], ranked[:5])


def parse_human_move(board: chess.Board, text: str) -> chess.Move:
    """'e2e4', 'e4', 'Nf3', 'O-O', 'e7e8q' -> legal Move, or ValueError."""
    text = text.strip()
    try:
        m = chess.Move.from_uci(text.lower())
        if m in board.legal_moves:
            return m
    except ValueError:
        pass
    return board.parse_san(text)  # raises ValueError subclasses on illegal/ambiguous


def why_illegal(board: chess.Board) -> list[str]:
    """Plain-language reasons a position can't be played from (empty list = legal)."""
    st = board.status()
    out = []
    for color, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        ks = [chess.square_name(s) for s in board.pieces(chess.KING, color)]
        if not ks:
            out.append(f"no {name} king")
        elif len(ks) > 1:
            out.append(f"{len(ks)} {name} kings ({', '.join(ks)})")
        n = len(board.pieces(chess.PAWN, color))
        if n > 8:
            out.append(f"{n} {name} pawns")
    back = [chess.square_name(s) for s in board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
            if chess.square_rank(s) in (0, 7)]
    if back:
        out.append(f"pawn on the first/last rank ({', '.join(back)})")
    if st & chess.STATUS_OPPOSITE_CHECK:
        out.append(f"{'black' if board.turn else 'white'} is in check but it's {'white' if board.turn else 'black'}'s turn")
    if st & chess.STATUS_TOO_MANY_CHECKERS or st & chess.STATUS_IMPOSSIBLE_CHECK:
        out.append("an impossible check")
    if not out and st != chess.STATUS_VALID:
        out.append(f"invalid ({st!r})")
    return out


def prune_impossible(seen: dict) -> tuple[dict, list[str]]:
    """Drop camera readings that cannot be real: a side has at most 16 pieces and 8 pawns, so the lowest-confidence extras are phantoms
    (a tall piece's head read as a pawn one square away, a duplicate box). `seen` = {square: (symbol, conf)}.
    Returns (pruned copy, dropped square names)."""
    seen = dict(seen)
    dropped = []
    for white in (True, False):
        def mine(only_pawns=False):
            return sorted((conf, sq) for sq, (sym, conf) in seen.items()
                          if sym.isupper() == white and (not only_pawns or sym.lower() == "p"))

        while len(mine(True)) > 8:
            _, sq = mine(True)[0]
            dropped.append(chess.square_name(sq))
            del seen[sq]
        while len(mine()) > 16:
            _, sq = mine()[0]
            dropped.append(chess.square_name(sq))
            del seen[sq]
    return seen, dropped


def start_mismatch(seen: dict, trust_conf: float = 0.8) -> list[str]:
    """Why the raw camera reading `seen` ({square: (symbol, conf)}) is NOT the standard starting setup (empty list = it is).

    Chess logic, independent of the unreliable far side of the picture (Black's pieces stand at the camera's near edge: their squares and
    types get misread, e.g. the e7 pawn seen on e6, the back rank cut off): White moves first, so if White's half still holds the
    standard setup and the middle ranks 3-5 are empty, nothing has been played yet, whatever Black's side looks like. Required:
      - White's ranks 1-2: all 16 squares hold a white piece (one may be missed), and no confident white piece has another type;
      - ranks 3-5: no confident piece (a real move leaves one there);
      - Black's ranks 7-8: at least 8 black pieces seen (so the board is not simply empty on that side)."""
    start = chess.Board()
    why = []
    white_missing = [chess.square_name(sq) for sq, p in start.piece_map().items()
                     if p.color == chess.WHITE and not (sq in seen and seen[sq][0].isupper())]
    if len(white_missing) > 1:
        why.append("white start squares not occupied: " + " ".join(white_missing))
    middle = [chess.square_name(sq) for sq, (sym, conf) in seen.items() if chess.square_rank(sq) in (2, 3, 4) and conf >= 0.5]
    if middle:
        why.append("pieces in the middle ranks: " + " ".join(middle))
    for sq, (sym, conf) in seen.items():
        sp = start.piece_at(sq)
        if sp is not None and sp.color == chess.WHITE and conf >= trust_conf and sym != sp.symbol():
            why.append(f"{chess.square_name(sq)} is a confident {sym} (start has {sp.symbol()})")
    black_far = sum(1 for sq, (sym, conf) in seen.items() if sym.islower() and chess.square_rank(sq) in (6, 7))
    if black_far < 8:
        why.append(f"only {black_far} black pieces seen on ranks 7-8")
    return why


def looks_like_start(board: chess.Board, min_pieces: int = 30, seen: dict | None = None, trust_conf: float = 0.8) -> bool:
    """The cameras show the starting setup, so the position IS the standard start (king e1, queen d1, ..., White to move)
    whatever piece types the cameras guess.

    With `seen` ({square: (symbol, confidence)}, the raw camera reading) the decision is by WHERE the pieces stand and which colour
    they are, which the camera gets right even on a grainy picture: nearly all 32 start squares hold a piece of the right colour
    (up to 3 may be missed, only 1 of them outside Black's back rank, which is the part cut off by the frame), and no confident piece
    stands anywhere else (a faint reading, like a phantom pawn, is ignored). Piece TYPES only veto it for White's pieces and
    Black's pawns (read reliably): a confident queen on d7 means this is not the standard setup. Black's back-rank types are
    often misread with high confidence, so they never veto.
    Without `seen`, `board` itself is judged: every piece on its side's first two ranks, at least `min_pieces` of them."""
    start = chess.Board()
    if seen:
        return not start_mismatch(seen, trust_conf)
    pm = board.piece_map()
    if len(pm) < min_pieces:
        return False
    return all(start.piece_at(sq) is not None and start.piece_at(sq).color == p.color for sq, p in pm.items())


def merge_seen(game: chess.Board, seen: dict, type_conf=None, votes=None) -> tuple[chess.Board, list[str]]:
    """Combine what the camera sees with what we already know, using chess sense. `seen` = {square: (symbol, conf)}.

    - occupancy and colour come from the camera; piece TYPES persist (a black piece that stays on its square is the
      same piece) unless the cameras call it another type with confidence >= type_conf (default SYNC_TYPE_CONF),
      or both cameras see it (`votes` = {square: [webcam symbol, gripper-cam symbol]}) and neither says the known
      type (then the known type was a misreading; the stronger camera's type is used, marked unsure);
    - a piece appearing where none of its colour was, while exactly one piece of that colour vanished, is that
      piece moved: it keeps its type;
    - only otherwise the camera's type is used, and a colour never gets two kings (extras are marked unsure).
    Returns (board with game.turn, unsure square names)."""
    if type_conf is None:
        from chess_player import config as C

        type_conf = C.SYNC_TYPE_CONF
    prev = game.piece_map()
    new = chess.Board(None)
    new.turn = game.turn
    unsure = []
    vanished = {c: [sq for sq, p in prev.items() if p.color == c and (sq not in seen or seen[sq][0].isupper() != c)]
                for c in (chess.WHITE, chess.BLACK)}
    appeared = []
    for sq, (sym, conf) in seen.items():
        color = sym.isupper()
        p = prev.get(sq)
        if p is not None and p.color == color:
            # same colour still there: same piece, unless the cameras are sure it's another type (piece swapped)
            both_reject = votes is not None and len(votes.get(sq, [])) == 2 and p.symbol() not in votes[sq]
            if p.piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):  # impossible: an old misreading
                sym = next((s for s in [sym] + (votes or {}).get(sq, []) if s.lower() != "p"), sym)
                both_reject = sym != p.symbol()
            if sym != p.symbol() and (conf >= type_conf or both_reject):
                new.set_piece_at(sq, chess.Piece.from_symbol(sym))
                if conf < type_conf:
                    unsure.append(chess.square_name(sq))
            else:
                new.set_piece_at(sq, p)
        else:
            appeared.append((sq, sym, conf))
    for sq, sym, conf in sorted(appeared, key=lambda a: -a[2]):
        color = sym.isupper()
        if len(vanished[color]) == 1:  # it moved here from where one piece of its colour disappeared
            new.set_piece_at(sq, prev[vanished[color].pop()])
        else:
            new.set_piece_at(sq, chess.Piece.from_symbol(sym))
            unsure.append(chess.square_name(sq))
    for color in (chess.WHITE, chess.BLACK):
        kings = list(new.pieces(chess.KING, color))
        old_king = next((sq for sq, p in prev.items() if p == chess.Piece(chess.KING, color)), None)
        if not kings and old_king is not None:
            # a king never leaves the board: a misread turned it into something else. Put it back where it was if
            # that square still holds one of its pieces, else on the piece the camera calls a king.
            back = old_king if (new.piece_at(old_king) and new.piece_at(old_king).color == color) else next(
                (sq for sq, p in new.piece_map().items() if p.color == color and seen.get(sq, ("",))[0].lower() == "k"), None)
            if back is not None:
                new.set_piece_at(back, chess.Piece(chess.KING, color))
                unsure.append(chess.square_name(back))
        if len(kings) > 1:
            # keep the one the camera most confidently calls a king; among equals, one we already knew as a king
            keep = max(kings, key=lambda k: (seen.get(k, ("", 0))[0].lower() == "k", seen.get(k, ("", 0))[1],
                                             prev.get(k) == chess.Piece(chess.KING, color)))
            for k in kings:
                if k != keep:
                    new.set_piece_at(k, chess.Piece(chess.QUEEN if not new.pieces(chess.QUEEN, color) else chess.BISHOP, color))
                    unsure.append(chess.square_name(k))
    new.castling_rights = new.clean_castling_rights()
    return new, sorted(set(unsure))
