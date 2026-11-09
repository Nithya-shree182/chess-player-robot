"""Stockfish via python-chess. One engine process for the whole game."""
import random

import chess
import chess.engine

from chess_player import config as C


class Engine:
    """Stockfish that restarts itself if it dies (e.g. after being handed an impossible position)."""

    def __init__(self, path=C.STOCKFISH, skill_level=None):
        self.path, self.skill = path, skill_level
        self._start()

    def _start(self):
        self.e = chess.engine.SimpleEngine.popen_uci(str(self.path))
        if self.skill is not None:  # 0..20, lower = weaker (nice for demos)
            self.e.configure({"Skill Level": int(self.skill)})

    def _call(self, fn, board, *args):
        if not board.is_valid():
            raise ValueError(f"not a legal chess position ({board.status().name}): fix it in Edit position")
        try:
            return fn(board, *args)
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError):
            self._start()
            return fn(board, *args)

    def best_move(self, board: chess.Board, seconds=C.ENGINE_TIME_S) -> chess.Move:
        """Stockfish's best move. Stockfish is deterministic (same position -> same move), so every game would open the same way. With
        ENGINE_VARIETY_CP > 0 it looks at the best few moves and picks randomly among those no worse than that many centipawns behind the
        best (nearly full strength, but different games). Forced lines and mates are never varied."""
        if C.ENGINE_VARIETY_CP > 0 and not board.is_game_over():
            try:
                infos = self._call(lambda b, t: self.e.analyse(b, chess.engine.Limit(time=t), multipv=C.ENGINE_VARIETY_LINES), board, seconds)
                scored = []
                for i in infos:
                    sc = i["score"].relative
                    if sc.is_mate() or not i.get("pv"):
                        scored = []
                        break
                    scored.append((sc.score(), i["pv"][0]))
                if len(scored) > 1:
                    best = max(s for s, _ in scored)
                    return random.choice([m for s, m in scored if s >= best - C.ENGINE_VARIETY_CP])
            except Exception:
                pass  # any trouble: fall back to the plain best move below
        return self._call(lambda b, t: self.e.play(b, chess.engine.Limit(time=t)).move, board, seconds)

    def evaluate(self, board: chess.Board, seconds=0.3) -> str:
        """Human-readable eval from White's side, e.g. '+0.35' or 'mate in 3'."""
        info = self._call(lambda b, t: self.e.analyse(b, chess.engine.Limit(time=t)), board, seconds)
        s = info["score"].white()
        return f"mate in {s.mate()}" if s.is_mate() else f"{s.score() / 100:+.2f}"

    def close(self):
        self.e.quit()
