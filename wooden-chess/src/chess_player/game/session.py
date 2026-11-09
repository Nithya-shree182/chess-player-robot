"""Everything a game needs, behind a few methods that both main.py and the Claude agent call.

Vision mode is "yolo" when data/yolo/best.pt exists, else "change" (image difference between looks).
With --dry-run there is no arm and no camera: the human types moves and the arm is simulated.
"""
import copy
import time

import chess
import numpy as np

from chess_player import config as C
from chess_player.game import state
from chess_player.game.engine import Engine
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.ik import IK, Unreachable
from chess_player.motion.pick_place import MoveNotVerified, PickPlace, Slots, Transfer
from chess_player.motion.wooden_pick_place import WoodenPickPlace


class Session:
    def __init__(self, dry_run=False, robot_color=C.ROBOT_COLOR, skill=None, use_camera=True, log=print):
        self.log = log
        self.board = chess.Board()
        self.robot_color = chess.BLACK if robot_color == "black" else chess.WHITE
        self.engine = Engine(skill_level=skill)
        self.slots = None  # set once the board->robot transform is known
        self.dry_run = dry_run
        if dry_run:
            from chess_player.motion.arm import FakeArm

            self.arm = FakeArm()
            T = BoardToRobot.load() if C.BOARD_TO_ROBOT.exists() else BoardToRobot.assumed(140, robot_side=robot_color)
        else:
            from chess_player.motion.arm import Arm

            self.arm = Arm()
            T = BoardToRobot.load()
        self.pp = (WoodenPickPlace if C.USE_RECIPE_GRASP else PickPlace)(self.arm, IK(), T, verify_grasp=not dry_run, log=log)
        self.slots = Slots(T)
        self.cam = None  # wrist camera (fallback vision)
        self.ext = None  # fixed third-person camera (main vision when its corners are calibrated)
        self.watcher = None
        from chess_player.vision.detect_board import available

        self.vision = "yolo" if available() else "change"
        if use_camera:  # cameras + YOLO work with a simulated arm too
            if C.EXT_CORNERS.exists():
                from chess_player.vision.ext_camera import BoardWatcher, ExtCamera

                try:
                    self.ext = ExtCamera()
                    if hasattr(self.pp, "locator"):  # centre every pick on where the piece really stands (arm at home = out of view)
                        from chess_player.vision.piece_locator import PieceLocator

                        self.pp.locator = PieceLocator(self.ext, log=log)
                    if self.vision == "yolo":
                        self.watcher = BoardWatcher(self.ext)
                except RuntimeError as e:
                    log(f"third-person camera unavailable ({e}); using the wrist camera")
            if self.ext is None and C.MANUAL_CORNERS.exists():  # gripper-cam vision only with its own corners
                from chess_player.vision.camera import Camera

                try:
                    self.cam = Camera()
                except RuntimeError as e:
                    log(f"no camera for vision ({e})")
        # a camera used ONLY to verify each move (read-only), also in terminal mode where the game itself runs without camera vision
        self.verifier = None
        self.verify_cam = self.ext
        self._own_verify_cam = False
        if self.verify_cam is None and not dry_run and self.vision == "yolo" and C.EXT_CORNERS.exists():
            try:
                from chess_player.vision.ext_camera import ExtCamera

                self.verify_cam = ExtCamera()
                self._own_verify_cam = True
                if hasattr(self.pp, "locator") and self.pp.locator is None:
                    from chess_player.vision.piece_locator import PieceLocator

                    self.pp.locator = PieceLocator(self.verify_cam, log=log)
            except RuntimeError as e:
                log(f"move verification camera unavailable ({e}): moves will NOT be checked by the camera")
        if self.verify_cam is not None and not dry_run and C.VERIFY_MOVES:
            from chess_player.game.verify import MoveVerifier

            self.verifier = MoveVerifier(self.verify_cam, log=log)
        self.disturbed = []  # [(square, was, now)] from the last verified move: pieces that changed although nothing should have
        self.parked = False
        self.illegal_seen = None  # (move text, why) when the camera shows a move the rules don't allow
        self.refine = None  # optional: webcam reading -> reading improved by the gripper camera (set by the UI)
        self.last_frame = None
        self.last_view = None

    # ---------------------------------------------------------------- perception
    @property
    def has_vision(self):
        return self.ext is not None or self.cam is not None

    def park(self):
        """Arm to the look/home pose (wrist cam sees the board, and it's out of the way of the fixed camera)."""
        if not self.parked:
            self.pp.go_look()
            self.parked = True
            if self.watcher:
                self.watcher.clear()  # only readings taken with the arm parked count

    def look(self):
        """Park the arm and capture. Returns (frame, BoardView) or (None, None) without a camera."""
        self.park()
        if self.ext is not None:
            from chess_player.vision.ext_camera import view

            time.sleep(0.5)
            frames = [self.ext.grab()] + [time.sleep(0.15) or self.ext.grab() for _ in range(2)]
            frame = np.median(np.stack(frames), axis=0).astype(np.uint8)
            return frame, view(frame)
        if self.cam is None:
            return None, None
        from chess_player.vision.aruco_homography import locate_board

        for attempt in range(3):
            frame = self.cam.grab_median(3)
            try:
                return frame, locate_board(frame)
            except RuntimeError as e:
                self.log(f"  board not located ({e}), retrying")
                time.sleep(0.5)
        raise RuntimeError("can't locate the board: are the markers visible from the look pose?")

    def snapshot(self):
        """Remember the current view as the 'before' image for the next move."""
        self.last_frame, self.last_view = self.look()

    def rank_moves(self, frame, view):
        from chess_player.vision import detect_board as D

        if self.vision == "yolo":
            return state.score_by_detections(self.board, D.detect_pieces(frame, view)), 1.0
        change = D.square_change(self.last_frame, frame, self.last_view, view)
        return state.score_by_change(self.board, change), 1.5

    def poll_human_move(self, n_stable=C.WATCH_STABLE_N):
        """Non-blocking, watcher only: the human's move once the board has looked the same for n_stable
        readings (hands gone) and that reading is the current position plus one legal move. Else None."""
        if self.watcher is None:
            return None
        dets = self.watcher.stable(n_stable)
        if dets is None:
            return None
        self.illegal_seen = None
        if self.refine is not None:  # the gripper camera's opinion where the webcam is unsure
            dets = self.refine(dets)
        inf = state.infer_move(self.board, state.score_by_detections(self.board, dets), 1.0)
        # the winning move must also beat "nothing happened" (the current position as is)
        stay = state.score_position(self.board, dets)
        if inf.move is not None and inf.score > stay + 1.0:
            return inf
        self.illegal_seen = state.illegal_move_seen(self.board, dets)  # just to explain why nothing is accepted
        return None

    def see_human_move(self):
        """Infer the move the human just made (on demand). Returns state.Inference (move None if unsure)."""
        frame, view = self.look()
        if frame is None:
            return state.Inference(None, 0, 0, [], "no camera")
        if self.vision == "change" and self.last_frame is None:
            return state.Inference(None, 0, 0, [], "no reference image: snapshot() before the human moves")
        ranked, margin = self.rank_moves(frame, view)
        inf = state.infer_move(self.board, ranked, margin)
        if inf.move is not None:
            self.last_frame, self.last_view = frame, view
        return inf

    def verify_position(self):
        """After a robot move: does the board look like self.board? Returns (ok, details)."""
        frame, view = self.look()
        if frame is None:
            return True, "no camera: not verified"
        from chess_player.vision import detect_board as D

        if self.vision == "yolo":
            # Only what the robot's move changed is checked (from / to squares, castling rook, en-passant pawn):
            # empty, or a piece of the right colour, in a steady reading of both cameras. Piece TYPES elsewhere are
            # not compared: YOLO misreads them (and pieces hidden behind others) without anything being wrong.
            self.last_frame, self.last_view = frame, view
            if not self.board.move_stack:
                return True, "no move to check"
            last = self.board.peek()
            prev = self.board.copy()
            prev.pop()
            touched = state.touched_squares(prev, last)
            dets, t_end = None, time.time() + 5.0
            while dets is None and time.time() < t_end and self.watcher is not None:
                time.sleep(0.3)
                dets = self.watcher.stable(C.WATCH_STABLE_N)
            if dets is None:
                dets = D.detect_pieces(frame, view)
            if self.refine is not None:
                dets = self.refine(dets)
            wrong = []
            for sq in touched:
                p, d = self.board.piece_at(sq), dets.get(sq)
                if (p is None) != (d is None) or (p and d and p.color != d[0].isupper()):
                    what = lambda x: "empty" if x is None else ("a white piece" if x else "a black piece")
                    wrong.append(f"{chess.square_name(sq)} should be {what(None if p is None else p.color)}, "
                                 f"the cameras see {what(None if d is None else d[0].isupper())}")
            return not wrong, "; ".join(wrong) or "matches"
        # change mode: the last robot move's squares should have changed, nothing else much
        last = self.board.peek()
        prev = self.board.copy()
        prev.pop()
        change = D.square_change(self.last_frame, frame, self.last_view, view)
        ranked = state.score_by_change(prev, change)
        self.last_frame, self.last_view = frame, view
        ok = ranked[0][0] == last
        return ok, "matches" if ok else f"looks more like {prev.san(ranked[0][0])} than {prev.san(last)}"

    # ---------------------------------------------------------------- decisions and actions
    def human_to_move(self):
        return self.board.turn != self.robot_color and not self.board.is_game_over()

    def push_human(self, move: chess.Move):
        if move not in self.board.legal_moves:
            raise ValueError(f"illegal move {move.uci()}")
        self.board.push(move)

    def engine_move(self) -> chess.Move:
        return self.engine.best_move(self.board)

    def check(self, move: chess.Move):
        """Feasibility report for `move` without moving (IK, tilt, travel height, collisions per transfer)."""
        return self.pp.check(self.board, move, self.slots)

    def execute(self, move: chess.Move, recover=None):
        """Physically play `move`, then push it. Unreachable is raised before anything moves (board unchanged).
        GraspFailed can happen mid-move: the board isn't pushed, the human finishes the move by hand and calls
        push_robot_by_hand()."""
        saved = copy.deepcopy(self.slots)
        if self.watcher:
            self.watcher.paused = True
        self.parked = False
        before = None
        if self.verifier is not None:  # observe: the arm is at home, out of the picture
            try:
                before = self.verifier.observe()
            except Exception as e:  # stale stream ...
                self.log(f"camera check skipped, could not look before the move ({e})")
        try:
            transfers = self.pp.play(self.board, move, self.slots, recover=recover)  # ends at the look pose
            self.parked = True
            if before is not None:  # verify (+ recover); raises MoveNotVerified when it cannot be fixed
                self._verify_and_fix(move, before, saved, recover)
        except Unreachable:
            self.slots = saved
            raise
        finally:
            if self.watcher:
                self.watcher.clear()
                self.watcher.paused = False
        self.board.push(move)
        return [t.label for t in transfers]

    MAX_FIXES = 2

    def _verify_and_fix(self, move: chess.Move, before, saved_slots, recover):
        """act -> VERIFY -> RECOVER. Look again with the arm home. If the move did not show up, fix it (at most MAX_FIXES times) when the
        fix is simple and safe: the piece is still on its source -> play the move again; it fell elsewhere -> pick it up from there and put
        it on the destination. Captures, castling and promotion are never retried automatically (they need the human). The game state
        (self.board) is never touched here: it is pushed only after this returns."""
        from chess_player.game.verify import describe

        v = self.verifier
        problems = []
        for attempt in range(self.MAX_FIXES + 1):
            time.sleep(4.0)  # the arm leaves the picture and the light settles
            try:
                after = v.observe()
            except Exception as e:
                self.log(f"camera check skipped, could not look after the move ({e})")
                return
            problems, disturbed = v.judge(before, after, self.board, move)
            self.disturbed = [(chess.square_name(sq), b, a) for sq, b, a in disturbed]
            if disturbed:
                self.log("  camera: other pieces changed during the move, please check: " + describe(disturbed, "disturbed"))
            if not problems:
                self.log("  camera: move verified (source empty, destination holds the right colour)")
                return
            self.log("  camera: the move did NOT show up: " + describe(problems, "problem"))
            if attempt == self.MAX_FIXES or not self._fix(move, before, after, saved_slots, recover):
                break
        raise MoveNotVerified(f"{move.uci()}: the camera does not show the move: {describe(problems, 'problem')}")

    def _fix(self, move: chess.Move, before, after, saved_slots, recover):
        """One automatic recovery step. True = something was done (look again), False = nothing safe to do."""
        if self.board.is_capture(move) or self.board.is_castling(move) or move.promotion:
            self.log("  camera: captures, castling and promotion are not fixed automatically")
            return False
        mover = "w" if self.board.turn else "b"
        src, dst = move.from_square, move.to_square
        if after[src] == mover:  # still standing on its source square
            self.log("  camera: the piece is still on its source square: playing the move again")
            self.slots = copy.deepcopy(saved_slots)
            self.pp.play(self.board, move, self.slots, recover=recover)
            return True
        strays = self.verifier.strays(before, after, self.board, move)
        if strays:  # a piece of the mover's colour appeared where none belongs: it fell or was put down wrong
            sq = min(strays, key=lambda q: float(np.linalg.norm(square_center_mm(q) - square_center_mm(dst))))
            piece = self.board.piece_at(src)
            self.log(f"  camera: a {mover} piece stands on {chess.square_name(sq)}: picking it up from there and putting it on "
                     f"{chess.square_name(dst)}")
            t = Transfer(piece.symbol(), tuple(square_center_mm(sq)), tuple(square_center_mm(dst)),
                         f"fix {chess.square_name(sq)}-{chess.square_name(dst)}")
            after_board = self.board.copy()
            after_board.push(move)
            obstacles = {tuple(square_center_mm(q)): p.symbol() for q, p in after_board.piece_map().items() if q != dst}
            obstacles = {xy: s for xy, s in obstacles.items() if float(np.hypot(xy[0] - t.src[0], xy[1] - t.src[1])) > 1.0}
            self.pp.execute(t, obstacles=obstacles)
            self.pp.go_look()
            return True
        return False

    def push_robot_by_hand(self, move: chess.Move):
        """The human made the robot's move physically (after Unreachable/GraspFailed)."""
        self.board.push(move)

    def describe(self):
        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn else "black",
            "robot_color": "white" if self.robot_color else "black",
            "move_stack_san": chess.Board().variation_san(self.board.move_stack) if self.board.move_stack else "",
            "eval": self.engine.evaluate(self.board),
            "game_over": self.board.is_game_over(),
            "result": self.board.result() if self.board.is_game_over() else None,
            "vision": self.vision,
            "dry_run": self.dry_run,
        }

    def close(self, go_home=True):
        """go_home=False after Ctrl+C: leave the arm frozen where it is (a joint-space swing home could drop a carried piece)."""
        try:
            if go_home:
                self.pp.go_look()
        finally:
            self.engine.close()
            if self.cam:
                self.cam.close()
            if self.watcher:
                self.watcher.close()
            if self.ext:
                self.ext.close()
            if self._own_verify_cam and self.verify_cam is not None:
                self.verify_cam.close()
            self.arm.close()
