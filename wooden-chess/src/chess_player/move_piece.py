"""Move one piece with the real arm: the motion learned from the demos, collision check always on. The arm MOVES
(after you confirm).

  python -m chess_player.move_piece e3 e5                         # uses the saved real-board position
  python -m chess_player.move_piece e3 e5 --pieces "Re3 Ke1 ke8"   # say where the pieces are (and save it)
  python -m chess_player.move_piece e3 e5 --fen "<FEN>"            # same, as a FEN
  python -m chess_player.move_piece e3 e5 --from-camera            # read the real board from the webcam first
  python -m chess_player.move_piece --show                         # print the saved position
  python -m chess_player.move_piece e3 e5 --dry-run                # plan + report only

Any from/to squares work (not a chess-rule check). The position (data/board_state.json, shared with the
pipeline window) must match the real board, because the collision check avoids the pieces it lists. After a
successful move the position is updated.
"""
import argparse

import chess

from chess_player import board_state
from chess_player import config as C
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.arm import Arm, FakeArm
from chess_player.motion.ik import IK, Unreachable
from chess_player.motion.pick_place import GraspFailed, PickPlace, Transfer, obstacles_for


class _NoSlots:
    occupied = {}


class _Recorder:
    """Records the external camera to an mp4 while the arm moves (to see what really happens at the grasp)."""

    def __init__(self, path, fps=10.0):
        import threading

        import cv2

        from chess_player.vision.ext_camera import ExtCamera

        self.cv2, self.path, self.fps = cv2, path, fps
        self.cam = ExtCamera()
        self._stop = threading.Event()
        self.frames = []
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        import time

        t0 = time.time()
        while not self._stop.is_set():
            try:
                self.frames.append((time.time() - t0, self.cam.grab(max_age_s=1.0).copy()))
            except Exception:
                pass
            time.sleep(1.0 / self.fps)

    def stop(self):
        self._stop.set()
        self.t.join(timeout=3)
        if self.frames:
            h, w = self.frames[0][1].shape[:2]
            vw = self.cv2.VideoWriter(self.path, self.cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            for _, f in self.frames:
                vw.write(f)
            vw.release()
            print(f"recorded {len(self.frames)} frames to {self.path}")
        try:
            self.cam.close()
        except Exception:
            pass


def locate_piece_mm(square_name, max_correction_mm=12.0):
    """(x_mm, y_mm) of the real piece base on `square_name`, read from the external camera, or None if not
    found / too far from the nominal square centre to trust (misdetection, wrong square, empty square)."""
    import numpy as np

    from chess_player.geometry import mm_to_square, square_center_mm
    from chess_player.vision import detect_board as D
    from chess_player.vision.ext_camera import ExtCamera, view

    cam = ExtCamera()
    try:
        frame = cam.grab(max_age_s=2.0)
    finally:
        cam.close()
    v = view(frame)
    boxes = []
    D.detect_pieces(frame, v, boxes_out=boxes)
    target = chess.parse_square(square_name)
    best = None
    for sym, conf, x1, y1, x2, y2, sq, h, note in boxes:
        if sq != target:
            continue
        if best is None or conf > best[0]:
            best = (conf, x1, y1, x2, y2)
    if best is None:
        print(f"locate: no piece detected on {square_name}, using the nominal square centre")
        return None
    conf, x1, y1, x2, y2 = best
    base_px = np.array([(x1 + x2) / 2, y2 - 0.12 * (y2 - y1)])
    x_mm, y_mm = v.px_to_mm([base_px])[0]
    cx, cy = square_center_mm(target)
    dist = float(np.hypot(x_mm - cx, y_mm - cy))
    if dist > max_correction_mm:
        print(f"locate: detected piece on {square_name} is {dist:.1f} mm from the nominal centre "
              f"(> {max_correction_mm} mm): not trusting it, using the nominal centre")
        return None
    print(f"locate: {square_name} real piece centre is {dist:.1f} mm off the nominal centre "
          f"(conf {conf:.2f}) -> using x {x_mm:.1f} y {y_mm:.1f} mm")
    return (x_mm, y_mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--pieces", help='where the pieces are, e.g. "Re3 Ke1 ke8 Pa2" (capital = white, small = black)')
    ap.add_argument("--fen", help="the position as a FEN ('start' = normal start position)")
    ap.add_argument("--show", action="store_true", help="print the saved position and exit")
    ap.add_argument("--from-camera", action="store_true", help="read the real board with the webcam (YOLO) first")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-camera", action="store_true",
                    help="after a successful move, read src/dst from the external camera (YOLO) and report whether "
                         "the piece actually left src and landed on dst")
    ap.add_argument("--record", metavar="MP4", help="record the external camera to this file during the move")
    ap.add_argument("--locate-src", action="store_true",
                    help="before planning, read the source piece's real position from the external camera "
                         "(YOLO) instead of assuming it sits exactly at the nominal square centre")
    a = ap.parse_args()

    if a.from_camera:
        from chess_player.vision.ext_camera import ExtCamera
        from chess_player.vision.read_board import read_position, save_training_sample

        cam = ExtCamera()
        try:
            board, unsure, frame = read_position(cam)
        finally:
            cam.close()
        print(board_state.show(board))
        print("read from the webcam" + (f"; unsure squares: {' '.join(unsure)}" if unsure else ""))
        if input("Is this the real board? [y/n]: ").strip().lower() != "y":
            raise SystemExit('cancelled: give the position with --pieces "..." instead')
        save_training_sample(frame, board)
        board_state.save(board)
        print("saved as the real-board position (and as YOLO training data)")
    elif a.pieces:
        board = board_state.from_pieces(a.pieces)
    elif a.fen:
        board = chess.Board() if a.fen == "start" else chess.Board(a.fen)
    else:
        board = board_state.load()
    if a.pieces or a.fen:
        board_state.save(board)
        print("saved as the real-board position")
    if not a.from_camera:
        print(board_state.show(board))
    if a.show or not (a.src and a.dst):
        if not a.show:
            print("\ngive two squares, e.g.  python -m chess_player.move_piece e3 e5")
        return
    s, d = chess.parse_square(a.src), chess.parse_square(a.dst)
    piece = board.piece_at(s)
    if piece is None:
        raise SystemExit(f"\nno piece on {a.src} in this position. Tell me where the pieces really are, e.g.\n"
                         f'  python -m chess_player.move_piece {a.src} {a.dst} --pieces "R{a.src} Ke1 ke8"')
    if board.piece_at(d) is not None:
        raise SystemExit(f"\n{a.dst} is occupied ({board.piece_at(d)}) in this position: clear it, or fix the position "
                         f"with --pieces")
    src_mm = tuple(square_center_mm(s))
    if a.locate_src:
        found = locate_piece_mm(a.src)
        if found is not None:
            src_mm = found
    t = Transfer(piece.symbol(), tuple(square_center_mm(s)), tuple(square_center_mm(d)), f"{piece.symbol()} {a.src}{a.dst}")
    (t, obst), = obstacles_for(board, _NoSlots(), [t])
    t.src = src_mm  # obstacles_for matches the moving piece by its nominal centre, so apply the located one after
    T = BoardToRobot.load()
    arm = FakeArm() if a.dry_run else Arm()
    pp = PickPlace(arm, IK(), T, verify_grasp=not a.dry_run)
    try:
        try:
            info = pp.plan_transfer(t, obst)["info"]
        except Unreachable as e:  # includes Blocked (would hit a piece)
            raise SystemExit(f"\ncan't do it safely: {e}")
        print(f"\nplan: {info['step']}: collision check ON ({len(obst)} other pieces), shift {info['jaw_shift_mm']} mm, "
              f"{info['route']}, up to {info['z_travel_mm']} mm, grasp tilt {info['max_grasp_tilt_deg']} deg, "
              f"~{info['est_seconds']} s, grip {C.PIECES[piece.symbol().lower()]['grip']} %")
        if a.dry_run:
            return
        input("Arm will move. Keep a hand near it. Enter to go (Ctrl+C to cancel)...")
        rec = _Recorder(a.record) if a.record else None
        moved = False
        try:
            pp.execute(t, obstacles=obst)
            board.remove_piece_at(s)
            board.set_piece_at(d, piece)
            board_state.save(board)
            print("placed; real-board position updated")
            moved = True
        except GraspFailed as e:
            print(f"grasp failed: {e} (position not changed)")
        pp.go_look()
        if rec:
            rec.stop()
        if moved and a.verify_camera:
            from chess_player.vision.ext_camera import ExtCamera
            from chess_player.vision.read_board import read_position

            cam = ExtCamera()
            try:
                seen, unsure, _ = read_position(cam, turn=board.turn)
            finally:
                cam.close()
            seen_src, seen_dst = seen.piece_at(s), seen.piece_at(d)
            ok_src = seen_src is None
            ok_dst = seen_dst is not None and seen_dst.symbol().lower() == piece.symbol().lower()
            print(f"\ncamera check: {a.src} {'empty, good' if ok_src else f'STILL SHOWS {seen_src} (bad)'}; "
                  f"{a.dst} {'has ' + str(seen_dst) + ', good' if ok_dst else f'shows {seen_dst} (expected {piece}, bad)'}"
                  + (f"; unsure: {' '.join(unsure)}" if unsure else ""))
            print("VERIFY: PASS" if ok_src and ok_dst else "VERIFY: FAIL - the move may not have actually worked")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
