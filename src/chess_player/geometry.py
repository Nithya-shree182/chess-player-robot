"""Board <-> robot coordinates. Board frame in mm (see config.py), robot frame in meters."""
import json

import chess
import numpy as np

from chess_player import config as C


def square_center_mm(sq: int) -> np.ndarray:
    return np.array([(chess.square_file(sq) + 0.5) * C.SQUARE_MM, (chess.square_rank(sq) + 0.5) * C.SQUARE_MM])


def mm_to_square(x: float, y: float):
    """Square index for a board-mm point, or None if it's off the board."""
    f, r = int(np.floor(x / C.SQUARE_MM)), int(np.floor(y / C.SQUARE_MM))
    return chess.square(f, r) if 0 <= f < 8 and 0 <= r < 8 else None


class BoardToRobot:
    """Rigid transform p_robot[m] = R @ p_board[mm] / 1000 + t."""

    def __init__(self, R, t, residual_mm=None, A=None, b=None):
        self.R, self.t, self.residual_mm = np.asarray(R, float), np.asarray(t, float), residual_mm
        # Optional 2-D correction measured by eye (calib/nudge_calibrate.py): to put the tip over the true board
        # point p, the arm is sent to A @ p + b instead. Fixes what the hand-touched rigid fit gets wrong (e.g. the
        # far ranks); None = no correction.
        self.A = None if A is None else np.asarray(A, float)
        self.b = None if b is None else np.asarray(b, float)

    def __call__(self, x_mm, y_mm, z_mm=0.0) -> np.ndarray:
        if self.A is not None:
            x_mm, y_mm = self.A @ np.array([x_mm, y_mm]) + self.b
        return self.R @ (np.array([x_mm, y_mm, z_mm]) / 1000.0) + self.t

    @classmethod
    def load(cls, path=C.BOARD_TO_ROBOT, apply_correction=True):
        d = json.loads(path.read_text())
        A = b = None
        if apply_correction and C.BOARD_CORRECTION.exists():
            c = json.loads(C.BOARD_CORRECTION.read_text())
            A, b = c["A"], c["b"]
        return cls(d["R"], d["t"], d.get("residual_mm"), A, b)

    def save(self, path=C.BOARD_TO_ROBOT, **extra):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"R": self.R.tolist(), "t": self.t.tolist(), "residual_mm": self.residual_mm, **extra}, indent=2)
        )

    @classmethod
    def assumed(cls, near_mm=110.0, lateral_mm=0.0, table_z_mm=0.0, robot_side="black"):
        """Placement guess for planning before calibration: board centered in front of the robot, the robot's
        own back rank nearest, near edge `near_mm` from the base axis."""
        B = 8 * C.SQUARE_MM
        if robot_side == "black":  # rank 8 nearest; files a->h run to the robot's left (+y)
            R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
            t = np.array([near_mm + B, -B / 2 + lateral_mm, table_z_mm]) / 1000
        else:  # rank 1 nearest; files a->h run to the robot's right (-y)
            R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
            t = np.array([near_mm, B / 2 + lateral_mm, table_z_mm]) / 1000
        return cls(R, t)


def fit_rigid(board_pts_mm, robot_pts_m):
    """Least-squares rigid transform (Kabsch, no scale) from board points to robot points.

    With all board points on z=0 the rotation's third column is fixed by the cross product of the first two.
    Returns (BoardToRobot, per-point residuals in mm).
    """
    P = np.asarray(board_pts_mm, float) / 1000.0
    Q = np.asarray(robot_pts_m, float)
    if len(P) < 3:
        raise ValueError("need at least 3 non-collinear points")
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = qc - R @ pc
    res = np.linalg.norm((P @ R.T + t) - Q, axis=1) * 1000
    return BoardToRobot(R, t, float(np.sqrt((res**2).mean()))), res
