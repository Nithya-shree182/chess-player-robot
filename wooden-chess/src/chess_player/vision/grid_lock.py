"""Re-find the board grid in the fixed camera on every look, so a nudged camera or board never needs a re-fit with an empty board.

PROTOTYPE (not wired into the pipeline yet; see docs/GRID_ROBUSTNESS.md).

  register : ONCE, with a good grid (from fit_phone_corners) and the markers stuck on the board -> data/grid_lock.json
  lock     : per frame -> the 4 playing-area corners (a1, h1, h8, a8) in the CURRENT image, from
               1. the ArUco/AprilTag markers (>= 2 seen; sub-pixel, immune to colour cast / exposure), else
               2. SIFT matches of the board's permanent features against the reference frame (frame, letters, seam, dots), else
               3. None (the caller keeps the last good grid and says so).

Nothing about the marker sizes or positions has to be measured: the reference stores where the markers were IN THE IMAGE when
the grid was known to be right, and a homography between that and now moves the grid. Markers must be on the BOARD (not the
table) so a moved board is followed too.
"""
import json

import cv2
import numpy as np

from chess_player import config as C

LOCK_FILE = C.DATA / "grid_lock.json"
REF_IMAGE = C.DATA / "grid_lock_ref.jpg"
DICT = getattr(cv2.aruco, getattr(C, "GRID_MARKER_DICT", "DICT_APRILTAG_36h11"))  # 4x4_50 gave false hits on the board texture
MAX_RMS_PX = 1.5  # markers must agree with one homography this well, else something is wrong (marker peeled, mirrored, ...)

_params = cv2.aruco.DetectorParameters()
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), _params)
_clahe = cv2.createCLAHE(3.0, (8, 8))


def detect(frame, want=None):
    """{marker id: 4x2 corner px}; tried on the plain gray image, then contrast-equalised (washed out / shadowed boards)."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = {}
    for img in (g, _clahe.apply(g)):
        corners, ids, _ = _detector.detectMarkers(img)
        for c, i in zip(corners, [] if ids is None else ids.ravel()):
            if (want is None or int(i) in want) and int(i) not in found:
                found[int(i)] = c.reshape(4, 2)
        if want is not None and len(found) == len(want):
            break
    return found


def _board_mask(shape, grid_px, margin=0.18):
    """Region of the image that is the board + its border (grown outward from the playing area)."""
    q = np.float32(grid_px)
    c = q.mean(0)
    q = c + (q - c) * (1 + 2 * margin)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, q.astype(np.int32), 255)
    return m


def register(frame, grid_px, lock_file=LOCK_FILE, ref_image=REF_IMAGE):
    """Store the reference. `grid_px` = the 4 corners of a grid known to be right in `frame`."""
    ref = detect(frame)
    if len(ref) < 3:
        raise RuntimeError(f"only {len(ref)} markers seen; need >= 3 on the board (is the light on them? are they in view?)")
    lock_file.write_text(json.dumps({"grid_px": np.asarray(grid_px).tolist(),
                                     "markers": {str(k): v.tolist() for k, v in ref.items()}}, indent=2))
    cv2.imwrite(str(ref_image), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return sorted(ref)


class GridLock:
    def __init__(self, lock_file=LOCK_FILE, ref_image=REF_IMAGE):
        d = json.loads(lock_file.read_text())
        self.grid = np.float32(d["grid_px"])
        self.ref_markers = {int(k): np.float32(v) for k, v in d["markers"].items()}
        self.ref_img = cv2.imread(str(ref_image))
        self._sift = cv2.SIFT_create(4000)
        self._bf = cv2.BFMatcher()
        gray = _clahe.apply(cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY))
        self._kr, self._dr = self._sift.detectAndCompute(gray, _board_mask(self.ref_img.shape, self.grid))

    def _by_markers(self, frame):
        now = detect(frame, want=set(self.ref_markers))
        ids = sorted(now)
        if len(ids) < 2:
            return None
        src = np.concatenate([self.ref_markers[i] for i in ids])
        dst = np.concatenate([now[i] for i in ids])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        if H is None:
            return None
        rms = float(np.sqrt(np.mean(np.sum((cv2.perspectiveTransform(src[None], H)[0] - dst) ** 2, axis=1))))
        # 2 markers (8 points, close together) can't pin the perspective down; accept only with 3+ unless they are far apart
        spread = np.ptp(np.concatenate([now[i] for i in ids]), axis=0).max()
        if rms > MAX_RMS_PX or (len(ids) == 2 and spread < 0.4 * frame.shape[1]):
            return None
        return H, {"source": "markers", "n": len(ids), "rms_px": rms}

    def _by_features(self, frame):
        k, d = self._sift.detectAndCompute(_clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)), None)
        if d is None or len(k) < 20:
            return None
        good = [a for a, b in self._bf.knnMatch(self._dr, d, k=2) if a.distance < 0.75 * b.distance]
        if len(good) < 25:
            return None
        src = np.float32([self._kr[g.queryIdx].pt for g in good])
        dst = np.float32([k[g.trainIdx].pt for g in good])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or inl.sum() < 25:
            return None
        return H, {"source": "features", "n": int(inl.sum()), "rms_px": None}

    def lock(self, frame):
        """(corners_px 4x2 or None, info). Never raises: None means 'could not lock; keep the previous grid'."""
        for method in (self._by_markers, self._by_features):
            r = method(frame)
            if r is not None:
                H, info = r
                px = cv2.perspectiveTransform(self.grid[None], H)[0]
                if cv2.isContourConvex(px.astype(np.float32).reshape(-1, 1, 2)) and cv2.contourArea(px) > 0.05 * frame.shape[0] * frame.shape[1]:
                    return px, info
        return None, {"source": "none"}
