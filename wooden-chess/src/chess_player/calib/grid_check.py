"""Camera picture + YOLO boxes + the board grid with the DIAGONALS of every square (their crossing = the square's centre in the picture),
and the hand-taught tip positions (data/center_teach.json) drawn in the same picture. One shot, no arm movement, nothing is changed.

  python -m chess_player.calib.grid_check [out.jpg] [square]   (needs scripts/livecam.py running; with a square also saves a 4x zoom <out>_zoom.jpg)

Board mm frame: a1 outer corner (0, 0), +x toward the h-file, +y toward rank 8; C.EXT_CORNERS holds the pixels of a1, h1, h8, a8.
NOTE: the crossing of a square's diagonals, mapped back to board mm through the same grid, IS the nominal centre (projective geometry), so the
picture check only shows whether the physical squares sit under the grid; the real comparison is taught tip - nominal centre (the offsets).
The taught tips are drawn on the board plane; they were taught 7-25 mm above the board, so a low camera shows them shifted in the picture.
"""
import json
import sys
import urllib.request

import cv2
import numpy as np

from chess_player import config as C

W = C.PLAY_MM


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else str(C.DATA / "grid_check.jpg")
    img = cv2.imdecode(np.frombuffer(urllib.request.urlopen(str(C.EXT_CAMERA_URL).replace("/video", "/shot"), timeout=8).read(), np.uint8), 1)
    px = np.array(json.loads(C.EXT_CORNERS.read_text())["px"], np.float32)
    H = cv2.getPerspectiveTransform(np.array([[0, 0], [W, 0], [W, W], [0, W]], np.float32), px)  # board mm -> pixel
    Hi = np.linalg.inv(H)

    def to_px(p):
        q = H @ np.array([p[0], p[1], 1.0])
        return q[:2] / q[2]

    def to_mm(p):
        q = Hi @ np.array([p[0], p[1], 1.0])
        return q[:2] / q[2]

    teach = json.loads(C.DATA.joinpath("center_teach.json").read_text())
    S = C.SQUARE_MM
    names = "abcdefgh"
    rows = []
    for f in range(8):
        for r in range(8):
            c = [to_px((f * S, r * S)), to_px(((f + 1) * S, r * S)), to_px(((f + 1) * S, (r + 1) * S)), to_px((f * S, (r + 1) * S))]
            cv2.polylines(img, [np.int32(c)], True, (0, 255, 255), 1)
            for a, b in ((0, 2), (1, 3)):
                cv2.line(img, tuple(np.int32(c[a])), tuple(np.int32(c[b])), (255, 160, 0), 1)
            # intersection of the two diagonals in the picture
            A = np.array([c[2] - c[0], -(c[3] - c[1])]).T
            t = np.linalg.solve(A, c[1] - c[0])
            cross = c[0] + t[0] * (c[2] - c[0])
            cv2.circle(img, tuple(np.int32(cross)), 3, (0, 255, 0), -1)
            name = f"{names[f]}{r + 1}"
            nom = np.array([(f + 0.5) * S, (r + 0.5) * S])
            if name in teach:
                tip = np.array(teach[name]["tip_board_mm"][:2], float)
                p = to_px(tip)
                cv2.circle(img, tuple(np.int32(p)), 4, (0, 0, 255), -1)
                cv2.line(img, tuple(np.int32(cross)), tuple(np.int32(p)), (0, 0, 255), 1)
                rows.append((name, float(np.linalg.norm(to_mm(cross) - nom)), tip - nom, float(np.linalg.norm(p - cross))))
    try:
        from ultralytics import YOLO

        res = YOLO(str(C.DATA / "yolo" / "best.pt"))(img.copy(), conf=0.35, verbose=False)[0]
        for b in res.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(img, res.names[int(b.cls)] + f" {float(b.conf):.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
        print(f"YOLO: {len(res.boxes)} boxes")
    except Exception as e:  # the picture and the grid are the point
        print("YOLO skipped:", e)
    cv2.imwrite(out, img)
    if len(sys.argv) > 2:  # 4x zoom around that square's centre
        sq = sys.argv[2].lower()
        cx, cy = to_px(((names.index(sq[0]) + 0.5) * S, (int(sq[1]) - 0.5) * S))
        x0, y0 = int(max(0, cx - 110)), int(max(0, cy - 80))
        crop = img[y0:int(cy + 80), x0:int(cx + 110)]
        zp = out.replace(".jpg", "_zoom.jpg")
        cv2.imwrite(zp, cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
        print("zoom:", zp)
    print("picture:", out)
    print("square  cross-vs-nominal(mm)  taught tip - nominal (X right, Y forward, mm)  tip-vs-cross in the picture (px)")
    for name, dm, d, dp in rows:
        print(f"  {name}    {dm:5.2f}              {d[0]:+6.1f}, {d[1]:+6.1f}                                {dp:5.1f}")
    if rows:
        arr = np.array([r[2] for r in rows])
        print(f"{len(rows)} taught squares: mean offset {arr.mean(0)[0]:+.1f}, {arr.mean(0)[1]:+.1f} mm, scatter (std) {arr.std(0)[0]:.1f}, {arr.std(0)[1]:.1f} mm")


if __name__ == "__main__":
    main()
