"""Capture one SYNCHRONIZED sample of (board camera picture, follower joints) while you hold the follower steady by hand.
READ ONLY: nothing is powered or moved.

  python -m chess_player.calib.handeye_sample [--wait 90] [--max-height 120] [--steady 2.0] [--tag c2]

It waits until the follower's jaw tip is below --max-height mm (over the board, not parked high) and the joints have not
changed by more than 0.4 deg for --steady seconds; then it grabs the picture and the joints within a few ms of each other and
saves data/handeye/<time>_<tag>.jpg (the picture), a _overlay.jpg (a green cross where the JOINTS put the tip, and the tip's
footprint on the board) and appends data/handeye/samples.json (joints, tip in board mm, pixel where it should appear).
Compare the cross with the real fingertips in the picture: that gap is the board->robot calibration error as the camera sees it.
"""
import argparse
import json
import time
import urllib.request

import cv2
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot
from chess_player.motion.ik import IK
from chess_player.teach.record_poses import FOLLOWER_CAL, Cal, make_bus, read_raw
from chess_player.vision import ext_camera as E

OUT = C.DATA / "handeye"


def shot(url):
    return cv2.imdecode(np.frombuffer(urllib.request.urlopen(url, timeout=3).read(), np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8771/shot", help="single-picture URL of the camera window")
    ap.add_argument("--wait", type=float, default=90.0, help="give up after this many seconds")
    ap.add_argument("--max-height", type=float, default=120.0, help="mm: the tip must be this low (over the board)")
    ap.add_argument("--steady", type=float, default=2.0, help="seconds the joints must stay still")
    ap.add_argument("--tag", default="pose")
    a = ap.parse_args()

    fcal, T, ik = Cal(FOLLOWER_CAL), BoardToRobot.load(), IK()
    bus = make_bus(C.FOLLOWER_PORT, FOLLOWER_CAL)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        t_end, still_since, last, said = time.time() + a.wait, None, None, 0.0
        print(f"waiting for a steady pose over the board (tip below {a.max_height:.0f} mm, still for {a.steady:.0f} s)...", flush=True)
        while time.time() < t_end:
            u = fcal.units(read_raw(bus))
            tip = T.R.T @ (ik.fk(u[:5])[:3, 3] - T.t) * 1000
            moving = last is None or np.abs(u - last).max() > 0.4
            low = tip[2] < a.max_height
            if moving or not low:
                still_since = None
            elif still_since is None:
                still_since = time.time()
            last = u
            if time.time() - said > 3:
                said = time.time()
                print(f"  tip height {tip[2]:6.1f} mm, {'MOVING' if moving else 'still'}"
                      + ("" if low else " (too high: bring it down over the board)"), flush=True)
            if still_since is not None and time.time() - still_since >= a.steady:
                frame = shot(a.url)
                raw2 = read_raw(bus)  # joints again right after the picture
                u2 = fcal.units(raw2)
                if np.abs(u2 - u).max() > 0.4:
                    still_since = None
                    continue
                tip = T.R.T @ (ik.fk(u2[:5])[:3, 3] - T.t) * 1000
                v = E.view(frame)
                pix = v.project([tuple(tip)])[0]
                foot = v.project([(tip[0], tip[1], 0.0)])[0]
                stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + a.tag
                cv2.imwrite(str(OUT / f"{stamp}.jpg"), frame)
                g = frame.copy()
                cv2.drawMarker(g, tuple(int(x) for x in pix), (0, 255, 0), cv2.MARKER_CROSS, 34, 2)
                cv2.circle(g, tuple(int(x) for x in foot), 6, (0, 255, 0), 2)
                cv2.imwrite(str(OUT / f"{stamp}_overlay.jpg"), g)
                log = json.loads((OUT / "samples.json").read_text()) if (OUT / "samples.json").exists() else []
                log.append({"stamp": stamp, "joints": [round(float(x), 2) for x in u2],
                            "tip_board_mm": [round(float(x), 1) for x in tip],
                            "tip_pixel": [round(float(pix[0]), 1), round(float(pix[1]), 1)],
                            "footprint_pixel": [round(float(foot[0]), 1), round(float(foot[1]), 1)]})
                (OUT / "samples.json").write_text(json.dumps(log, indent=1))
                print(f"CAPTURED {stamp}: joints {np.round(u2, 1).tolist()}\n  jaw tip by the joints: board mm "
                      f"({tip[0]:.1f}, {tip[1]:.1f}, {tip[2]:.1f}) -> should appear at pixel ({pix[0]:.0f}, {pix[1]:.0f})\n"
                      f"  picture: {OUT / (stamp + '_overlay.jpg')}")
                return
            time.sleep(0.15)
        print("timed out: no steady pose over the board was held")
    finally:
        bus.port_handler.closePort()


if __name__ == "__main__":
    main()
