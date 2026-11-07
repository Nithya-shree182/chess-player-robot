"""Correct the board -> robot calibration by CLICKING THE TIP in the (calibrated) overhead camera view, then VALIDATE.

  python -m chess_player.calib.nudge_visual [--squares a8 h8 ...] [--height 15] [--port 8780]

The arm MOVES. Open http://localhost:8780/ . For each square the arm hovers its jaw tip --height mm above where the
current calibration says the square's centre is. The page shows a zoomed live view of that square. The GREEN CROSSHAIR
marks where the tip must end up (it is drawn where the tip appears in the camera at that hover height). Click the very
TIP of the closed jaws (the point between the two fingertips): the program measures how far the tip is from the
crosshair in the picture, converts that to millimetres on the board, and moves the arm by exactly the difference.
Click the tip again until it sits on the crosshair, then press Accept. Clicking ON the crosshair does nothing (the
measured error is zero).

When all squares are done a 2-D correction (shift + small rotation + scale) is fitted and saved to
data/board_correction.json. Then comes VALIDATION: the arm goes to a few squares that were NOT used in the fit, with the
new correction applied. Click the tip once on each (the arm does not move): the error in mm is measured and reported,
and the whole result is PASS (every error < 3.5 mm) or FAIL.

The phone must not move and the board must be taped down. Pieces off the board. If the tip ever touches the board,
press Ctrl+C in the terminal (the arm lifts and goes home).
"""
import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess
import cv2
import numpy as np

from chess_player import config as C
from chess_player.calib.nudge_calibrate import DEFAULT_SQUARES, MAX_NUDGE_MM, MIN_HEIGHT_MM, fit_similarity
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.arm import Arm, FakeArm, look_pose
from chess_player.motion.ik import IK, solve_or_raise
from chess_player.vision import ext_camera as E

HALF_PX = 90  # half-size of the zoomed crop around the target, in camera pixels
ZOOM = 6  # displayed pixels per camera pixel
MAX_CLICK_MM = 30.0  # a click further than this from the crosshair is probably a mistake
VALIDATION_SQUARES = ["g2", "b7", "e4"]  # not in DEFAULT_SQUARES: held out of the fit
PASS_MM = 3.5

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Nudge calibration</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#14161a;--fg:#e8e8ea;--mut:#9aa0a8;--acc:#38d27a;--warn:#ffb84a}
@media (prefers-color-scheme: light){:root{--bg:#f5f5f2;--fg:#1c1c1e;--mut:#5f6368}}
body{background:var(--bg);color:var(--fg);font:16px system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:18px;margin:0 0 4px} .sub{color:var(--mut);margin-bottom:12px}
#wrap{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
#v{width:min(640px,92vw);height:auto;cursor:crosshair;border:2px solid var(--mut);border-radius:6px;background:#000}
#side{min-width:260px;flex:1}
button{font:inherit;padding:10px 16px;margin:4px 6px 4px 0;border-radius:8px;border:1px solid var(--mut);background:transparent;color:var(--fg);cursor:pointer}
button.go{background:var(--acc);color:#04210f;border-color:var(--acc);font-weight:600}
button.stop{border-color:#e5534b;color:#e5534b}
#msg{margin:10px 0;min-height:4em} .big{font-size:22px;font-weight:600} #phase{color:var(--warn);font-weight:600}
</style></head><body>
<h1>Nudge calibration</h1>
<div class=sub>Click the <b>tip of the closed jaws</b>, not the crosshair. The tip must end up <b>on</b> the green crosshair.</div>
<div id=wrap><img id=v alt="live view"><div id=side>
<div id=phase></div><div class=big id=sq>...</div><div id=off class=sub></div><div id=msg></div>
<button class=go id=ok onclick="cmd('accept')">Accept (tip is on the crosshair)</button><br>
<button onclick="cmd('skip')">Skip this square</button><button onclick="cmd('done')">Finish now</button>
<button class=stop onclick="cmd('stop')">STOP</button></div></div>
<script>
const v=document.getElementById('v'); v.src='/stream';
v.addEventListener('click',e=>{const r=v.getBoundingClientRect();
 fetch('/click',{method:'POST',body:JSON.stringify({x:(e.clientX-r.left)/r.width,y:(e.clientY-r.top)/r.height})});});
function cmd(c){fetch('/cmd',{method:'POST',body:JSON.stringify({cmd:c})});}
async function poll(){try{const s=await (await fetch('/state')).json();
 document.getElementById('phase').textContent=s.phase||'';
 document.getElementById('sq').textContent=s.square?('Square '+s.square+'  ('+s.index+' of '+s.total+')'):s.status;
 document.getElementById('off').textContent=(s.square&&s.phase!='VALIDATION')?('correction so far: '+s.off[0].toFixed(1)+', '+s.off[1].toFixed(1)+' mm'):'';
 document.getElementById('ok').textContent=(s.phase=='VALIDATION')?'Next square':'Accept (tip is on the crosshair)';
 document.getElementById('msg').textContent=s.message;}catch(e){} setTimeout(poll,400);}
poll();
</script></body></html>"""


# ---------------------------------------------------------------- geometry (pure functions, unit-tested)
def tip_crosshair_px(view, xy0, height_mm):
    """Camera pixel where the jaw tip appears when it is exactly over board point xy0, `height_mm` above the board.
    = the board point's pixel on the plane + the height parallax (a raised point shifts away from the camera's nadir)."""
    plane = view.mm_to_px([xy0])[0]
    par = view.project([(xy0[0], xy0[1], height_mm)])[0] - view.project([(xy0[0], xy0[1], 0.0)])[0]
    return plane + par


def px_per_mm_matrix(view, xy0):
    """2x2 J: camera-pixel change per board-mm change (columns: board x, board y), around board point xy0."""
    d = 2.5
    px = view.mm_to_px([(xy0[0] + d, xy0[1]), (xy0[0] - d, xy0[1]), (xy0[0], xy0[1] + d), (xy0[0], xy0[1] - d)])
    return np.column_stack([(px[0] - px[1]) / (2 * d), (px[2] - px[3]) / (2 * d)])


def tip_error_mm(view, xy0, height_mm, click_px):
    """Where the clicked tip is relative to where it should be, in board mm (x along files, y along ranks)."""
    J = px_per_mm_matrix(view, xy0)
    return np.linalg.solve(J, np.asarray(click_px, float) - tip_crosshair_px(view, xy0, height_mm))


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = queue.Queue()
        self.state = {"status": "starting", "phase": "", "square": None, "index": 0, "total": 0, "off": [0.0, 0.0],
                      "message": ""}
        self.crop = None  # (x0, y0, size_px) of the displayed crop in camera pixels
        self.view = None
        self.target_px = None
        self.square_xy = None
        self.marker = None  # camera pixel of the last click (drawn as a circle)

    def set(self, **kw):
        with self.lock:
            self.state.update(kw)

    def get(self):
        with self.lock:
            return dict(self.state)


def make_handler(sh, cam):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/":
                b = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif self.path == "/state":
                self._json(sh.get())
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                try:
                    while True:
                        img = render(sh, cam)
                        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ok:
                            b = jpg.tobytes()
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(b))
                            self.wfile.write(b + b"\r\n")
                        time.sleep(0.12)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/click" and sh.crop is not None:
                x0, y0, size = sh.crop
                sh.jobs.put(("click", (x0 + d["x"] * size, y0 + d["y"] * size)))
                self._json({"ok": True})
            elif self.path == "/cmd":
                sh.jobs.put((d.get("cmd", ""), None))
                self._json({"ok": True})
            else:
                self._json({"error": "bad request"}, 400)

    return H


def render(sh, cam):
    """Zoomed live crop around the target: outline of the real square, green crosshair = where the tip must be."""
    try:
        frame = cam.grab(max_age_s=3.0)
    except Exception:
        frame = np.zeros((720, 1280, 3), np.uint8)
    if sh.crop is None or sh.view is None:
        out = cv2.resize(frame, (960, 540))
        cv2.putText(out, "waiting for the arm...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return out
    x0, y0, size = sh.crop
    pad = size
    big = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    crop = big[int(round(y0)) + pad : int(round(y0)) + pad + size, int(round(x0)) + pad : int(round(x0)) + pad + size]
    out = cv2.resize(crop, (size * ZOOM, size * ZOOM), interpolation=cv2.INTER_CUBIC)
    ox, oy = int(round(x0)), int(round(y0))

    def to_out(p):
        return int(round((p[0] - ox) * ZOOM)), int(round((p[1] - oy) * ZOOM))

    sq = sh.get().get("square")
    if sq:
        xy = square_center_mm(chess.parse_square(sq))
        s = C.SQUARE_MM / 2
        quad = sh.view.mm_to_px([(xy[0] - s, xy[1] - s), (xy[0] + s, xy[1] - s), (xy[0] + s, xy[1] + s), (xy[0] - s, xy[1] + s)])
        cv2.polylines(out, [np.array([to_out(p) for p in quad], np.int32)], True, (80, 220, 255), 1, cv2.LINE_AA)
    cx, cy = to_out(sh.target_px)
    cv2.line(out, (cx - 34, cy), (cx + 34, cy), (60, 230, 120), 2, cv2.LINE_AA)
    cv2.line(out, (cx, cy - 34), (cx, cy + 34), (60, 230, 120), 2, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), 7, (60, 230, 120), 2, cv2.LINE_AA)
    cv2.putText(out, "tip goes here", (cx + 12, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 230, 120), 2, cv2.LINE_AA)
    if sh.marker is not None:
        mx, my = to_out(sh.marker)
        cv2.circle(out, (mx, my), 10, (60, 60, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squares", nargs="+", default=DEFAULT_SQUARES)
    ap.add_argument("--validate", nargs="*", default=VALIDATION_SQUARES, help="held-out squares for the validation")
    ap.add_argument("--height", type=float, default=15.0, help="mm above the board while you judge (>= 12)")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.height < MIN_HEIGHT_MM:
        raise SystemExit(f"--height {a.height} is too low: use at least {MIN_HEIGHT_MM:.0f}.")

    T, ik = BoardToRobot.load(apply_correction=False), IK()
    cam = E.ExtCamera()
    sh = Shared()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(sh, cam))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Open http://localhost:{a.port}/ in your browser.")
    input("Board taped, pieces off, phone fixed, hand clear of the arm? Press Enter to start the arm... ")

    arm = FakeArm() if a.dry_run else Arm()
    look = {**look_pose(), "gripper": C.GRIPPER_LOOK}
    true_pts, cmd_pts, names = [], [], []
    saved = False
    try:
        frame = cam.grab(max_age_s=5.0)
        sh.view = E.view(frame)
        q = np.array([look[j] for j in C.ARM_JOINTS])
        q[4] = C.WRIST_ROLL_DEG

        def aim(name, TT):
            """Park the camera crop on `name`; return (xy0, hi, lo) joints for hovering there with transform TT."""
            nonlocal q
            xy0 = square_center_mm(chess.parse_square(name))
            sh.square_xy = xy0
            sh.target_px = tip_crosshair_px(sh.view, xy0, a.height)
            sh.crop = (float(sh.target_px[0] - HALF_PX), float(sh.target_px[1] - HALF_PX), 2 * HALF_PX)
            sh.marker = None
            sh.set(status="moving", square=None, message=f"moving to {name}...")
            hi = solve_or_raise(ik, TT(*xy0, C.Z_SAFE), q, what=f"{name} high")
            arm.move_to({**dict(zip(C.ARM_JOINTS, hi)), "gripper": 0.0})
            arm.settle()
            lo = solve_or_raise(ik, TT(*xy0, a.height), hi, C.GRASP_MAX_TILT, what=f"{name} low")
            arm.move_to(dict(zip(C.ARM_JOINTS, lo)), C.DESCEND_S)
            arm.settle()
            q = lo
            return xy0

        def lift(xy0, TT, off=np.zeros(2)):
            nonlocal q
            sh.set(status="lifting", square=None, message="lifting...")
            q = solve_or_raise(ik, TT(*(xy0 + off), C.Z_SAFE), q, what="lift")
            arm.move_to(dict(zip(C.ARM_JOINTS, q)), C.DESCEND_S)

        quit_all = False
        # ------------------------------------------------ phase 1: nudge each square
        for idx, name in enumerate(a.squares, 1):
            if quit_all:
                break
            xy0 = aim(name, T)
            off = np.zeros(2)
            sh.set(status="ready", phase="CALIBRATION", square=name, index=idx, total=len(a.squares), off=[0.0, 0.0],
                   message=f"Click the TIP of the closed jaws. It must end up on the green crosshair (the true centre of {name} at this height).")
            while True:
                kind, val = sh.jobs.get()
                if kind == "stop":
                    if hasattr(arm, "freeze"):
                        arm.freeze()
                    sh.set(status="stopped", square=None, message="Stopped: arm frozen, nothing saved.")
                    print("stopped: arm frozen, nothing saved")
                    return
                if kind == "accept":
                    true_pts.append(xy0)
                    cmd_pts.append(xy0 + off)
                    names.append(name)
                    print(f"accepted {name}: correction ({off[0]:+.1f}, {off[1]:+.1f}) mm")
                    break
                if kind == "skip":
                    break
                if kind == "done":
                    quit_all = True
                    break
                if kind != "click":
                    continue
                sh.marker = np.asarray(val, float)
                err = tip_error_mm(sh.view, xy0, a.height, val)  # tip minus target, board mm
                if np.linalg.norm(err) > MAX_CLICK_MM:
                    sh.set(message=f"That click is {np.linalg.norm(err):.0f} mm from the crosshair: click the TIP of the jaws.")
                    continue
                if np.linalg.norm(err) < 0.7:
                    sh.set(message="That is on the crosshair already (error under 1 mm). Press Accept.")
                    continue
                new = off - err  # command the opposite of the error
                if np.linalg.norm(new) > MAX_NUDGE_MM:
                    sh.set(message="That would move further than I'll allow from the calibrated point.")
                    continue
                try:
                    q = solve_or_raise(ik, T(*(xy0 + new), a.height), q, C.GRASP_MAX_TILT, what=name)
                except Exception as e:
                    sh.set(message=f"Can't reach that: {e}")
                    continue
                off = new
                sh.set(status="moving", message="moving...", off=[float(off[0]), float(off[1])])
                arm.move_to(dict(zip(C.ARM_JOINTS, q)), 0.6)
                arm.settle(iters=2)
                sh.marker = None
                sh.set(status="ready",
                       message=f"Measured the tip {np.linalg.norm(err):.1f} mm from the crosshair ({err[0]:+.1f}, {err[1]:+.1f}); moved the arm by ({-err[0]:+.1f}, {-err[1]:+.1f}). Click the tip again, or Accept if it is on the crosshair.")
            lift(xy0, T, off)

        # ------------------------------------------------ fit and save
        verdict = None
        if len(true_pts) >= 4:
            A, b, res = fit_similarity(true_pts, cmd_pts)
            scale = float(np.sqrt(abs(np.linalg.det(A))))
            rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
            msg = (f"fitted from {len(true_pts)} squares: shift ({b[0]:+.1f}, {b[1]:+.1f}) mm, rotation {rot:+.2f} deg, "
                   f"scale {scale:.4f}; leftover per square (mm) {dict(zip(names, np.round(res, 1).tolist()))}")
            print(msg)
            if scale < 0.9 or scale > 1.1 or abs(rot) > 6:
                print("WARNING: that is a big correction: check the clicks were on the tip. Not saved.")
                sh.set(status="done", phase="", square=None, message="Big correction: not saved. " + msg)
            else:
                C.BOARD_CORRECTION.write_text(json.dumps(
                    {"A": A.tolist(), "b": b.tolist(), "squares": names,
                     "nudges_mm": [(cc - t).tolist() for t, cc in zip(true_pts, cmd_pts)],
                     "residual_mm": float(np.sqrt((res ** 2).mean())), "height_mm": a.height, "method": "visual"}, indent=2))
                saved = True
                print("saved", C.BOARD_CORRECTION)
                # -------------------------------------------- phase 2: validate on held-out squares
                T2 = BoardToRobot.load()  # with the new correction
                errs = []
                for idx, name in enumerate(a.validate, 1):
                    if quit_all:
                        break
                    xy0 = aim(name, T2)
                    sh.set(status="ready", phase="VALIDATION", square=name, index=idx, total=len(a.validate),
                           message="Do not move the arm. Click the TIP once: I will measure the error. Then press Next square.")
                    measured = False
                    while True:
                        kind, val = sh.jobs.get()
                        if kind == "stop":
                            if hasattr(arm, "freeze"):
                                arm.freeze()
                            print("stopped")
                            return
                        if kind in ("done",):
                            quit_all = True
                            break
                        if kind == "skip":
                            break
                        if kind == "accept":
                            if not measured:
                                sh.set(message="Click the tip first so I can measure it (or press Skip this square).")
                                continue
                            break
                        if kind == "click":
                            sh.marker = np.asarray(val, float)
                            e = tip_error_mm(sh.view, xy0, a.height, val)
                            errs.append((name, float(np.linalg.norm(e)), e.tolist()))
                            measured = True
                            print(f"validation {name}: error {np.linalg.norm(e):.1f} mm ({e[0]:+.1f}, {e[1]:+.1f})")
                            sh.set(message=f"Error at {name}: {np.linalg.norm(e):.1f} mm ({e[0]:+.1f}, {e[1]:+.1f}). Press Next square.")
                    lift(xy0, T2)
                if errs:
                    worst = max(x[1] for x in errs)
                    mean = float(np.mean([x[1] for x in errs]))
                    verdict = "PASS" if worst < PASS_MM else "FAIL"
                    out = f"VALIDATION {verdict}: mean error {mean:.1f} mm, worst {worst:.1f} mm over {len(errs)} held-out squares"
                    print(out)
                    d = json.loads(C.BOARD_CORRECTION.read_text())
                    d["validation"] = {"verdict": verdict, "errors": errs, "pass_mm": PASS_MM}
                    C.BOARD_CORRECTION.write_text(json.dumps(d, indent=2))
                    sh.set(status="done", phase="", square=None, message=out + ". " + msg)
                else:
                    sh.set(status="done", phase="", square=None, message="Saved, but not validated. " + msg)
        else:
            print("fewer than 4 squares accepted: nothing saved")
            sh.set(status="done", phase="", square=None, message="Fewer than 4 squares accepted: nothing saved.")
    finally:
        try:  # lift first, then go home (never sweep sideways at board height)
            now = arm.present()
            qn = np.array([now[j] for j in C.ARM_JOINTS])
            sol, err_, _ = ik.solve(ik.fk(qn)[:3, 3] + np.array([0, 0, 0.08]), qn)
            if err_ < 5:
                arm.move_to(dict(zip(C.ARM_JOINTS, sol)), 1.5)
            arm.move_to(look, 3.0)
        except Exception as e:
            print("(could not return to the look pose:", e, ")")
        arm.close()
        time.sleep(1.0)  # let the page read the final message
        cam.close()
    if saved:
        print("Calibration saved.")


if __name__ == "__main__":
    main()
