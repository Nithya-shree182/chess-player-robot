"""Board -> robot calibration by touching the board ANYWHERE and clicking where the tip is in the camera picture.

  python -m chess_player.calib.touch_click        then open http://localhost:8790/

Why: touching a named grid point by eye is good to only ~10-15 mm with a tilted tool. Here the tip only has to rest on the board
(any spot, spread over the board); the fixed camera sees the contact point, and the grid (a plane homography) turns the clicked pixel
into exact board mm. The joints at that moment go through forward kinematics: (board mm, robot xyz) pairs -> rigid fit, saved like
board_to_robot.py does (data/board_to_robot.json).

Torque is off on pan / lift / elbow / wrist-flex while you do this (wrist roll locked); hold the arm. Needs the camera window
(scripts/livecam.py, http://127.0.0.1:8771/shot) running.
"""
import argparse
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from chess_player import config as C
from chess_player.calib.board_to_robot import read_joints, report_and_save
from chess_player.motion.ik import IK
from chess_player.vision import ext_camera as E

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Touch + click</title>
<style>
:root{--bg:#14161a;--fg:#e8e8ea;--mut:#9aa0a8;--acc:#38d27a;--warn:#ffb84a}
body{background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif;margin:0;padding:12px}
button{font:inherit;padding:9px 14px;margin:3px 6px 3px 0;border-radius:8px;border:1px solid var(--mut);background:transparent;color:var(--fg);cursor:pointer}
button.go{background:var(--acc);color:#04210f;border-color:var(--acc);font-weight:600}
canvas{border:2px solid var(--mut);border-radius:6px;background:#000;cursor:crosshair;max-width:100%}
#row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start} #msg{margin:8px 0;min-height:3em;color:var(--warn)}
</style></head><body>
<div><b>1.</b> <button onclick="post('/torque')">Torque OFF (hold the arm!)</button>
<b>2.</b> put the closed jaws' tip ON the board, any spot, then press <button class=go onclick="cap()">Capture (or Space)</button>
<b>3.</b> click the contact point (the BOTTOM of the FIXED finger's inner face, where it meets the board) in the picture, then again in the zoom.
<button onclick="post('/undo')">Undo last</button><button onclick="post('/finish')">Finish and fit</button></div>
<div id=msg></div>
<div id=row><canvas id=c width=1280 height=720 style="width:min(1000px,70vw)"></canvas>
<div><canvas id=z width=480 height=480 style="width:420px"></canvas><div id=info></div></div></div>
<script>
const c=document.getElementById('c'),x=c.getContext('2d'),z=document.getElementById('z'),zx=z.getContext('2d');
let img=null,rough=null,fine=null;
const msg=t=>document.getElementById('msg').textContent=t;
async function post(u,b){const r=await (await fetch(u,{method:'POST',body:JSON.stringify(b||{})})).json();msg(r.message||'');info(r);return r;}
function info(r){document.getElementById('info').textContent=r.points!==undefined?('points so far: '+r.points):'';}
async function cap(){const r=await post('/capture');if(!r.ok)return;img=new Image();img.onload=()=>{x.drawImage(img,0,0);rough=null;fine=null;zx.clearRect(0,0,480,480)};img.src='/pending.jpg?'+Date.now();}
document.addEventListener('keydown',e=>{if(e.code=='Space'){e.preventDefault();cap();}});
c.addEventListener('click',e=>{if(!img)return;const r=c.getBoundingClientRect();
 rough=[(e.clientX-r.left)/r.width*1280,(e.clientY-r.top)/r.height*720];fine=null;draw();});
z.addEventListener('click',async e=>{if(!rough)return;const r=z.getBoundingClientRect();
 const u=rough[0]-20+(e.clientX-r.left)/r.width*40,v=rough[1]-20+(e.clientY-r.top)/r.height*40;fine=[u,v];draw();
 const res=await post('/commit',{u:u,v:v});if(res.ok){img=null;rough=null;fine=null;}});
function draw(){x.drawImage(img,0,0);x.strokeStyle='#38d27a';x.lineWidth=2;x.beginPath();x.arc(rough[0],rough[1],14,0,7);x.stroke();
 zx.imageSmoothingEnabled=false;zx.drawImage(img,rough[0]-20,rough[1]-20,40,40,0,0,480,480);
 zx.strokeStyle='#ff5555';zx.lineWidth=1;zx.beginPath();zx.moveTo(240,0);zx.lineTo(240,480);zx.moveTo(0,240);zx.lineTo(480,240);zx.stroke();}
</script></body></html>"""


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.bus = None
        self.ik = IK()
        self.pending = None  # (jpg bytes, joints, frame)
        self.names, self.board, self.robot, self.joints = [], [], [], []
        self.done = threading.Event()
        self.msg = ""


def draw_overlay(frame, view):
    out = frame.copy()
    B = 8
    for i in range(B + 1):
        for a, b in (((i, 0), (i, B)), ((0, i), (B, i))):
            p = view.mm_to_px([(a[0] * C.SQUARE_MM, a[1] * C.SQUARE_MM), (b[0] * C.SQUARE_MM, b[1] * C.SQUARE_MM)]).astype(int)
            cv2.line(out, tuple(p[0]), tuple(p[1]), (0, 255, 0), 1)
    return out


def make_handler(st, shot_url):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, **kw):
            self._send(json.dumps({"points": len(st.board), **kw}).encode())

        def do_GET(self):
            if self.path.startswith("/pending.jpg") and st.pending:
                self._send(st.pending[0], "image/jpeg")
            elif self.path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n) or b"{}")
            with st.lock:
                if self.path == "/torque":
                    if st.bus is None:
                        st.bus = read_joints()
                        free = [j for j in C.ARM_JOINTS if j != "wrist_roll"]
                        st.bus.disable_torque(free)
                        now = st.bus.sync_read("Present_Position", num_retry=2)
                        st.bus.write("Goal_Position", "wrist_roll", float(now["wrist_roll"]))
                        st.bus.write("Torque_Enable", "wrist_roll", 1)
                        st.bus.write("Goal_Position", "wrist_roll", float(C.WRIST_ROLL_DEG))
                        time.sleep(1.0)
                    self._json(ok=True, message="Torque is OFF (wrist roll locked). Guide the tip onto the board and press Space.")
                elif self.path == "/capture":
                    if st.bus is None:
                        return self._json(ok=False, message="Press 'Torque OFF' first.")
                    q = st.bus.sync_read("Present_Position", num_retry=2)
                    qv = np.array([q[j] for j in C.ARM_JOINTS])
                    frame = cv2.imdecode(np.frombuffer(urllib.request.urlopen(shot_url, timeout=5).read(), np.uint8), 1)
                    view = E.view(frame)
                    ok, jpg = cv2.imencode(".jpg", draw_overlay(frame, view), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    st.pending = (jpg.tobytes(), qv, frame, view)
                    tilt = float(np.degrees(np.arccos(np.clip(-st.ik.fk(qv)[2, 2], -1, 1))))
                    self._json(ok=True, message=f"Captured (tool tilt {tilt:.0f} deg). Click the contact point, then click it again in the zoom.")
                elif self.path == "/commit" and st.pending:
                    jpg, qv, frame, view = st.pending
                    mm = view.px_to_mm([(d["u"], d["v"])])[0]
                    p = st.ik.fk(qv)[:3, 3]
                    inside = -C.EDGE_MM <= mm[0] <= C.PLAY_MM + C.EDGE_MM and -C.EDGE_MM <= mm[1] <= C.PLAY_MM + C.EDGE_MM
                    if not inside:
                        return self._json(ok=False, message=f"That is {mm[0]:.0f}, {mm[1]:.0f} mm: off the board. Click the contact point.")
                    if any(np.linalg.norm(p - r) < 0.015 for r in st.robot):
                        return self._json(ok=False, message="The arm is in the same place as an earlier touch (within 15 mm). Move the tip to a DIFFERENT spot on the board, then Capture again.")
                    st.names.append(f"touch {len(st.board) + 1}")
                    st.board.append([float(mm[0]), float(mm[1]), 0.0])
                    st.robot.append(p)
                    st.joints.append(qv.tolist())
                    print(f"touch {len(st.board)}: board ({mm[0]:.1f}, {mm[1]:.1f}) mm  tip robot {np.round(p * 1000, 1)}  joints {np.round(qv, 1)}")
                    st.pending = None
                    self._json(ok=True, message=f"Saved point {len(st.board)} (board {mm[0]:.0f}, {mm[1]:.0f} mm). Move to another spot (spread them over the board).")
                elif self.path == "/undo":
                    if st.board:
                        for l in (st.names, st.board, st.robot, st.joints):
                            l.pop()
                    self._json(ok=True, message="Removed the last point.")
                elif self.path == "/finish":
                    if len(st.board) < 8:
                        return self._json(ok=False, message=f"Need at least 8 points (have {len(st.board)}); 12-16 spread over the board is better.")
                    b = np.array(st.board)[:, :2]
                    if np.ptp(b[:, 0]) < 120 or np.ptp(b[:, 1]) < 120:
                        return self._json(ok=False, message="The points do not cover the board (need to span at least 120 mm in both directions). Touch the corners and the middle.")
                    st.done.set()
                    self._json(ok=True, message="Fitting... see the terminal.")
                else:
                    self._json(ok=False, message="nothing to do")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--shot-url", default="http://127.0.0.1:8771/shot")
    a = ap.parse_args()
    st = State()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(st, a.shot_url))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Open http://localhost:{a.port}/  (Ctrl+C to quit without saving)")
    try:
        st.done.wait()
    finally:
        if st.bus is not None:
            st.bus.port_handler.closePort()
    report_and_save(st.names, st.board, st.robot, st.joints, corners_only=False)


if __name__ == "__main__":
    main()
