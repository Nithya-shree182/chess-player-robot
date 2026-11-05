# Moving to the new Ubuntu machine (written 2026-11-05)

Goal: be productive within ~30 minutes of getting the new machine. Everything needed is either in this repo (branch
`windows-new-rig-2026-11-05`) or in the backup bundle `chess_rig_bundle_2026-11-05.zip` (same files + model weights + the
Windows environment list). Read [../STATUS.md](../STATUS.md) afterwards for where the project stands.

## 0. What carries over and what must be redone
| Item | Carries over? | Notes |
|---|---|---|
| Arm calibrations (`data/calibration_backup/my_follower.json`, `my_leader.json`) | **Yes** (same arms) | The motors also keep their homing offsets inside themselves, so the JSON files and the arms belong together. Copied by `setup/migrate_calibration.sh` |
| Board <-> robot (`data/board_to_robot.json`) | **Only if the robot and board stay in the SAME places relative to each other** | It is robot-relative, not machine-specific. If the robot base or the board moved even a few mm: redo `board_to_robot --grid --ring` (~10 min, torque off, you guide the tip to 17 points) and check with `hover_test` |
| Home / look pose, gripper table, piece profiles | Yes | `data/look_pose.json`, `data/gripper_gap.json`, `data/piece_profiles.json` |
| Phone corners (`ext_corners*.json`) | **No, redo** | They were already stale when the session ended. Empty board, phone fixed: `python -m chess_player.calib.fit_phone_corners --rough "a1x,y h1x,y h8x,y a8x,y"` |
| Stockfish | Re-download (Linux build) | `setup/download_models.sh` (done by the install script) |
| YOLO | Public pretrained model only | Our own model is NOT trained yet. Do that on the new GPU (dataset capture + `vision/train_yolo.py`), that is where the better GPU pays off |
| Windows-only things | not needed | `scripts/*.ps1`, DirectShow camera indexes, COM ports |

## 1. Get the code
```bash
git clone --recurse-submodules https://github.com/Nithya-shree182/chess-player-robot chess && cd chess
git checkout windows-new-rig-2026-11-05        # (once it has been pushed), or unzip the bundle:  unzip chess_rig_bundle_2026-11-05.zip
```

## 2. Python environment (NVIDIA driver must already work: `nvidia-smi`)
```bash
bash setup/install_ubuntu.sh          # CUDA 12.8 PyTorch (RTX 50xx and older), pinned packages, Stockfish, models. ~10 min
```
It ends by printing `cuda available: True` and the GPU name. If it says NO GPU, fix the driver first. The exact versions that
worked on Windows are in `setup/windows_pip_freeze_2026-11-05.txt` (LeRobot 0.6.1, torch 2.11, ultralytics 8.4.202, numpy 2.2.6, opencv-headless 4.13).

## 3. Arms: calibration + device names
Plug both arms in (USB data cables; the leader's first cable was charge-only) and switch their power on, then:
```bash
bash setup/migrate_calibration.sh
source scripts/env.sh
python src/tools/read_pose.py          # six joints with torque=on: the follower talks
```
The new boards are `58FA082690` (follower -> `/dev/so100_follower`) and `5AE6082864` (leader -> `/dev/so100_leader`); the udev
rules are in `setup/99-so100.rules`. If `/dev/so100_*` do not appear, re-plug the arms (or log out/in once for the `dialout` group).
If the board arrangement changed, check the pose reads sensibly and run `python -m chess_player.calib.hover_test --height 15 --squares a8 h8 a1 h1 d4`
(the arm moves; hand near the power) to confirm the calibration still holds.

## 4. Cameras
- **Phone** (fixed camera): install/open the "IP Webcam" app, Start server, then `export CHESS_CAMERA_URL=http://<phone-ip>:<port>/video`
  (the address changes with the network; same Wi-Fi as the PC, or use the phone's hotspot). Photo size is set to 4000x2256 by the code
  (zoom). Plug the phone in (it overheated and ran low in the first session).
- **Webcam on the gripper** (Ausha): `v4l2-ctl --list-devices`; the default path is `/dev/v4l/by-id/usb-Ausha_Webcam_Ausha_Webcam-video-index0`
  (`GRASP_DEBUG_CAM` in `config.py`; used for the debug photos in `data/grasp_debug/`).
- Linux uses V4L2 instead of DirectShow: nothing to change, `vision/camera.py` handles both.

## 5. Smoke tests (no arm movement unless stated)
```bash
source scripts/env.sh
python -m chess_player.move_piece e2 e4 --dry-run --fen start       # plans a move: prints "plan: P e2e4: ..."
python -m chess_player.vision.zoom                                    # phone still read at full resolution (needs the phone + corners)
scripts/run_pipeline.sh --dry-run                                     # UI at http://localhost:8770/ with a simulated arm
```
Then with the arm (hand near the power): `hover_test` as above, then the lone-pawn pick test from STATUS.md.

## 5b. Time-savers on the better GPU
- Train YOLO from the phone view (varied lighting, board shifted a little): `vision/capture_dataset.py`, `vision/train_yolo.py --model yolo11s.pt --imgsz 1280` etc.; compare models on a held-out session. The 4 GB laptop could not do this properly.
- Qwen/Whisper (voice) fit easily now: `scripts/run_pipeline.sh --voice` (needs internet for the first model download; pre-download for the venue).

## 6. Things that behave differently on Linux
- `scripts/run_pipeline.sh` opens Chrome (or `xdg-open`); `--voice` is OFF by default (it was ON in the original).
- Serial ports are `/dev/so100_*`, not COM5/COM6 (override: `CHESS_FOLLOWER_PORT`, `CHESS_LEADER_PORT`).
- The inverse kinematics is pure Python/SciPy now (no `placo` needed); results were checked against the old placo numbers.
- The Windows-only helper tools (`tools/torque.py`, `tools/safe_lift.py`, `tools/read_pose.py`) work on both.

## 7. If something is off
| Symptom | Likely cause |
|---|---|
| `Could not connect on port` / no motors found | arm not powered, USB cable is charge-only, or another program has the port |
| Arm lands a few mm off at some squares | robot or board moved: redo the 17-point touch calibration |
| Camera grid does not sit on the board | phone or board moved: refit the phone corners (empty board) |
| `torch.cuda.is_available()` False | NVIDIA driver / CUDA build mismatch: `TORCH_CUDA=cu126 bash setup/install_ubuntu.sh` |
