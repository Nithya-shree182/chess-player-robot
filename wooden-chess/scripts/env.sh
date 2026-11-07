# Shared settings for all scripts. Usage (from the repo folder): source scripts/env.sh
# Python environment: the repo's own .venv (setup/install_ubuntu.sh creates it) if present, else the old conda env `lerobot`.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$REPO/.venv/bin/activate" ]; then
  source "$REPO/.venv/bin/activate"
else
  source ~/miniforge3/etc/profile.d/conda.sh
  conda activate lerobot
fi
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

FOLLOWER_PORT=/dev/so100_follower
# Leader: stable udev name if the leader rule is installed, else whichever ttyACM isn't the follower
if [ -e /dev/so100_leader ]; then
  LEADER_PORT=/dev/so100_leader
else
  LEADER_PORT=$(ls /dev/ttyACM* 2>/dev/null | grep -v "$(readlink -f "$FOLLOWER_PORT")" | head -1)
fi
# the webcam mounted on the gripper (Ausha)
GRIPPER_CAM=/dev/v4l/by-id/usb-Ausha_Webcam_Ausha_Webcam-video-index0
FOLLOWER_ID=my_follower   # the new rig's LeRobot calibration ids (files in data/calibration_backup/)
LEADER_ID=my_leader
