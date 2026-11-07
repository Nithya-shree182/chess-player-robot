#!/usr/bin/env bash
# Put the NEW rig's saved calibrations in the places the software looks, and give the arms stable device names.
#   bash setup/migrate_calibration.sh        (plug both arms in first; you will be asked for sudo)
set -e
cd "$(dirname "$0")/.."

CAL="$HOME/.cache/huggingface/lerobot/calibration"
mkdir -p "$CAL/robots/so_follower" "$CAL/teleoperators/so_leader"
cp -v data/calibration_backup/my_follower.json "$CAL/robots/so_follower/my_follower.json"
cp -v data/calibration_backup/my_leader.json "$CAL/teleoperators/so_leader/my_leader.json"

# stable names /dev/so100_follower and /dev/so100_leader from the boards' serial numbers (new rig: 58FA082690, 5AE6082864)
sudo cp setup/99-so100.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo usermod -aG dialout "$USER" || true   # serial access without root (log out and in once if it was not set before)

echo
echo "--- USB boards (expect two CH343 1a86:55d3):"
lsusb | grep -i 1a86 || echo "NO arm boards found: plug the USB cables in, switch the arm power on"
echo "--- stable names:"
ls -l /dev/so100_follower /dev/so100_leader 2>/dev/null || echo "(symlinks appear after re-plugging the arms; check: udevadm info -a -n /dev/ttyACM0 | grep serial)"
echo "--- cameras (the Ausha webcam is mounted on the gripper):"
v4l2-ctl --list-devices 2>/dev/null || true
echo
echo "Check the arm talks:   source scripts/env.sh && python src/tools/read_pose.py"
