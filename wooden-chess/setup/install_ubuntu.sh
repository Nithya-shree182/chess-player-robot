#!/usr/bin/env bash
# One-shot Python environment for the chess robot on Ubuntu (22.04 / 24.04) with an NVIDIA GPU.
#   bash setup/install_ubuntu.sh                 # RTX 50xx (5090) and older GPUs: CUDA 12.8 build of PyTorch
#   TORCH_CUDA=cu126 bash setup/install_ubuntu.sh    # pick another CUDA build if the driver is older
# Creates ./.venv (scripts/env.sh uses it). Takes ~10 min, most of it downloads.
set -e
cd "$(dirname "$0")/.."

sudo apt-get update
sudo apt-get install -y git curl ffmpeg v4l-utils libgl1 build-essential python3-venv python3-dev

PY="${PYTHON:-python3}"
$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 1) PyTorch with CUDA (cu128 = Blackwell/RTX 50xx and everything older; driver >= 570)
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA:-cu128}"
# 2) the rest, pinned
pip install -r setup/requirements-ubuntu.txt
# 3) YOLO without letting it replace opencv-headless / numpy
pip install --no-deps ultralytics==8.4.202 ultralytics-thop
# 4) this repo (editable)
pip install --no-deps -e .

python - <<'EOF'
import torch, cv2, numpy, scipy, chess, lerobot, ultralytics
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("numpy", numpy.__version__, "| opencv", cv2.__version__, "| lerobot", lerobot.__version__, "| ultralytics", ultralytics.__version__)
EOF

# 5) Stockfish (Linux build), Piper voice, the public pretrained chess YOLO
bash setup/download_models.sh

echo
echo "Environment ready. Next: plug in the arms, then   bash setup/migrate_calibration.sh"
