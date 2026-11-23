#!/usr/bin/env bash
# One-time setup inside WSL2 (Ubuntu) on the RTX 4050 laptop.
# Prerequisite on Windows: a current NVIDIA driver (WSL2 uses the Windows driver; do NOT install a Linux driver).
set -euo pipefail
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: update the Windows NVIDIA driver and restart WSL (wsl --shutdown)"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv
sudo apt-get update -y && sudo apt-get install -y python3-venv python3-pip make git
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.lock
pip install -e . --no-deps
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
PY
