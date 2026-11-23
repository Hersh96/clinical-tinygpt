#!/usr/bin/env bash
# One-time setup inside WSL2 (Ubuntu) on the RTX 4050 laptop.
# Prerequisite on Windows: a current NVIDIA driver (WSL2 uses the Windows driver; do NOT install a Linux driver).
set -euo pipefail
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: update the Windows NVIDIA driver and restart WSL (wsl --shutdown)"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv
sudo apt-get update -y && sudo apt-get install -y make git software-properties-common
# The pinned packages (numpy 2.5, scipy 1.18) need Python 3.12+. Ubuntu 24.04 has it;
# on 22.04 we add the deadsnakes PPA for python3.12.
if ! command -v python3.12 >/dev/null; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update -y
fi
sudo apt-get install -y python3.12 python3.12-venv
python3.12 -m venv .venv
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
