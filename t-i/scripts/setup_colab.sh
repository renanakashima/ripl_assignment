#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        f"Python {sys.version.split()[0]} is unsupported; use a Colab Python 3.12 runtime."
    )
print(f"Using Python {sys.version.split()[0]}")
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA GPU detected. In Colab choose Runtime > Change runtime type > T4 GPU."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# This mirrors the Vulkan setup cell in ManiSkill's official quickstart notebook.
mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d
curl -fsSL \
  https://raw.githubusercontent.com/mani-skill/ManiSkill/main/docker/nvidia_icd.json \
  -o /usr/share/vulkan/icd.d/nvidia_icd.json
curl -fsSL \
  https://raw.githubusercontent.com/mani-skill/ManiSkill/main/docker/10_nvidia.json \
  -o /usr/share/glvnd/egl_vendor.d/10_nvidia.json
apt-get update -qq
apt-get install -y --no-install-recommends libvulkan-dev

python -m pip install --upgrade pip
python -m pip install -e .
python scripts/verify_setup.py

