#!/usr/bin/env bash
set -euo pipefail

RIPL_PROJECT_DIR="${RIPL_PROJECT_DIR:-/workspace/ripl_assignment/t-i}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

if [[ ! -d "$RIPL_PROJECT_DIR" ]]; then
  echo "Project directory not found: $RIPL_PROJECT_DIR" >&2
  echo "Clone the repository into /workspace/ripl_assignment first." >&2
  exit 1
fi

cd "$RIPL_PROJECT_DIR"

python - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        f"Python {sys.version.split()[0]} is unsupported; use Python 3.10-3.12."
    )
print(f"Using Python {sys.version.split()[0]}")
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA GPU detected. Launch this script inside a Runpod GPU Pod." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends curl git libvulkan-dev

# Match ManiSkill's official headless Vulkan setup.
mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d
RIPL_VULKAN_TMP_DIR="$(mktemp -d)"
trap 'rm -f "$RIPL_VULKAN_TMP_DIR/nvidia_icd.json" "$RIPL_VULKAN_TMP_DIR/10_nvidia.json"; rmdir "$RIPL_VULKAN_TMP_DIR"' EXIT
curl -fsSL \
  https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/nvidia_icd.json \
  -o "$RIPL_VULKAN_TMP_DIR/nvidia_icd.json"
curl -fsSL \
  https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/10_nvidia.json \
  -o "$RIPL_VULKAN_TMP_DIR/10_nvidia.json"
if [[ ! -s /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  install -m 0644 \
    "$RIPL_VULKAN_TMP_DIR/nvidia_icd.json" \
    /usr/share/vulkan/icd.d/nvidia_icd.json
fi
if [[ ! -s /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
  install -m 0644 \
    "$RIPL_VULKAN_TMP_DIR/10_nvidia.json" \
    /usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi

if python -m pip install --help | grep -- '--break-system-packages' >/dev/null; then
  python -m pip install --break-system-packages --upgrade pip
  python -m pip install --break-system-packages -e .
else
  python -m pip install --upgrade pip
  python -m pip install -e .
fi

export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
python - <<'PY'
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the Runpod GPU")

env = gym.make(
    "PushCube-v1",
    num_envs=1,
    sim_backend="physx_cpu",
    obs_mode="rgb",
    control_mode="pd_ee_delta_pos",
    render_mode="rgb_array",
)
try:
    observation, _ = env.reset(seed=0)
    camera_shapes = {
        name: tuple(camera["rgb"].shape)
        for name, camera in observation["sensor_data"].items()
    }
    print(f"Runpod PushCube RGB setup is ready; camera tensors: {camera_shapes}")
finally:
    env.close()
PY
