#!/usr/bin/env bash
set -euo pipefail

NUM_DEMOS="${NUM_DEMOS:-100}"
REPLAY_ENVS="${REPLAY_ENVS:-10}"
DEMO_ROOT="${MS_ASSET_DIR:-${HOME}/.maniskill}/demos"
RAW_TRAJECTORY="${DEMO_ROOT}/PushCube-v1/motionplanning/trajectory.h5"
RGB_TRAJECTORY="${DEMO_ROOT}/PushCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"

python -m mani_skill.utils.download_demo PushCube-v1

if [[ ! -f "${RAW_TRAJECTORY}" ]]; then
  echo "Expected raw PushCube trajectory was not found: ${RAW_TRAJECTORY}"
  echo "Inspect ${DEMO_ROOT}/PushCube-v1 to see which demonstration source was downloaded."
  exit 1
fi

if [[ -f "${RGB_TRAJECTORY}" ]]; then
  EXISTING_DEMOS="$(python - "${RGB_TRAJECTORY}" <<'PY'
import h5py
import sys
with h5py.File(sys.argv[1], "r") as handle:
    print(len(handle.keys()))
PY
)"
  if (( EXISTING_DEMOS > 0 )); then
    echo "PushCube RGB demonstrations already exist (${EXISTING_DEMOS}): ${RGB_TRAJECTORY}"
    exit 0
  fi
  echo "The existing RGB trajectory contains no demonstrations: ${RGB_TRAJECTORY}"
  exit 1
fi

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "${RAW_TRAJECTORY}" \
  --use-first-env-state \
  --target-control-mode pd_ee_delta_pos \
  --obs-mode rgb \
  --save-traj \
  --count "${NUM_DEMOS}" \
  --num-envs "${REPLAY_ENVS}" \
  --sim-backend physx_cpu

test -f "${RGB_TRAJECTORY}"
python - "${RGB_TRAJECTORY}" "${NUM_DEMOS}" <<'PY'
import h5py
import sys
with h5py.File(sys.argv[1], "r") as handle:
    actual = len(handle.keys())
expected = int(sys.argv[2])
if actual == 0:
    raise SystemExit("Replay did not write any PushCube trajectories")
if actual < expected:
    print(
        f"Warning: replay requested {expected} episodes but retained {actual}. "
        "Training will use all retained trajectories."
    )
else:
    print(f"Verified {actual} PushCube RGB trajectories")
PY
echo "Prepared PushCube RGB demonstrations: ${RGB_TRAJECTORY}"
