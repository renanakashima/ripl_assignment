#!/usr/bin/env bash
set -euo pipefail

NUM_DEMOS="${NUM_DEMOS:-100}"
REPLAY_ENVS="${REPLAY_ENVS:-64}"
DEMO_ROOT="${MS_ASSET_DIR:-${HOME}/.maniskill}/demos"
RAW_TRAJECTORY="${DEMO_ROOT}/PushT-v1/rl/trajectory.none.pd_ee_delta_pos.physx_cuda.h5"
RGB_TRAJECTORY="${DEMO_ROOT}/PushT-v1/rl/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5"

python -m mani_skill.utils.download_demo PushT-v1

if [[ ! -f "${RAW_TRAJECTORY}" ]]; then
  echo "Expected raw PushT trajectory was not found: ${RAW_TRAJECTORY}"
  echo "Inspect ${DEMO_ROOT}/PushT-v1/rl for the downloaded controller variant."
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
    echo "RGB demonstrations already exist (${EXISTING_DEMOS}): ${RGB_TRAJECTORY}"
    exit 0
  fi
  echo "The existing RGB trajectory contains no demonstrations: ${RGB_TRAJECTORY}"
  exit 1
fi

# Keep the source controller unchanged so ManiSkill can replay the demonstrations
# in parallel on the GPU while rendering RGB observations.
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "${RAW_TRAJECTORY}" \
  --use-env-states \
  --target-control-mode pd_ee_delta_pos \
  --obs-mode rgb \
  --save-traj \
  --count "${NUM_DEMOS}" \
  --num-envs "${REPLAY_ENVS}" \
  --sim-backend physx_cuda

test -f "${RGB_TRAJECTORY}"
python - "${RGB_TRAJECTORY}" "${NUM_DEMOS}" <<'PY'
import h5py
import sys

with h5py.File(sys.argv[1], "r") as handle:
    actual = len(handle.keys())
expected = int(sys.argv[2])
if actual == 0:
    raise SystemExit("Replay did not write any trajectories")
if actual < expected:
    print(
        f"Warning: replay requested {expected} source episodes but retained {actual}. "
        "Training will use all retained trajectories."
    )
else:
    print(f"Verified {actual} RGB trajectories")
PY
echo "Prepared RGB demonstrations: ${RGB_TRAJECTORY}"
