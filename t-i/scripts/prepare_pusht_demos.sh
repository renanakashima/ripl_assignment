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
  echo "Inspect ${DEMO_ROOT}/PushT-v1 to see which controller variant was downloaded."
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
  if (( EXISTING_DEMOS >= NUM_DEMOS )); then
    echo "RGB demonstrations already exist (${EXISTING_DEMOS}): ${RGB_TRAJECTORY}"
    exit 0
  fi
  echo "Existing RGB file has ${EXISTING_DEMOS} demos, but ${NUM_DEMOS} were requested."
  echo "Move it aside or delete it intentionally, then rerun this script."
  exit 1
fi

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
if actual < expected:
    raise SystemExit(f"Replay wrote {actual} trajectories; expected at least {expected}")
print(f"Verified {actual} RGB trajectories")
PY
echo "Prepared RGB demonstrations: ${RGB_TRAJECTORY}"
