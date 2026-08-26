#!/usr/bin/env bash
set -euo pipefail

NUM_DEMOS="${NUM_DEMOS:-100}"
REPLAY_ENVS="${REPLAY_ENVS:-256}"
DEMO_ROOT="${MS_ASSET_DIR:-${HOME}/.maniskill}/demos"
RAW_TRAJECTORY="${DEMO_ROOT}/PushT-v1/rl/trajectory.none.pd_ee_delta_pos.physx_cuda.h5"
RGB_TRAJECTORY="${DEMO_ROOT}/PushT-v1/rl/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5"

python -m mani_skill.utils.download_demo PushT-v1

if [[ ! -f "${RAW_TRAJECTORY}" ]]; then
  echo "Expected raw PushT trajectory was not found: ${RAW_TRAJECTORY}"
  echo "Inspect ${DEMO_ROOT}/PushT-v1/rl for the downloaded controller variant."
  exit 1
fi

# PushT replay can report false-negative final success labels even when the
# downloaded source trajectory is successful. Verify the requested source
# episodes before allowing replay to retain those false negatives.
python - "${RAW_TRAJECTORY}" "${NUM_DEMOS}" <<'PY'
import json
from pathlib import Path
import sys

import h5py

trajectory_path = Path(sys.argv[1])
expected = int(sys.argv[2])
metadata_path = trajectory_path.with_suffix(".json")

with metadata_path.open() as stream:
    episodes = json.load(stream)["episodes"][:expected]

if len(episodes) != expected:
    raise SystemExit(
        f"Requested {expected} source episodes, but only {len(episodes)} are available"
    )

invalid = []
with h5py.File(trajectory_path, "r") as handle:
    for episode in episodes:
        episode_id = episode["episode_id"]
        group = handle[f"traj_{episode_id}"]
        if "success" not in group or not bool(group["success"][-1]):
            invalid.append(episode_id)

if invalid:
    raise SystemExit(
        "Refusing --allow-failure because these source episodes are not "
        f"successful at the end: {invalid}"
    )

print(f"Verified {expected} successful source trajectories")
PY

if [[ -f "${RGB_TRAJECTORY}" ]]; then
  EXISTING_DEMOS="$(python - "${RGB_TRAJECTORY}" <<'PY'
import h5py
import sys

with h5py.File(sys.argv[1], "r") as handle:
    print(len(handle.keys()))
PY
)"
  if (( EXISTING_DEMOS == NUM_DEMOS )); then
    echo "Verified ${EXISTING_DEMOS} existing RGB demonstrations: ${RGB_TRAJECTORY}"
    exit 0
  fi
  echo "Existing RGB trajectory has ${EXISTING_DEMOS} demos; expected ${NUM_DEMOS}: ${RGB_TRAJECTORY}"
  echo "Move both this HDF5 file and its matching JSON metadata aside, then rerun."
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
  --allow-failure \
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
if actual != expected:
    raise SystemExit(f"Replay wrote {actual} trajectories; expected exactly {expected}")
print(f"Verified {actual} RGB trajectories")
PY
echo "Prepared RGB demonstrations: ${RGB_TRAJECTORY}"
