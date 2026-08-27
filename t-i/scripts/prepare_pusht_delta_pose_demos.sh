#!/usr/bin/env bash
set -euo pipefail

NUM_DEMOS="${NUM_DEMOS:-100}"
REPLAY_ENVS="${REPLAY_ENVS:-1024}"
ALLOW_REPLAY_ENV_MISMATCH="${ALLOW_REPLAY_ENV_MISMATCH:-0}"
CONTROL_MODE="pd_ee_delta_pose"
DEMO_ROOT="${MS_ASSET_DIR:-${HOME}/.maniskill}/demos"
RAW_TRAJECTORY="${DEMO_ROOT}/PushT-v1/rl/trajectory.none.${CONTROL_MODE}.physx_cuda.h5"
RGB_TRAJECTORY="${DEMO_ROOT}/PushT-v1/rl/trajectory.rgb.${CONTROL_MODE}.physx_cuda.h5"

python -m mani_skill.utils.download_demo PushT-v1

VALIDATE_SOURCE_ARGS=(
  source
  --trajectory "${RAW_TRAJECTORY}"
  --count "${NUM_DEMOS}"
  --control-mode "${CONTROL_MODE}"
  --replay-envs "${REPLAY_ENVS}"
)
if [[ "${ALLOW_REPLAY_ENV_MISMATCH}" == "1" ]]; then
  VALIDATE_SOURCE_ARGS+=(--allow-replay-env-mismatch)
fi
python scripts/validate_pusht_replay.py "${VALIDATE_SOURCE_ARGS[@]}"

if [[ -f "${RGB_TRAJECTORY}" ]]; then
  python scripts/validate_pusht_replay.py output \
    --trajectory "${RGB_TRAJECTORY}" \
    --count "${NUM_DEMOS}" \
    --control-mode "${CONTROL_MODE}"
  echo "Using existing validated RGB demonstrations: ${RGB_TRAJECTORY}"
  exit 0
fi

# The source already uses the controller selected by ManiSkill's tuned Push-T
# baseline, so this is observation rendering rather than controller conversion.
python scripts/replay_trajectory_aligned.py \
  --traj-path "${RAW_TRAJECTORY}" \
  --use-env-states \
  --target-control-mode "${CONTROL_MODE}" \
  --obs-mode rgb \
  --save-traj \
  --allow-failure \
  --count "${NUM_DEMOS}" \
  --num-envs "${REPLAY_ENVS}" \
  --sim-backend physx_cuda

python scripts/validate_pusht_replay.py output \
  --trajectory "${RGB_TRAJECTORY}" \
  --count "${NUM_DEMOS}" \
  --control-mode "${CONTROL_MODE}"
echo "Prepared native-controller RGB demonstrations: ${RGB_TRAJECTORY}"
