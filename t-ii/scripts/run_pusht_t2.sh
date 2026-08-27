#!/usr/bin/env bash
set -euo pipefail

RIPL_STAGE="${1:-}"
if [[ -z "$RIPL_STAGE" ]]; then
  echo "Usage: $0 {discover|analyze|targeted}" >&2
  exit 2
fi

RIPL_CHECKPOINT="${RIPL_CHECKPOINT:-}"
RIPL_T2_ROOT="${RIPL_T2_ROOT:-/workspace/ripl-artifacts/t2-pusht}"
RIPL_NUM_ENVS="${RIPL_NUM_ENVS:-20}"
RIPL_FAILURE_MODE="${RIPL_FAILURE_MODE:-discovery}"

RIPL_T2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RIPL_T2_DIR"

run_evaluation() {
  if [[ -z "$RIPL_CHECKPOINT" || ! -f "$RIPL_CHECKPOINT" ]]; then
    echo "Set RIPL_CHECKPOINT to an existing Push-T checkpoint." >&2
    exit 1
  fi
  local seed="$1"
  local episodes="$2"
  local output_dir="$RIPL_T2_ROOT/$RIPL_FAILURE_MODE/seed-$seed"
  local video_args=(--no-capture-video)
  if [[ "${RIPL_CAPTURE_VIDEO:-0}" == "1" ]]; then
    video_args=(--capture-video)
  fi
  python -u eval_pusht_failures.py \
    --checkpoint "$RIPL_CHECKPOINT" \
    --failure-mode-name "$RIPL_FAILURE_MODE" \
    --num-eval-episodes "$episodes" \
    --num-eval-envs "$RIPL_NUM_ENVS" \
    --seed "$seed" \
    --output-dir "$output_dir" \
    --x-rel-min "${RIPL_X_REL_MIN:--0.10}" \
    --x-rel-max "${RIPL_X_REL_MAX:-0.10}" \
    --y-rel-min "${RIPL_Y_REL_MIN:--0.10}" \
    --y-rel-max "${RIPL_Y_REL_MAX:-0.20}" \
    --theta-deg-min "${RIPL_THETA_DEG_MIN:-0}" \
    --theta-deg-max "${RIPL_THETA_DEG_MAX:-360}" \
    "${video_args[@]}"
}

case "$RIPL_STAGE" in
  discover)
    RIPL_FAILURE_MODE="${RIPL_FAILURE_MODE:-discovery}"
    run_evaluation "${RIPL_SEED:-0}" "${RIPL_EPISODES:-200}"
    ;;
  analyze)
    python -u scripts/analyze_pusht_failures.py \
      "$RIPL_T2_ROOT/discovery" \
      --output-dir "$RIPL_T2_ROOT/analysis"
    ;;
  targeted)
    if [[ "$RIPL_FAILURE_MODE" == "discovery" ]]; then
      echo "Set RIPL_FAILURE_MODE and pose-range variables for a targeted evaluation." >&2
      exit 1
    fi
    for seed in ${RIPL_EVAL_SEEDS:-0 1 2}; do
      run_evaluation "$seed" "${RIPL_EPISODES:-100}"
    done
    python -u scripts/analyze_pusht_failures.py \
      "$RIPL_T2_ROOT/$RIPL_FAILURE_MODE" \
      --output-dir "$RIPL_T2_ROOT/$RIPL_FAILURE_MODE/analysis"
    ;;
  *)
    echo "Unknown stage: $RIPL_STAGE; choose discover, analyze, or targeted." >&2
    exit 2
    ;;
esac
