#!/usr/bin/env bash
set -euo pipefail

RIPL_EXPERIMENT="${1:-}"
if [[ -z "$RIPL_EXPERIMENT" ]]; then
  echo "Internal usage: $0 {pushcube|pusht} {setup|prepare|smoke|train|eval|all}" >&2
  exit 2
fi
shift

RIPL_STAGE="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$RIPL_STAGE" in
  setup|prepare|smoke|train|eval|all) ;;
  *)
    echo "Unknown stage: $RIPL_STAGE" >&2
    echo "Choose one of: setup, prepare, smoke, train, eval, all." >&2
    exit 2
    ;;
esac

case "$RIPL_EXPERIMENT" in
  pushcube)
    RIPL_CONFIG="configs/pushcube_rgb.yaml"
    RIPL_SMOKE_NAME="pushcube-rgb-runpod-smoke"
    RIPL_RUN_GLOB="pushcube-rgb-diffusion-spatial__seed1__*"
    RIPL_LOG_STEM="pushcube"
    RIPL_PREPARE_SCRIPT="scripts/prepare_pushcube_demos.sh"
    RIPL_DEFAULT_REPLAY_ENVS=10
    RIPL_DEFAULT_EVAL_ENVS=10
    RIPL_MISSING_RUN_MESSAGE="No completed PushCube run found"
    ;;
  pusht)
    RIPL_CONFIG="configs/pusht_rgb_delta_pose.yaml"
    RIPL_SMOKE_NAME="pusht-rgb-delta-pose-runpod-smoke"
    RIPL_RUN_GLOB="pusht-rgb-diffusion-delta-pose-spatial-replan1__seed1__*"
    RIPL_LOG_STEM="pusht"
    RIPL_PREPARE_SCRIPT="scripts/prepare_pusht_delta_pose_demos.sh"
    RIPL_DEFAULT_REPLAY_ENVS=1024
    RIPL_DEFAULT_EVAL_ENVS=20
    RIPL_MISSING_RUN_MESSAGE="No completed native delta-pose PushT run found"
    ;;
  *)
    echo "Unknown experiment: $RIPL_EXPERIMENT" >&2
    echo "Choose either pushcube or pusht." >&2
    exit 2
    ;;
esac

RIPL_PROJECT_DIR="${RIPL_PROJECT_DIR:-/workspace/ripl_assignment/t-i}"
RIPL_ARTIFACT_ROOT="${RIPL_ARTIFACT_ROOT:-/workspace/ripl-artifacts}"
RIPL_RUN_ROOT="${RIPL_RUN_ROOT:-$RIPL_ARTIFACT_ROOT/runs}"
RIPL_LOG_ROOT="${RIPL_LOG_ROOT:-$RIPL_ARTIFACT_ROOT/logs}"
RIPL_EVAL_SEEDS="${RIPL_EVAL_SEEDS:-0 1 2}"
RIPL_EVAL_ENVS="${RIPL_EVAL_ENVS:-$RIPL_DEFAULT_EVAL_ENVS}"

export MS_ASSET_DIR="${MS_ASSET_DIR:-/workspace/maniskill}"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

mkdir -p "$RIPL_ARTIFACT_ROOT" "$RIPL_RUN_ROOT" "$RIPL_LOG_ROOT" "$MS_ASSET_DIR/demos"
mkdir -p "$HOME/.maniskill"
if [[ ! -e "$HOME/.maniskill/demos" && ! -L "$HOME/.maniskill/demos" ]]; then
  ln -s "$MS_ASSET_DIR/demos" "$HOME/.maniskill/demos"
fi

cd "$RIPL_PROJECT_DIR"

RIPL_MONITOR_PID=""
cleanup() {
  if [[ -n "${RIPL_MONITOR_PID:-}" ]]; then
    kill "$RIPL_MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT TERM INT

latest_run() {
  local candidates=()
  shopt -s nullglob
  candidates=("$RIPL_RUN_ROOT"/$RIPL_RUN_GLOB)
  shopt -u nullglob
  if [[ ${#candidates[@]} -eq 0 ]]; then
    return 1
  fi
  printf '%s\n' "${candidates[$((${#candidates[@]} - 1))]}"
}

setup() {
  bash scripts/setup_runpod.sh
}

prepare() {
  NUM_DEMOS="${NUM_DEMOS:-100}" REPLAY_ENVS="${REPLAY_ENVS:-$RIPL_DEFAULT_REPLAY_ENVS}" \
    bash "$RIPL_PREPARE_SCRIPT"
}

smoke() {
  python -u train_dp.py \
    --config "$RIPL_CONFIG" \
    --exp-name "$RIPL_SMOKE_NAME" \
    --output-dir "$RIPL_RUN_ROOT" \
    --num-demos 4 \
    --batch-size 8 \
    --total-iters 1 \
    --warmup-steps 1 \
    --eval-freq 1 \
    --save-freq 1 \
    --num-eval-episodes 1 \
    --num-eval-envs 1 \
    --no-capture-video
}

train() {
  local gpu_log="$RIPL_LOG_ROOT/gpu-$RIPL_LOG_STEM-$(date -u +%Y%m%dT%H%M%SZ).csv"
  local train_log="$RIPL_LOG_ROOT/$RIPL_LOG_STEM-train-$(date -u +%Y%m%dT%H%M%SZ).log"
  local resume_args=()
  local start_epoch
  local end_epoch

  if [[ -n "${RIPL_RESUME:-}" ]]; then
    if [[ ! -f "$RIPL_RESUME" ]]; then
      echo "Resume checkpoint not found: $RIPL_RESUME" >&2
      exit 1
    fi
    resume_args=(--resume "$RIPL_RESUME")
  fi

  echo "Start: $(date --iso-8601=seconds)"
  python --version
  nvidia-smi
  nvidia-smi \
    --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
    --format=csv \
    --loop=10 \
    > "$gpu_log" &
  RIPL_MONITOR_PID=$!
  start_epoch=$(date +%s)

  python -u train_dp.py \
    --config "$RIPL_CONFIG" \
    --output-dir "$RIPL_RUN_ROOT" \
    "${resume_args[@]}" \
    "$@" 2>&1 | tee "$train_log"

  end_epoch=$(date +%s)
  cleanup
  RIPL_MONITOR_PID=""
  echo "End: $(date --iso-8601=seconds)"
  echo "Training wall time: $((end_epoch - start_epoch)) seconds"
  echo "Training log: $train_log"
  echo "GPU measurements: $gpu_log"
}

evaluate() {
  local run_dir="${RIPL_RUN_DIR:-}"
  local checkpoint
  local eval_root

  if [[ -z "$run_dir" ]]; then
    run_dir="$(latest_run)" || {
      echo "$RIPL_MISSING_RUN_MESSAGE under $RIPL_RUN_ROOT" >&2
      exit 1
    }
  fi
  checkpoint="${RIPL_CHECKPOINT:-$run_dir/checkpoints/best_success_once.pt}"
  eval_root="${RIPL_EVAL_ROOT:-$run_dir/evaluation-final}"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint" >&2
    exit 1
  fi

  mkdir -p "$eval_root"
  echo "Run directory: $run_dir"
  echo "Checkpoint: $checkpoint"
  echo "Evaluation seeds: $RIPL_EVAL_SEEDS"
  echo "Parallel evaluation environments: $RIPL_EVAL_ENVS"
  for seed in $RIPL_EVAL_SEEDS; do
    python -u eval_dp.py \
      --checkpoint "$checkpoint" \
      --num-eval-episodes 100 \
      --num-eval-envs "$RIPL_EVAL_ENVS" \
      --seed "$seed" \
      --output-dir "$eval_root/seed-$seed" \
      --no-capture-video
  done
  python -u scripts/aggregate_eval.py --evaluation-root "$eval_root"
}

case "$RIPL_STAGE" in
  setup)
    setup
    ;;
  prepare)
    prepare
    ;;
  smoke)
    smoke
    ;;
  train)
    train "$@"
    ;;
  eval)
    evaluate
    ;;
  all)
    setup
    prepare
    smoke
    train "$@"
    evaluate
    ;;
esac
