#!/bin/bash
#SBATCH --job-name=pusht-eval-dpose
#SBATCH --partition=techfee
#SBATCH --nodes=1
#SBATCH --nodelist=nvidia-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pusht-eval-delta-pose-%j.out

set -euo pipefail

cd "$HOME/ripl_assignment/t-i"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

if [[ -z "${RIPL_RUN_DIR:-}" ]]; then
    shopt -s nullglob
    RIPL_RUN_CANDIDATES=(runs/pusht-rgb-diffusion-delta-pose-spatial-replan1__seed1__*)
    if [[ ${#RIPL_RUN_CANDIDATES[@]} -eq 0 ]]; then
        echo "No completed delta-pose Push-T run found. Set RIPL_RUN_DIR explicitly." >&2
        exit 1
    fi
    RIPL_RUN_DIR="${RIPL_RUN_CANDIDATES[$((${#RIPL_RUN_CANDIDATES[@]} - 1))]}"
fi

RIPL_CHECKPOINT="${RIPL_CHECKPOINT:-$RIPL_RUN_DIR/checkpoints/best_success_once.pt}"
RIPL_EVAL_ROOT="${RIPL_EVAL_ROOT:-$RIPL_RUN_DIR/evaluation-final}"
RIPL_EVAL_SEEDS="${RIPL_EVAL_SEEDS:-0 1 2}"

if [[ ! -f "$RIPL_CHECKPOINT" ]]; then
    echo "Checkpoint not found: $RIPL_CHECKPOINT" >&2
    exit 1
fi

mkdir -p "$RIPL_EVAL_ROOT"

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Run directory: $RIPL_RUN_DIR"
echo "Checkpoint: $RIPL_CHECKPOINT"
echo "Evaluation seeds: $RIPL_EVAL_SEEDS"
python --version
nvidia-smi

for RIPL_SEED in $RIPL_EVAL_SEEDS; do
    python -u eval_dp.py \
      --checkpoint "$RIPL_CHECKPOINT" \
      --num-eval-episodes 100 \
      --num-eval-envs 10 \
      --seed "$RIPL_SEED" \
      --output-dir "$RIPL_EVAL_ROOT/seed-$RIPL_SEED" \
      --no-capture-video
done

python -u scripts/aggregate_eval.py --evaluation-root "$RIPL_EVAL_ROOT"
echo "End: $(date --iso-8601=seconds)"
