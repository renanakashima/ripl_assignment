#!/bin/bash
#SBATCH --job-name=pushcube-dp
#SBATCH --partition=techfee
#SBATCH --nodes=1
#SBATCH --nodelist=nvidia-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pushcube-dp-%j.out

set -euo pipefail

cd "$HOME/ripl_assignment/t-i"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

RIPL_GPU_LOG="$HOME/ripl_assignment/t-i/logs/gpu-pushcube-${SLURM_JOB_ID}.csv"
RIPL_MONITOR_PID=""

cleanup() {
    if [[ -n "${RIPL_MONITOR_PID:-}" ]]; then
        kill "$RIPL_MONITOR_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT TERM INT

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
python --version
nvidia-smi

nvidia-smi \
  --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
  --format=csv \
  --loop=10 \
  > "$RIPL_GPU_LOG" &

RIPL_MONITOR_PID=$!
RIPL_START_EPOCH=$(date +%s)

# Run directly inside the allocation; nested srun has a CPU-binding failure on HCE.
python -u train_dp.py --config configs/pushcube_rgb.yaml

RIPL_END_EPOCH=$(date +%s)
echo "End: $(date --iso-8601=seconds)"
echo "Training wall time: $((RIPL_END_EPOCH - RIPL_START_EPOCH)) seconds"
echo "GPU measurements: $RIPL_GPU_LOG"
