#!/bin/bash
#SBATCH --job-name=pusht-replay-dpose
#SBATCH --partition=techfee
#SBATCH --nodelist=nvidia-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pusht-replay-delta-pose-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

echo "job_id=${SLURM_JOB_ID:-none}"
echo "host=$(hostname)"
nvidia-smi

NUM_DEMOS="${NUM_DEMOS:-100}" \
REPLAY_ENVS="${REPLAY_ENVS:-1024}" \
bash scripts/prepare_pusht_delta_pose_demos.sh
