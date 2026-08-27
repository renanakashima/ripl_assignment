#!/usr/bin/env bash
set -euo pipefail

MODEL="${RIPL_T3_MODEL:-Qwen/Qwen3.8-27B}"
PORT="${RIPL_T3_PORT:-8000}"
MAX_MODEL_LEN="${RIPL_T3_MAX_MODEL_LEN:-32768}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to verify GPU memory." >&2
  exit 1
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo "Install a vLLM release that supports Qwen3.8 before running this script." >&2
  exit 1
fi

TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if (( TOTAL_MIB < 75000 )); then
  echo "Refusing unquantized $MODEL on ${TOTAL_MIB} MiB; use one 80+ GB GPU." >&2
  exit 1
fi

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"video": 1}' \
  --media-io-kwargs '{"video": {"num_frames": -1}}'

