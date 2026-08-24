#!/usr/bin/env bash
set -euo pipefail
python train_dp.py --config configs/pusht_rgb.yaml "$@"

