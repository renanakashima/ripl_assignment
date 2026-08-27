#!/usr/bin/env bash
set -euo pipefail

RIPL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$RIPL_SCRIPT_DIR/_runpod_experiment.sh" pusht "$@"
