#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "dino" ]]; then
    echo "warning: expected conda environment 'dino'; current='${CONDA_DEFAULT_ENV:-none}'" >&2
fi

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" -m vtm_lcg.train.train_predictor \
    --config configs/predictor/phase1_smoke.yaml \
    --all \
    "$@"

