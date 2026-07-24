#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "dino" ]]; then
    echo "warning: expected conda environment 'dino'; current='${CONDA_DEFAULT_ENV:-none}'" >&2
fi

"$PYTHON_BIN" -m vtm_lcg.train.train_full_coco \
    --config configs/coco_karpathy_full/phase1_predictor.yaml \
    --all \
    "$@"

