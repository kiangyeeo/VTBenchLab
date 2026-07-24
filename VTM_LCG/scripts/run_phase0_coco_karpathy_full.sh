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

for split in train validation test; do
    echo "==> extracting full COCO Karpathy split: $split"
    "$PYTHON_BIN" -m vtm_lcg.cache.extract \
        --config "configs/coco_karpathy_full/phase0_${split}.yaml" \
        --all \
        "$@"
done

