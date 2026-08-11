#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-/cache/ma-user/VTBenchLab}"
DINO_REPO="${DINO_REPO:-$WORKSPACE/dinov2}"
HF_DATA_ROOT="${HF_DATA_ROOT:-$WORKSPACE/data/hf_datasets}"
OUT_ROOT="${OUT_ROOT:-$WORKSPACE/outputs/sun397_linear_probing_dinov2_single_surface}"
NUM_WORKERS="${NUM_WORKERS:-8}"

run_tokenizer_sun397_probe() {
    local model="$1"
    shift

    [[ -d "$DINO_REPO/dinov2" ]] || { echo "!! missing DINOv2 repository: $DINO_REPO" >&2; exit 1; }
    [[ -d "$HF_DATA_ROOT/sun397" ]] || { echo "!! missing SUN397 dataset: $HF_DATA_ROOT/sun397" >&2; exit 1; }

    echo ">> SUN397 tokenizer linear probing: $model"
    echo "   fixed readout; batch=1024; 13-LR grid; SGD/cosine"
    echo "   75 updates/epoch ~= one 76,127-image train pass"
    echo "   validation selects the head; test is evaluated once at the end"
    echo "   output_root=$OUT_ROOT"

    PYTHONPATH="$DINO_REPO${PYTHONPATH:+:$PYTHONPATH}" \
        python "$SCRIPT_DIR/linear_probe.py" \
        --model "$model" \
        --data-root "$HF_DATA_ROOT" \
        --output-root "$OUT_ROOT" \
        --num-workers "$NUM_WORKERS" \
        "$@"
}
