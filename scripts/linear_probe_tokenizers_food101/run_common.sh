#!/usr/bin/env bash
set -euo pipefail

FOOD101_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FOOD101_WORKSPACE="${FOOD101_WORKSPACE:-/cache/ma-user/VTBenchLab}"
FOOD101_DINO_REPO="${FOOD101_DINO_REPO:-$FOOD101_WORKSPACE/dinov2}"
FOOD101_DATA_ROOT="${FOOD101_DATA_ROOT:-$FOOD101_WORKSPACE/data/hf_datasets}"
FOOD101_OUT_ROOT="${FOOD101_OUT_ROOT:-$FOOD101_WORKSPACE/outputs/food101_linear_probing_dinov2_single_surface}"
FOOD101_NUM_WORKERS="${FOOD101_NUM_WORKERS:-8}"

run_food101_tokenizer_probe() {
    local model="$1"
    shift

    local argument
    for argument in "$@"; do
        if [[ "$argument" == "--model" || "$argument" == --model=* ]]; then
            echo "!! --model cannot be overridden through launcher passthrough arguments" >&2
            return 2
        fi
    done

    [[ -d "$FOOD101_DINO_REPO/dinov2" ]] || { echo "!! missing DINOv2 repository: $FOOD101_DINO_REPO" >&2; exit 1; }
    [[ -d "$FOOD101_DATA_ROOT/food101" ]] || { echo "!! missing Food-101 dataset: $FOOD101_DATA_ROOT/food101" >&2; exit 1; }

    echo ">> balanced Food-101 tokenizer linear probing: $model"
    echo "   650 train / 100 validation / 250 official test images per class"
    echo "   fixed readout; batch=1024; 13-LR grid; SGD/cosine"
    echo "   65 updates/epoch ~= one 65,650-image train pass"
    echo "   validation selects the head; official test is evaluated once"
    echo "   output_root=$FOOD101_OUT_ROOT"

    PYTHONPATH="$FOOD101_DINO_REPO${PYTHONPATH:+:$PYTHONPATH}" \
        python "$FOOD101_SCRIPT_DIR/linear_probe.py" \
        --model "$model" \
        --data-root "$FOOD101_DATA_ROOT" \
        --output-root "$FOOD101_OUT_ROOT" \
        --num-workers "$FOOD101_NUM_WORKERS" \
        "$@"
}
