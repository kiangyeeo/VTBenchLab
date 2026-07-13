#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-/cache/ma-user/VTBenchLab}"
DINO_REPO="${DINO_REPO:-$WORKSPACE/dinov2}"
DATA="${DATA:-$WORKSPACE/data/imagenet1k}"
EXTRA="${EXTRA:-$DATA/extra}"
OUT_ROOT="${OUT_ROOT:-$WORKSPACE/outputs/vae_linear_probing_dinov2_single_paperlr}"
NUM_WORKERS="${NUM_WORKERS:-8}"

run_tokenizer_linear_probe() {
    local model="$1"
    shift

    if [[ "${CONDA_DEFAULT_ENV:-}" != "dino" ]]; then
        echo "!! expected conda environment 'dino'; current='${CONDA_DEFAULT_ENV:-none}'" >&2
        exit 1
    fi
    [[ -d "$DINO_REPO/dinov2" ]] || { echo "!! missing DINOv2 repository: $DINO_REPO" >&2; exit 1; }
    [[ -d "$DATA" ]] || { echo "!! missing ImageNet root: $DATA" >&2; exit 1; }
    [[ -d "$EXTRA" ]] || { echo "!! missing ImageNet extra directory: $EXTRA" >&2; exit 1; }

    echo ">> tokenizer linear probing: $model"
    echo "   single visible GPU; physical/global batch=1024; accumulation=1"
    echo "   updates=12500; base LR grid=0.0001 ... 0.5; DINO batch scaling=x4"
    echo "   output_root=$OUT_ROOT"

    PYTHONPATH="$DINO_REPO${PYTHONPATH:+:$PYTHONPATH}" \
        python "$SCRIPT_DIR/linear_probe.py" \
        --model "$model" \
        --data-root "$DATA" \
        --extra-root "$EXTRA" \
        --output-root "$OUT_ROOT" \
        --num-workers "$NUM_WORKERS" \
        "$@"
}
