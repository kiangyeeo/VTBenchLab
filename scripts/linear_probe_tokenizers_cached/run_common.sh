#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-/cache/ma-user/VTBenchLab}"
DINO_REPO="${DINO_REPO:-$WORKSPACE/dinov2}"
DATA="${DATA:-$WORKSPACE/data/imagenet1k}"
EXTRA="${EXTRA:-$DATA/extra}"
OUT_ROOT="${OUT_ROOT:-$WORKSPACE/outputs/vae_linear_probing_dinov2_single_noaug_cached_paperlr}"
CACHE_ROOT="${CACHE_ROOT:-$OUT_ROOT/_feature_cache}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-3}"

run_cached_tokenizer_linear_probe() {
    local model="$1"
    shift

    [[ -d "$DINO_REPO/dinov2" ]] || { echo "!! missing DINOv2 repository: $DINO_REPO" >&2; exit 1; }
    [[ -d "$DATA" ]] || { echo "!! missing ImageNet root: $DATA" >&2; exit 1; }
    [[ -d "$EXTRA" ]] || { echo "!! missing ImageNet extra directory: $EXTRA" >&2; exit 1; }

    echo ">> deterministic cached tokenizer linear probing: $model"
    echo "   extract train/val features once; no train-time augmentation"
    echo "   full LR schedule=10 epochs; stop_after_epoch=$STOP_AFTER_EPOCH"
    echo "   output_root=$OUT_ROOT"
    echo "   cache_root=$CACHE_ROOT"

    PYTHONPATH="$DINO_REPO${PYTHONPATH:+:$PYTHONPATH}" \
        python "$SCRIPT_DIR/linear_probe_cached.py" \
        --model "$model" \
        --data-root "$DATA" \
        --extra-root "$EXTRA" \
        --output-root "$OUT_ROOT" \
        --cache-root "$CACHE_ROOT" \
        --num-workers "$NUM_WORKERS" \
        --stop-after-epoch "$STOP_AFTER_EPOCH" \
        "$@"
}
