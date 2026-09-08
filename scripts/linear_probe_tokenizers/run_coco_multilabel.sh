#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-/cache/ma-user/VTBenchLab}"
DINO_REPO="${DINO_REPO:-$WORKSPACE/dinov2}"
DATA="${DATA:-$WORKSPACE/data/gvt/raw/coco}"
OUT_ROOT="${OUT_ROOT:-$WORKSPACE/outputs/coco2014_multilabel_linear_probing}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [[ $# -lt 1 ]]; then
    echo "usage: CUDA_VISIBLE_DEVICES=GPU $0 MODEL [probe arguments...]" >&2
    exit 2
fi

MODEL="$1"
shift

[[ -d "$DINO_REPO/dinov2" ]] || { echo "!! missing DINOv2 repository: $DINO_REPO" >&2; exit 1; }
[[ -d "$DATA/train2014" ]] || { echo "!! missing COCO train2014: $DATA/train2014" >&2; exit 1; }
[[ -d "$DATA/val2014" ]] || { echo "!! missing COCO val2014: $DATA/val2014" >&2; exit 1; }
[[ -f "$DATA/annotations/instances_train2014.json" ]] || { echo "!! missing train annotations" >&2; exit 1; }
[[ -f "$DATA/annotations/instances_val2014.json" ]] || { echo "!! missing val annotations" >&2; exit 1; }

echo ">> COCO-2014 multi-label tokenizer linear probing: $MODEL"
echo "   train/val=82081/40137 labeled images; classes=80; batch=1024; epochs=1"
echo "   loss=BCEWithLogits; LR grid/SGD/cosine/readout follow the ImageNet probe"
echo "   training augmentation follows each tokenizer's ImageNet linear-probe transform"
echo "   WebSSL and Pixio: non-affine BatchNorm; all other models: no BatchNorm"
echo "   output_root=$OUT_ROOT"

FEATURE_ARGS=()
if [[ -n "${FEATURE_MICROBATCH_SIZE:-}" ]]; then
    FEATURE_ARGS=(--feature-microbatch-size "$FEATURE_MICROBATCH_SIZE")
fi

PYTHONPATH="$DINO_REPO${PYTHONPATH:+:$PYTHONPATH}" \
    python "$SCRIPT_DIR/coco_multilabel_probe.py" \
    --model "$MODEL" \
    --data-root "$DATA" \
    --output-root "$OUT_ROOT" \
    --num-workers "$NUM_WORKERS" \
    "${FEATURE_ARGS[@]}" \
    "$@"
