#!/usr/bin/env bash
# Evaluate UniFlow reconstructions produced from the 1024 canvas with TokBench's
# 256 evaluation buckets. Intermediate JSON files are temporary and are removed
# when this script exits; the final tables are printed to the terminal only.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/image_reconstruction_results}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-16}"

SOURCE_RES=1024
EVAL_SETTING=256
METHOD_NAME=uniflow_1024_as_256
TEXT_DATASETS=(ic13 ic15 tt textocr cord sroie infograph docvqa)

TMP_OUTPUT="$(mktemp -d "${TMPDIR:-/tmp}/tokbench-uniflow-1024-as-256.XXXXXX")"
cleanup() {
    rm -rf -- "$TMP_OUTPUT"
}
trap cleanup EXIT INT TERM

require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "Missing $2: $1" >&2
        exit 1
    fi
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Missing $2: $1" >&2
        exit 1
    fi
}

require_dir "$DATA_ROOT/images/face_data/wflw" "original WFLW images"
require_file "$DATA_ROOT/annotations/face_meta.json" "WFLW metadata"
require_dir "$RECON_ROOT/uniflow/face_data/wflw_${SOURCE_RES}" \
    "UniFlow ${SOURCE_RES} WFLW reconstructions"

for dataset in "${TEXT_DATASETS[@]}"; do
    require_file "$DATA_ROOT/annotations/text_${dataset}.json" "$dataset annotation"
    require_dir "$RECON_ROOT/uniflow/text_data/${dataset}_${SOURCE_RES}" \
        "UniFlow ${SOURCE_RES} $dataset reconstructions"
done

cd "$REPO_ROOT"
python check_eval_requirements.py

echo "Evaluating UniFlow ${SOURCE_RES} reconstructions with ${EVAL_SETTING} text buckets"
for dataset in "${TEXT_DATASETS[@]}"; do
    echo "[text] $dataset"
    python eval_text.py \
        --img_folder "$RECON_ROOT/uniflow/text_data/${dataset}_${SOURCE_RES}/" \
        --gt_path "$DATA_ROOT/annotations/text_${dataset}.json" \
        --dataset "$dataset" \
        --data_type image \
        --batch_size "$BATCH_SIZE" \
        --workers "$WORKERS" \
        --method_name "$METHOD_NAME" \
        --setting "$EVAL_SETTING" \
        --save_dir "$TMP_OUTPUT"
done

echo "Evaluating UniFlow ${SOURCE_RES} reconstructions with ${EVAL_SETTING} face buckets"
python eval_face.py \
    --original_image_path "$DATA_ROOT/images/face_data/wflw" \
    --reconstruction_image_path "$RECON_ROOT/uniflow/face_data/wflw_${SOURCE_RES}/" \
    --tokenizer "$METHOD_NAME" \
    --data_type image \
    --meta_path "$DATA_ROOT/annotations/face_meta.json" \
    --setting "$EVAL_SETTING" \
    --save_dir "$TMP_OUTPUT"

echo "Final TokBench tables (source=${SOURCE_RES}, evaluation buckets=${EVAL_SETTING})"
python compute_all_metrics.py \
    --setting "$EVAL_SETTING" \
    --data_type image \
    --output_path "$TMP_OUTPUT" \
    --summary_path "$TMP_OUTPUT/summary.txt"

echo "Done. Temporary evaluation files were not retained."
