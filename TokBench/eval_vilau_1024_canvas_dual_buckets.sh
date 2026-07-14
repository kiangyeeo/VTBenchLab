#!/usr/bin/env bash
# Evaluate the same native-256 VILA-U / 1024-canvas reconstructions with both
# TokBench's 256 and 1024 ratio buckets.
#
# The 1024-bucket result is an evaluation-bucket ablation, not a native-1024
# VILA-U result: the tokenizer still receives 256x256 inputs internally.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/image_reconstruction_results}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/image_outputs/vilau_native256_canvas1024_dual_buckets}"
TOKENIZER_DIR="${TOKENIZER_DIR:-vilau_7b_256}"
METHOD_NAME="${METHOD_NAME:-vilau_native256_canvas1024}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-16}"
SOURCE_CANVAS=1024

TEXT_DATASETS=(ic13 ic15 tt textocr cord sroie infograph docvqa)
EVAL_SETTINGS=(256 1024)

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

# Validate every reconstruction and annotation before starting the long runs.
require_dir "$DATA_ROOT/images/face_data/wflw" "original WFLW images"
require_file "$DATA_ROOT/annotations/face_meta.json" "WFLW metadata"
require_dir "$RECON_ROOT/$TOKENIZER_DIR/face_data/wflw_${SOURCE_CANVAS}" \
    "VILA-U ${SOURCE_CANVAS}-canvas WFLW reconstructions"

for dataset in "${TEXT_DATASETS[@]}"; do
    require_file "$DATA_ROOT/annotations/text_${dataset}.json" "$dataset annotation"
    require_dir "$RECON_ROOT/$TOKENIZER_DIR/text_data/${dataset}_${SOURCE_CANVAS}" \
        "VILA-U ${SOURCE_CANVAS}-canvas $dataset reconstructions"
done

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"
python check_eval_requirements.py

for setting in "${EVAL_SETTINGS[@]}"; do
    echo "Evaluating source canvas ${SOURCE_CANVAS} with TokBench ${setting} buckets"

    for dataset in "${TEXT_DATASETS[@]}"; do
        echo "[buckets=$setting] [text] $dataset"
        python eval_text.py \
            --img_folder "$RECON_ROOT/$TOKENIZER_DIR/text_data/${dataset}_${SOURCE_CANVAS}/" \
            --gt_path "$DATA_ROOT/annotations/text_${dataset}.json" \
            --dataset "$dataset" \
            --data_type image \
            --batch_size "$BATCH_SIZE" \
            --workers "$WORKERS" \
            --method_name "$METHOD_NAME" \
            --setting "$setting" \
            --save_dir "$OUT_DIR"
    done

    echo "[buckets=$setting] [face] wflw"
    python eval_face.py \
        --original_image_path "$DATA_ROOT/images/face_data/wflw" \
        --reconstruction_image_path "$RECON_ROOT/$TOKENIZER_DIR/face_data/wflw_${SOURCE_CANVAS}/" \
        --tokenizer "$METHOD_NAME" \
        --data_type image \
        --meta_path "$DATA_ROOT/annotations/face_meta.json" \
        --setting "$setting" \
        --save_dir "$OUT_DIR"

    python compute_all_metrics.py \
        --setting "$setting" \
        --data_type image \
        --output_path "$OUT_DIR" \
        --summary_path "$OUT_DIR/image_summary_${setting}.txt"
done

echo "Evaluation complete"
echo "256-bucket summary:  $OUT_DIR/image_summary_256.txt"
echo "1024-bucket summary: $OUT_DIR/image_summary_1024.txt"
