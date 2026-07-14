#!/usr/bin/env bash
# Evaluate UniFlow 448- and 512-canvas reconstructions with TokBench's 256
# evaluation buckets. Intermediate JSON files are deleted on exit; only terminal
# output is retained.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/image_reconstruction_results}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-16}"

SOURCE_RESOLUTIONS=(448 512)
EVAL_SETTING=256
TEXT_DATASETS=(ic13 ic15 tt textocr cord sroie infograph docvqa)

TMP_OUTPUT="$(mktemp -d "${TMPDIR:-/tmp}/tokbench-uniflow-448-512-as-256.XXXXXX")"
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

# Validate every input before starting the long evaluation.
require_dir "$DATA_ROOT/images/face_data/wflw" "original WFLW images"
require_file "$DATA_ROOT/annotations/face_meta.json" "WFLW metadata"

for source_res in "${SOURCE_RESOLUTIONS[@]}"; do
    require_dir "$RECON_ROOT/uniflow/face_data/wflw_${source_res}" \
        "UniFlow ${source_res} WFLW reconstructions"
    for dataset in "${TEXT_DATASETS[@]}"; do
        require_file "$DATA_ROOT/annotations/text_${dataset}.json" "$dataset annotation"
        require_dir "$RECON_ROOT/uniflow/text_data/${dataset}_${source_res}" \
            "UniFlow ${source_res} $dataset reconstructions"
    done
done

cd "$REPO_ROOT"
python check_eval_requirements.py

for source_res in "${SOURCE_RESOLUTIONS[@]}"; do
    method_name="uniflow_${source_res}_as_${EVAL_SETTING}"

    echo "Evaluating UniFlow ${source_res} reconstructions with ${EVAL_SETTING} text buckets"
    for dataset in "${TEXT_DATASETS[@]}"; do
        echo "[source=$source_res] [text] $dataset"
        python eval_text.py \
            --img_folder "$RECON_ROOT/uniflow/text_data/${dataset}_${source_res}/" \
            --gt_path "$DATA_ROOT/annotations/text_${dataset}.json" \
            --dataset "$dataset" \
            --data_type image \
            --batch_size "$BATCH_SIZE" \
            --workers "$WORKERS" \
            --method_name "$method_name" \
            --setting "$EVAL_SETTING" \
            --save_dir "$TMP_OUTPUT"
    done

    echo "Evaluating UniFlow ${source_res} reconstructions with ${EVAL_SETTING} face buckets"
    python eval_face.py \
        --original_image_path "$DATA_ROOT/images/face_data/wflw" \
        --reconstruction_image_path "$RECON_ROOT/uniflow/face_data/wflw_${source_res}/" \
        --tokenizer "$method_name" \
        --data_type image \
        --meta_path "$DATA_ROOT/annotations/face_meta.json" \
        --setting "$EVAL_SETTING" \
        --save_dir "$TMP_OUTPUT"
done

echo "Final TokBench tables (sources=448,512; evaluation buckets=${EVAL_SETTING})"
python compute_all_metrics.py \
    --setting "$EVAL_SETTING" \
    --data_type image \
    --output_path "$TMP_OUTPUT" \
    --summary_path "$TMP_OUTPUT/summary.txt"

echo "Done. Temporary evaluation files were not retained."
