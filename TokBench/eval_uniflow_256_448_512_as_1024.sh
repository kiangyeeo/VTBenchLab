#!/usr/bin/env bash
# Evaluate UniFlow reconstructions produced with 256, 448, and 512 TokBench
# canvases using the same TokBench 1024 text/face ratio buckets.
#
# UniFlow itself still runs at its fixed native 448x448 input for every source
# canvas. This script is a source-canvas/bucket ablation, not a variable-native-
# resolution UniFlow evaluation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/image_reconstruction_results}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/image_outputs/uniflow_256_448_512_as_1024}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-16}"

SOURCE_CANVASES=(256 448 512)
EVAL_SETTING=1024
TEXT_DATASETS=(ic13 ic15 tt textocr cord sroie infograph docvqa)

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

# Validate all inputs before starting the long OCR and face evaluations.
require_dir "$DATA_ROOT/images/face_data/wflw" "original WFLW images"
require_file "$DATA_ROOT/annotations/face_meta.json" "WFLW metadata"

for source_canvas in "${SOURCE_CANVASES[@]}"; do
    require_dir "$RECON_ROOT/uniflow/face_data/wflw_${source_canvas}" \
        "UniFlow ${source_canvas}-canvas WFLW reconstructions"
    for dataset in "${TEXT_DATASETS[@]}"; do
        require_file "$DATA_ROOT/annotations/text_${dataset}.json" "$dataset annotation"
        require_dir "$RECON_ROOT/uniflow/text_data/${dataset}_${source_canvas}" \
            "UniFlow ${source_canvas}-canvas $dataset reconstructions"
    done
done

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"
python check_eval_requirements.py

for source_canvas in "${SOURCE_CANVASES[@]}"; do
    method_name="uniflow_canvas${source_canvas}_as_${EVAL_SETTING}"
    echo "Evaluating UniFlow source canvas ${source_canvas} with ${EVAL_SETTING} buckets"

    for dataset in "${TEXT_DATASETS[@]}"; do
        echo "[source=$source_canvas] [buckets=$EVAL_SETTING] [text] $dataset"
        python eval_text.py \
            --img_folder "$RECON_ROOT/uniflow/text_data/${dataset}_${source_canvas}/" \
            --gt_path "$DATA_ROOT/annotations/text_${dataset}.json" \
            --dataset "$dataset" \
            --data_type image \
            --batch_size "$BATCH_SIZE" \
            --workers "$WORKERS" \
            --method_name "$method_name" \
            --setting "$EVAL_SETTING" \
            --save_dir "$OUT_DIR"
    done

    echo "[source=$source_canvas] [buckets=$EVAL_SETTING] [face] wflw"
    python eval_face.py \
        --original_image_path "$DATA_ROOT/images/face_data/wflw" \
        --reconstruction_image_path "$RECON_ROOT/uniflow/face_data/wflw_${source_canvas}/" \
        --tokenizer "$method_name" \
        --data_type image \
        --meta_path "$DATA_ROOT/annotations/face_meta.json" \
        --setting "$EVAL_SETTING" \
        --save_dir "$OUT_DIR"
done

python compute_all_metrics.py \
    --setting "$EVAL_SETTING" \
    --data_type image \
    --output_path "$OUT_DIR" \
    --summary_path "$OUT_DIR/image_summary_${EVAL_SETTING}.txt"

echo "Evaluation complete"
echo "Summary: $OUT_DIR/image_summary_${EVAL_SETTING}.txt"
echo "Raw JSON: $OUT_DIR"
