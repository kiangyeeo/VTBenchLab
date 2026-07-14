#!/usr/bin/env bash
# Evaluate UniTok 512x512 and 1024x1024 reconstructions using the same TokBench
# 256 text/face ratio buckets. Unlike fixed-native wrappers, UniTok receives the
# source canvas resolution directly and uses a correspondingly larger token grid.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/image_reconstruction_results}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/image_outputs/unitok_512_1024_as_256}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-16}"

SOURCE_RESOLUTIONS=(512 1024)
EVAL_SETTING=256
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

# Fail before evaluation if any annotation or reconstruction directory is absent.
require_dir "$DATA_ROOT/images/face_data/wflw" "original WFLW images"
require_file "$DATA_ROOT/annotations/face_meta.json" "WFLW metadata"

for source_resolution in "${SOURCE_RESOLUTIONS[@]}"; do
    require_dir "$RECON_ROOT/unitok/face_data/wflw_${source_resolution}" \
        "UniTok ${source_resolution} WFLW reconstructions"
    for dataset in "${TEXT_DATASETS[@]}"; do
        require_file "$DATA_ROOT/annotations/text_${dataset}.json" "$dataset annotation"
        require_dir "$RECON_ROOT/unitok/text_data/${dataset}_${source_resolution}" \
            "UniTok ${source_resolution} $dataset reconstructions"
    done
done

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"
python check_eval_requirements.py

for source_resolution in "${SOURCE_RESOLUTIONS[@]}"; do
    method_name="unitok_${source_resolution}_as_${EVAL_SETTING}"
    echo "Evaluating UniTok ${source_resolution} reconstructions with ${EVAL_SETTING} buckets"

    for dataset in "${TEXT_DATASETS[@]}"; do
        echo "[source=$source_resolution] [buckets=$EVAL_SETTING] [text] $dataset"
        python eval_text.py \
            --img_folder "$RECON_ROOT/unitok/text_data/${dataset}_${source_resolution}/" \
            --gt_path "$DATA_ROOT/annotations/text_${dataset}.json" \
            --dataset "$dataset" \
            --data_type image \
            --batch_size "$BATCH_SIZE" \
            --workers "$WORKERS" \
            --method_name "$method_name" \
            --setting "$EVAL_SETTING" \
            --save_dir "$OUT_DIR"
    done

    echo "[source=$source_resolution] [buckets=$EVAL_SETTING] [face] wflw"
    python eval_face.py \
        --original_image_path "$DATA_ROOT/images/face_data/wflw" \
        --reconstruction_image_path "$RECON_ROOT/unitok/face_data/wflw_${source_resolution}/" \
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
