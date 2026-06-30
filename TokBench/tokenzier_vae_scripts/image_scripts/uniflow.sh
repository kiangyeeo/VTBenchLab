#!/bin/bash
# Reconstruction with the UniFlow unified pixel-flow tokenizer.
# UniFlow always runs at a fixed 448x448 input internally; PADDING_SIZE only controls
# the TokBench canvas the original image is fit into (matching the other tokenizers).
# Usage:
#   CUDA_VISIBLE_DEVICES=0 PADDING_SIZES="256" bash uniflow.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/image_reconstruction_results}"
MODEL_ZOO="${MODEL_ZOO:-$REPO_ROOT/tokenizer_modelzoo}"
MODEL_NAME="${MODEL_NAME:-uniflow}"
UNIFLOW_PATH="${UNIFLOW_PATH:-$SCRIPT_DIR/UniFlow}"
MODEL_DIR="${MODEL_DIR:-$MODEL_ZOO/uniflow}"
CONFIG_PATH="${CONFIG_PATH:-$MODEL_DIR}"
CKPT_PATH="${CKPT_PATH:-$MODEL_DIR/model.safetensors}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
if [ "${PADDING_SIZES+x}" ]; then PADDING_SIZES=($PADDING_SIZES); else PADDING_SIZES=(256 512 1024); fi
if [ "${TEXT_DATAS+x}" ]; then TEXT_DATAS=($TEXT_DATAS); else TEXT_DATAS=(ic13 ic15 textocr tt cord docvqa infograph sroie); fi
if [ "${FACE_DATAS+x}" ]; then FACE_DATAS=($FACE_DATAS); else FACE_DATAS=(wflw); fi

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS="${CHUNKS:-${#GPULIST[@]}}"

require_dir() {
    if [ ! -d "$1" ]; then
        echo "Missing $2: $1"
        echo "Clone https://github.com/ZhengrongYue/UniFlow into $SCRIPT_DIR/UniFlow and"
        echo "download weights to $MODEL_DIR (see uniflow download instructions)."
        exit 1
    fi
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "Missing $2: $1"
        echo "Download UniFlow weights to $MODEL_DIR (model.safetensors + config.json)."
        exit 1
    fi
}

require_dir "$UNIFLOW_PATH/uniflow" "UniFlow code directory"
require_file "$CONFIG_PATH/config.json" "UniFlow config.json"
require_file "$CKPT_PATH" "UniFlow checkpoint"

cd "$SCRIPT_DIR"

for DATA in "${TEXT_DATAS[@]}"; do
    for PADDING_SIZE in "${PADDING_SIZES[@]}"; do
        echo "[$MODEL_NAME] padding=$PADDING_SIZE dataset=$DATA (text)"
        for IDX in $(seq 0 $((CHUNKS - 1))); do
            GPU_IDX=$((IDX % ${#GPULIST[@]}))
            CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX]} python uniflow_rec.py \
                --image_path "$DATA_ROOT/images/text_data/$DATA" \
                --save_path "$RECON_ROOT/$MODEL_NAME/text_data/$DATA" \
                --uniflow_path "$UNIFLOW_PATH" \
                --config_path "$CONFIG_PATH" \
                --ckpt_path "$CKPT_PATH" \
                --dtype "$DTYPE" \
                --padding_size "$PADDING_SIZE" \
                --batch_size "$BATCH_SIZE" \
                --num_chunks "$CHUNKS" \
                --chunk_idx "$IDX" &
        done
        wait
    done
done

for DATA in "${FACE_DATAS[@]}"; do
    for PADDING_SIZE in "${PADDING_SIZES[@]}"; do
        echo "[$MODEL_NAME] padding=$PADDING_SIZE dataset=$DATA (face)"
        for IDX in $(seq 0 $((CHUNKS - 1))); do
            GPU_IDX=$((IDX % ${#GPULIST[@]}))
            CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX]} python uniflow_rec.py \
                --image_path "$DATA_ROOT/images/face_data/$DATA" \
                --save_path "$RECON_ROOT/$MODEL_NAME/face_data/$DATA" \
                --uniflow_path "$UNIFLOW_PATH" \
                --config_path "$CONFIG_PATH" \
                --ckpt_path "$CKPT_PATH" \
                --dtype "$DTYPE" \
                --padding_size "$PADDING_SIZE" \
                --batch_size "$BATCH_SIZE" \
                --num_chunks "$CHUNKS" \
                --chunk_idx "$IDX" &
        done
        wait
    done
done

echo "Image reconstruction done -> $RECON_ROOT/$MODEL_NAME"
