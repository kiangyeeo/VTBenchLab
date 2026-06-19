#!/bin/bash
# Reconstruction with the Chameleon VQGAN image tokenizer.
# Usage:
#   CUDA_VISIBLE_DEVICES=0 PADDING_SIZES="256" bash chameleon.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/image_reconstruction_results}"
MODEL_ZOO="${MODEL_ZOO:-$REPO_ROOT/tokenizer_modelzoo}"
MODEL_NAME="${MODEL_NAME:-chameleon}"
CHAMELEON_PATH="${CHAMELEON_PATH:-$SCRIPT_DIR/chameleon}"
CHAMELEON_DIR="${CHAMELEON_DIR:-$MODEL_ZOO/chameleon}"
CFG_PATH="${CFG_PATH:-$CHAMELEON_DIR/vqgan.yaml}"
CKPT_PATH="${CKPT_PATH:-$CHAMELEON_DIR/vqgan.ckpt}"
BATCH_SIZE="${BATCH_SIZE:-1}"
if [ "${PADDING_SIZES+x}" ]; then PADDING_SIZES=($PADDING_SIZES); else PADDING_SIZES=(256 512 1024); fi
if [ "${TEXT_DATAS+x}" ]; then TEXT_DATAS=($TEXT_DATAS); else TEXT_DATAS=(ic13 ic15 textocr tt cord docvqa infograph sroie); fi
if [ "${FACE_DATAS+x}" ]; then FACE_DATAS=($FACE_DATAS); else FACE_DATAS=(wflw); fi

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS="${CHUNKS:-${#GPULIST[@]}}"

require_dir() {
    if [ ! -d "$1" ]; then
        echo "Missing $2: $1"
        echo "See ../reconstruct.md for download instructions."
        exit 1
    fi
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "Missing $2: $1"
        echo "See ../reconstruct.md for download instructions."
        exit 1
    fi
}

require_dir "$CHAMELEON_PATH" "chameleon code directory"
require_file "$CFG_PATH" "Chameleon VQGAN config"
require_file "$CKPT_PATH" "Chameleon VQGAN checkpoint"

cd "$SCRIPT_DIR"

for DATA in "${TEXT_DATAS[@]}"; do
    for PADDING_SIZE in "${PADDING_SIZES[@]}"; do
        echo "[$MODEL_NAME] padding=$PADDING_SIZE dataset=$DATA (text)"
        for IDX in $(seq 0 $((CHUNKS - 1))); do
            GPU_IDX=$((IDX % ${#GPULIST[@]}))
            CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX]} python chameleon_vae_rec.py \
                --image_path "$DATA_ROOT/images/text_data/$DATA" \
                --save_path "$RECON_ROOT/$MODEL_NAME/text_data/$DATA" \
                --chameleon_path "$CHAMELEON_PATH" \
                --cfg_path "$CFG_PATH" \
                --ckpt_path "$CKPT_PATH" \
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
            CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX]} python chameleon_vae_rec.py \
                --image_path "$DATA_ROOT/images/face_data/$DATA" \
                --save_path "$RECON_ROOT/$MODEL_NAME/face_data/$DATA" \
                --chameleon_path "$CHAMELEON_PATH" \
                --cfg_path "$CFG_PATH" \
                --ckpt_path "$CKPT_PATH" \
                --padding_size "$PADDING_SIZE" \
                --batch_size "$BATCH_SIZE" \
                --num_chunks "$CHUNKS" \
                --chunk_idx "$IDX" &
        done
        wait
    done
done

echo "Image reconstruction done -> $RECON_ROOT/$MODEL_NAME"
