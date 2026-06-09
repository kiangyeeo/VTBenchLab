#!/bin/bash
# Reconstruction with the SDXL VAE tokenizer.
# Usage:
#   PADDING_SIZES="256 512 1024" bash sdxl.sh
set -e

# ---- config (override via env, e.g. `CUDA_VISIBLE_DEVICES=0,1 bash sdxl.sh`) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"    # contains images/ and annotations/
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/image_reconstruction_results}"
MODEL_NAME="${MODEL_NAME:-sdxl}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/tokenizer_modelzoo/sdxl-vae}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PADDING_SIZES=(${PADDING_SIZES:-256})

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS="${CHUNKS:-${#GPULIST[@]}}"

cd "$SCRIPT_DIR"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Missing SDXL VAE path: $MODEL_PATH"
    echo "Download it first or pass MODEL_PATH=/path/to/sdxl-vae."
    exit 1
fi

DATAS=("ic13" "ic15" "textocr" "tt" "cord" "docvqa" "infograph" "sroie")
for DATA in "${DATAS[@]}"; do
    for PADDING_SIZE in "${PADDING_SIZES[@]}"; do
        echo "[$MODEL_NAME] padding=$PADDING_SIZE dataset=$DATA (text)"
        for IDX in $(seq 0 $((CHUNKS-1))); do
            GPU_IDX=$((IDX % ${#GPULIST[@]}))
            CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX]} python sdxl_vae_rec.py \
                --image_path "$DATA_ROOT/images/text_data/$DATA" \
                --model_name "$MODEL_NAME" \
                --model_path "$MODEL_PATH" \
                --save_path "$RECON_ROOT/$MODEL_NAME/text_data/$DATA" \
                --padding_size "$PADDING_SIZE" \
                --batch_size "$BATCH_SIZE" \
                --num_chunks "$CHUNKS" \
                --chunk_idx "$IDX" &
        done
        wait
    done
done

DATAS=("wflw")
for DATA in "${DATAS[@]}"; do
    for PADDING_SIZE in "${PADDING_SIZES[@]}"; do
        echo "[$MODEL_NAME] padding=$PADDING_SIZE dataset=$DATA (face)"
        for IDX in $(seq 0 $((CHUNKS-1))); do
            GPU_IDX=$((IDX % ${#GPULIST[@]}))
            CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX]} python sdxl_vae_rec.py \
                --image_path "$DATA_ROOT/images/face_data/$DATA" \
                --model_name "$MODEL_NAME" \
                --model_path "$MODEL_PATH" \
                --save_path "$RECON_ROOT/$MODEL_NAME/face_data/$DATA" \
                --padding_size "$PADDING_SIZE" \
                --batch_size "$BATCH_SIZE" \
                --num_chunks "$CHUNKS" \
                --chunk_idx "$IDX" &
        done
        wait
    done
done

echo "Reconstruction done -> $RECON_ROOT/$MODEL_NAME"
