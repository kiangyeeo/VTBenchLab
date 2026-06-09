#!/bin/bash
# Reconstruction with the CogVideoX 4x8x8 VAE.
# Usage:
#   CUDA_VISIBLE_DEVICES=0 SHORT_SIZES="256" bash cogvideox_4x8x8.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/video_reconstruction_results}"
MODEL_ZOO="${MODEL_ZOO:-$REPO_ROOT/tokenizer_modelzoo}"
MODEL_NAME="${MODEL_NAME:-cogvideox_4x8x8}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ZOO/CogVideoX1.5-5B/vae}"
BATCH_SIZE="${BATCH_SIZE:-1}"
if [ "${SHORT_SIZES+x}" ]; then SHORT_SIZES=($SHORT_SIZES); else SHORT_SIZES=(480 256); fi
if [ "${TEXT_DATAS+x}" ]; then TEXT_DATAS=($TEXT_DATAS); else TEXT_DATAS=(ds ch3); fi
if [ "${FACE_DATAS+x}" ]; then FACE_DATAS=($FACE_DATAS); else FACE_DATAS=(face_clip_3s); fi

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS="${CHUNKS:-${#GPULIST[@]}}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Missing CogVideoX VAE directory: $MODEL_PATH"
    echo "See ../tokenizer_setup_9x.md for download instructions."
    exit 1
fi

cd "$SCRIPT_DIR"

for DATA in "${TEXT_DATAS[@]}"; do
    for SHORT_SIZE in "${SHORT_SIZES[@]}"; do
        echo "[$MODEL_NAME] short=$SHORT_SIZE dataset=$DATA (text)"
        for IDX in $(seq 0 $((CHUNKS - 1))); do
            GPU_IDX=$((IDX % ${#GPULIST[@]}))
            CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX]} python cogvideo_rec.py \
                --video_path "$DATA_ROOT/videos/text_data/$DATA" \
                --save_path "$RECON_ROOT/$MODEL_NAME/text_data/$DATA" \
                --model_path "$MODEL_PATH" \
                --short_size "$SHORT_SIZE" \
                --batch_size "$BATCH_SIZE" \
                --num_chunks "$CHUNKS" \
                --chunk_idx "$IDX" &
        done
        wait
    done
done

for DATA in "${FACE_DATAS[@]}"; do
    for SHORT_SIZE in "${SHORT_SIZES[@]}"; do
        echo "[$MODEL_NAME] short=$SHORT_SIZE dataset=$DATA (face)"
        for IDX in $(seq 0 $((CHUNKS - 1))); do
            GPU_IDX=$((IDX % ${#GPULIST[@]}))
            CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX]} python cogvideo_rec.py \
                --video_path "$DATA_ROOT/videos/face_data/$DATA" \
                --save_path "$RECON_ROOT/$MODEL_NAME/face_data/$DATA" \
                --model_path "$MODEL_PATH" \
                --short_size "$SHORT_SIZE" \
                --batch_size "$BATCH_SIZE" \
                --num_chunks "$CHUNKS" \
                --chunk_idx "$IDX" &
        done
        wait
    done
done

echo "Video reconstruction done -> $RECON_ROOT/$MODEL_NAME"
