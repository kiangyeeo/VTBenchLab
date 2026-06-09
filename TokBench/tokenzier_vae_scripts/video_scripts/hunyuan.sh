#!/bin/bash
# Video reconstruction with the HunyuanVideo VAE tokenizer.
# Usage:
#   CUDA_VISIBLE_DEVICES=0 SHORT_SIZES="256 480" bash hunyuan.sh
set -e

# ---- config (override via env) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/tokbench_data}"    # contains videos/ and video_annotations/
RECON_ROOT="${RECON_ROOT:-$REPO_ROOT/video_reconstruction_results}"
MODEL_NAME="${MODEL_NAME:-hunyuan}"
MODEL_ZOO="${MODEL_ZOO:-$REPO_ROOT/tokenizer_modelzoo}"
VAE_PATH="${VAE_PATH:-$MODEL_ZOO/hunyuan-video-t2v-720p/vae}"
HUNYUANVIDEO_PATH="${HUNYUANVIDEO_PATH:-$SCRIPT_DIR/HunyuanVideo}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SHORT_SIZES=(${SHORT_SIZES:-256})

# Download only the VAE tokenizer weights by default. The full HunyuanVideo model
# is much larger and is not needed for reconstruction evaluation.
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_REPO="${HF_REPO:-tencent/HunyuanVideo}"
HF_INCLUDE="${HF_INCLUDE:-hunyuan-video-t2v-720p/vae/*}"

# The tokenizer script imports hyvideo.vae.load_vae from Tencent's HunyuanVideo code.
DOWNLOAD_CODE="${DOWNLOAD_CODE:-1}"
HUNYUANVIDEO_GIT_URL="${HUNYUANVIDEO_GIT_URL:-https://github.com/Tencent-Hunyuan/HunyuanVideo.git}"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS="${CHUNKS:-${#GPULIST[@]}}"

download_vae_if_needed() {
    if [ -f "$VAE_PATH/config.json" ] && { [ -f "$VAE_PATH/pytorch_model.pt" ] || [ -f "$VAE_PATH/diffusion_pytorch_model.safetensors" ]; }; then
        return
    fi

    if [ "$DOWNLOAD_MODEL" != "1" ]; then
        echo "Missing Hunyuan VAE path: $VAE_PATH"
        echo "Download it first or pass VAE_PATH=/path/to/hunyuan-video-t2v-720p/vae."
        exit 1
    fi

    if ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "huggingface-cli not found. Install huggingface-hub or set DOWNLOAD_MODEL=0 and VAE_PATH manually."
        exit 1
    fi

    mkdir -p "$MODEL_ZOO"
    echo "Downloading HunyuanVideo VAE from $HF_REPO via HF_ENDPOINT=$HF_ENDPOINT"
    HF_ENDPOINT="$HF_ENDPOINT" huggingface-cli download "$HF_REPO" \
        --include "$HF_INCLUDE" \
        --local-dir "$MODEL_ZOO"
}

download_code_if_needed() {
    if [ -d "$HUNYUANVIDEO_PATH/hyvideo" ]; then
        return
    fi

    if [ -d "$HUNYUANVIDEO_PATH/HunyuanVideo/hyvideo" ]; then
        HUNYUANVIDEO_PATH="$HUNYUANVIDEO_PATH/HunyuanVideo"
        return
    fi

    if [ "$DOWNLOAD_CODE" != "1" ]; then
        echo "Missing HunyuanVideo code path: $HUNYUANVIDEO_PATH"
        echo "Clone Tencent-Hunyuan/HunyuanVideo there or pass HUNYUANVIDEO_PATH=/path/to/HunyuanVideo."
        exit 1
    fi

    if ! command -v git >/dev/null 2>&1; then
        echo "git not found. Clone HunyuanVideo manually or pass HUNYUANVIDEO_PATH."
        exit 1
    fi

    echo "Cloning HunyuanVideo code into $HUNYUANVIDEO_PATH"
    git clone --depth 1 "$HUNYUANVIDEO_GIT_URL" "$HUNYUANVIDEO_PATH"
}

download_vae_if_needed
download_code_if_needed

cd "$SCRIPT_DIR"
export HUNYUANVIDEO_PATH

# ---- text videos ----
DATAS=("ds" "ch3")
for DATA in "${DATAS[@]}"; do
    for SHORT_SIZE in "${SHORT_SIZES[@]}"; do
        echo "[$MODEL_NAME] short=$SHORT_SIZE dataset=$DATA (text)"
        for IDX in $(seq 0 $((CHUNKS-1))); do
            GPU_IDX=$((IDX % ${#GPULIST[@]}))
            CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX]} python hunyuan_rec.py \
                --video_path "$DATA_ROOT/videos/text_data/$DATA" \
                --save_path "$RECON_ROOT/$MODEL_NAME/text_data/$DATA" \
                --model_path "$VAE_PATH" \
                --short_size "$SHORT_SIZE" \
                --batch_size "$BATCH_SIZE" \
                --num_chunks "$CHUNKS" \
                --chunk_idx "$IDX" &
        done
        wait
    done
done

# ---- face videos ----
DATAS=("face_clip_3s")
for DATA in "${DATAS[@]}"; do
    for SHORT_SIZE in "${SHORT_SIZES[@]}"; do
        echo "[$MODEL_NAME] short=$SHORT_SIZE dataset=$DATA (face)"
        for IDX in $(seq 0 $((CHUNKS-1))); do
            GPU_IDX=$((IDX % ${#GPULIST[@]}))
            CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX]} python hunyuan_rec.py \
                --video_path "$DATA_ROOT/videos/face_data/$DATA" \
                --save_path "$RECON_ROOT/$MODEL_NAME/face_data/$DATA" \
                --model_path "$VAE_PATH" \
                --short_size "$SHORT_SIZE" \
                --batch_size "$BATCH_SIZE" \
                --num_chunks "$CHUNKS" \
                --chunk_idx "$IDX" &
        done
        wait
    done
done

echo "Video reconstruction done -> $RECON_ROOT/$MODEL_NAME"
