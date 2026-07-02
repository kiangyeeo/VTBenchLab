#!/usr/bin/env bash
# LlamaGen VQ linear-probing evaluation on ImageNet-1k.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash knn_tools/run_linear_llamagen.sh [extra args]
#   VARIANT=vq16 CUDA_VISIBLE_DEVICES=1 bash knn_tools/run_linear_llamagen.sh
# Examples:
#   bash knn_tools/run_linear_llamagen.sh --epochs 10
#   bash knn_tools/run_linear_llamagen.sh --variant vq16 --batch-size 768
set -euo pipefail

REPO="${REPO:-/cache/ma-user/VTBenchLab/dinov2}"
DATA="${DATA:-/cache/ma-user/VTBenchLab/data/imagenet1k}"
EXTRA="${EXTRA:-$DATA/extra}"
VARIANT="${VARIANT:-vq8}"
LLAMAGEN_PATH="${LLAMAGEN_PATH:-/cache/ma-user/VTBenchLab/TokBench/tokenzier_vae_scripts/image_scripts/LlamaGen}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

case "$VARIANT" in
    vq8)
        MODEL_NAME="llamagen_vq8"
        CKPT_PATH="${CKPT_PATH:-/cache/ma-user/VTBenchLab/TokBench/tokenizer_modelzoo/LlamaGen/vq_ds8_c2i.pt}"
        ;;
    vq16)
        MODEL_NAME="llamagen_vq16"
        CKPT_PATH="${CKPT_PATH:-/cache/ma-user/VTBenchLab/TokBench/tokenizer_modelzoo/LlamaGen/vq_ds16_c2i.pt}"
        ;;
    *)
        echo "!! unknown VARIANT=$VARIANT (expected vq8 or vq16)"
        exit 1
        ;;
esac

OUTDIR="${OUTDIR:-/cache/ma-user/VTBenchLab/outputs/vae_linear_probing/$MODEL_NAME}"

[ -d "$DATA" ] || { echo "!! missing ImageNet root: $DATA"; exit 1; }
[ -d "$EXTRA" ] || { echo "!! missing ImageNet extra dir: $EXTRA"; exit 1; }
[ -d "$LLAMAGEN_PATH/tokenizer/tokenizer_image" ] || { echo "!! missing LlamaGen tokenizer_image dir: $LLAMAGEN_PATH/tokenizer/tokenizer_image"; exit 1; }
[ -f "$CKPT_PATH" ] || { echo "!! missing LlamaGen checkpoint: $CKPT_PATH"; exit 1; }

cd "$REPO"
mkdir -p "$OUTDIR"

echo ">> LlamaGen linear probing"
echo "   variant=$VARIANT"
echo "   out=$OUTDIR"
echo "   ckpt=$CKPT_PATH"
echo "   batch_size=$BATCH_SIZE"
echo "   extra args=$*"

if [[ " $* " == *" --batch-size "* ]]; then
    BATCH_SIZE_ARG=()
else
    BATCH_SIZE_ARG=(--batch-size "$BATCH_SIZE")
fi

if [[ " $* " == *" --variant "* ]]; then
    VARIANT_ARG=()
else
    VARIANT_ARG=(--variant "$VARIANT")
fi

PYTHONPATH=. torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" knn_tools/run_linear_llamagen.py \
    --output-dir "$OUTDIR" \
    --ckpt-path "$CKPT_PATH" \
    --llamagen-path "$LLAMAGEN_PATH" \
    --train-dataset "ImageNet:split=TRAIN:root=$DATA:extra=$EXTRA" \
    --val-dataset   "ImageNet:split=VAL:root=$DATA:extra=$EXTRA" \
    "${VARIANT_ARG[@]}" \
    "${BATCH_SIZE_ARG[@]}" \
    "$@"

echo ">> Results -> $OUTDIR/results_eval_linear.json"
