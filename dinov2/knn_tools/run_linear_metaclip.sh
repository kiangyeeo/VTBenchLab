#!/usr/bin/env bash
# MetaCLIP linear-probing evaluation on ImageNet-1k.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash knn_tools/run_linear_metaclip.sh [extra args]
# Examples:
#   bash knn_tools/run_linear_metaclip.sh --epochs 10 --batch-size 64
#   CKPT_PATH=/path/to/metaclip_dir bash knn_tools/run_linear_metaclip.sh
set -euo pipefail

REPO="${REPO:-/cache/ma-user/VTBenchLab/dinov2}"
DATA="${DATA:-/cache/ma-user/VTBenchLab/data/imagenet1k}"
EXTRA="${EXTRA:-$DATA/extra}"
MODEL="${MODEL:-vit_base_patch16_clip_224.metaclip_2pt5b}"
CKPT_PATH="${CKPT_PATH:-/cache/ma-user/VTBenchLab/TokBench/tokenizer_modelzoo/MetaCLIP/vit_base_patch16_clip_224.metaclip_2pt5b}"
OUTDIR="${OUTDIR:-/cache/ma-user/VTBenchLab/outputs/vae_linear_probing/metaclip_b16_2pt5b}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

[ -d "$DATA" ] || { echo "!! missing ImageNet root: $DATA"; exit 1; }
[ -d "$EXTRA" ] || { echo "!! missing ImageNet extra dir: $EXTRA"; exit 1; }
[ -e "$CKPT_PATH" ] || { echo "!! missing MetaCLIP checkpoint path: $CKPT_PATH"; exit 1; }

cd "$REPO"
mkdir -p "$OUTDIR"

echo ">> MetaCLIP linear probing"
echo "   model=$MODEL"
echo "   out=$OUTDIR"
echo "   ckpt=$CKPT_PATH"
echo "   batch_size=$BATCH_SIZE"
echo "   extra args=$*"

if [[ " $* " == *" --batch-size "* ]]; then
    BATCH_SIZE_ARG=()
else
    BATCH_SIZE_ARG=(--batch-size "$BATCH_SIZE")
fi

PYTHONPATH=. torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" knn_tools/run_linear_metaclip.py \
    --model "$MODEL" \
    --output-dir "$OUTDIR" \
    --checkpoint-path "$CKPT_PATH" \
    --train-dataset "ImageNet:split=TRAIN:root=$DATA:extra=$EXTRA" \
    --val-dataset   "ImageNet:split=VAL:root=$DATA:extra=$EXTRA" \
    "${BATCH_SIZE_ARG[@]}" \
    "$@"

echo ">> Results -> $OUTDIR/results_eval_linear.json"
