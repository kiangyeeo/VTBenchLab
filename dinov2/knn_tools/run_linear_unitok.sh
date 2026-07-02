#!/usr/bin/env bash
# UniTok linear-probing evaluation on ImageNet-1k.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash knn_tools/run_linear_unitok.sh [extra args]
# Example:
#   bash knn_tools/run_linear_unitok.sh --epochs 10 --batch-size 64
set -euo pipefail

REPO="${REPO:-/cache/ma-user/VTBenchLab/dinov2}"
DATA="${DATA:-/cache/ma-user/VTBenchLab/data/imagenet1k}"
EXTRA="${EXTRA:-$DATA/extra}"
OUTDIR="${OUTDIR:-/cache/ma-user/VTBenchLab/outputs/vae_linear_probing/unitok}"
UNITOK_PATH="${UNITOK_PATH:-/cache/ma-user/VTBenchLab/TokBench/tokenzier_vae_scripts/image_scripts/UniTok}"
CKPT_PATH="${CKPT_PATH:-/cache/ma-user/VTBenchLab/TokBench/tokenizer_modelzoo/unitok_20250227/unitok_tokenizer.pth}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

[ -d "$DATA" ] || { echo "!! missing ImageNet root: $DATA"; exit 1; }
[ -d "$EXTRA" ] || { echo "!! missing ImageNet extra dir: $EXTRA"; exit 1; }
[ -d "$UNITOK_PATH" ] || { echo "!! missing UniTok code dir: $UNITOK_PATH"; exit 1; }
[ -f "$CKPT_PATH" ] || { echo "!! missing UniTok checkpoint: $CKPT_PATH"; exit 1; }

cd "$REPO"
mkdir -p "$OUTDIR"

echo ">> unitok linear probing"
echo "   out=$OUTDIR"
echo "   ckpt=$CKPT_PATH"
echo "   extra args=$*"

PYTHONPATH=. torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" knn_tools/run_linear_unitok.py \
    --output-dir "$OUTDIR" \
    --ckpt-path "$CKPT_PATH" \
    --unitok-path "$UNITOK_PATH" \
    --train-dataset "ImageNet:split=TRAIN:root=$DATA:extra=$EXTRA" \
    --val-dataset   "ImageNet:split=VAL:root=$DATA:extra=$EXTRA" \
    "$@"

echo ">> Results -> $OUTDIR/results_eval_linear.json"
