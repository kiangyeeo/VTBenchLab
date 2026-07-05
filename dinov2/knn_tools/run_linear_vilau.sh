#!/usr/bin/env bash
# VILA-U semantic-latent linear-probing evaluation on ImageNet-1k.
set -euo pipefail

REPO="${REPO:-/cache/ma-user/VTBenchLab/dinov2}"
DATA="${DATA:-/cache/ma-user/VTBenchLab/data/imagenet1k}"
EXTRA="${EXTRA:-$DATA/extra}"
OUTDIR="${OUTDIR:-/cache/ma-user/VTBenchLab/outputs/vae_linear_probing/vilau_7b_256_semantic_penultimate}"
VILAU_PATH="${VILAU_PATH:-/cache/ma-user/VTBenchLab/TokBench/tokenzier_vae_scripts/image_scripts/vila-u}"
MODEL_PATH="${MODEL_PATH:-/cache/ma-user/VTBenchLab/TokBench/tokenizer_modelzoo/VILA-U/vila-u-7b-256}"
SIGLIP_CONFIG_PATH="${SIGLIP_CONFIG_PATH:-/cache/ma-user/VTBenchLab/TokBench/tokenizer_modelzoo/VILA-U/siglip-large-patch16-256}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

[ -d "$DATA" ] || { echo "!! missing ImageNet root: $DATA"; exit 1; }
[ -d "$EXTRA" ] || { echo "!! missing ImageNet extra dir: $EXTRA"; exit 1; }
[ -d "$VILAU_PATH" ] || { echo "!! missing VILA-U code dir: $VILAU_PATH"; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "!! missing VILA-U model path: $MODEL_PATH"; exit 1; }
[ -d "$SIGLIP_CONFIG_PATH" ] || { echo "!! missing SigLIP config path: $SIGLIP_CONFIG_PATH"; exit 1; }

cd "$REPO"
mkdir -p "$OUTDIR"

echo ">> VILA-U semantic linear probing"
echo "   out=$OUTDIR"
echo "   model=$MODEL_PATH"
echo "   extra args=$*"

PYTHONPATH=. torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" knn_tools/run_linear_vilau.py \
    --output-dir "$OUTDIR" \
    --vilau-path "$VILAU_PATH" \
    --model-path "$MODEL_PATH" \
    --siglip-config-path "$SIGLIP_CONFIG_PATH" \
    --feature semantic \
    --semantic-layer penultimate \
    --train-dataset "ImageNet:split=TRAIN:root=$DATA:extra=$EXTRA" \
    --val-dataset   "ImageNet:split=VAL:root=$DATA:extra=$EXTRA" \
    "$@"

echo ">> Results -> $OUTDIR/results_eval_linear.json"
