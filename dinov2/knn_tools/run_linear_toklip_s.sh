#!/usr/bin/env bash
# TokLIP-S semantic-token linear-probing evaluation on ImageNet-1k.
set -euo pipefail

REPO="${REPO:-/cache/ma-user/VTBenchLab/dinov2}"
LEGACY_OUTDIR="/cache/ma-user/VTBenchLab/outputs/vae_linear_probing/toklip_s_semantic_256"
TOKLIP_PATH="${TOKLIP_PATH:-/cache/ma-user/VTBenchLab/TokBench/tokenzier_vae_scripts/image_scripts/TokLIP}"
CKPT_PATH="${CKPT_PATH:-/cache/ma-user/VTBenchLab/TokBench/tokenizer_modelzoo/TokLIP/TokLIP_S_256.pt}"
VQ_CKPT_PATH="${VQ_CKPT_PATH:-/cache/ma-user/VTBenchLab/TokBench/tokenizer_modelzoo/TokLIP/vq_ds16_t2i.pt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

source "$REPO/knn_tools/linear_dataset_common.sh"
setup_linear_dataset_args "toklips" "$LEGACY_OUTDIR" "$@"
setup_linear_training_args "$@"

[ -d "$TOKLIP_PATH" ] || { echo "!! missing TokLIP code dir: $TOKLIP_PATH"; exit 1; }
[ -f "$CKPT_PATH" ] || { echo "!! missing TokLIP-S checkpoint: $CKPT_PATH"; exit 1; }
[ -f "$VQ_CKPT_PATH" ] || { echo "!! missing TokLIP VQ checkpoint: $VQ_CKPT_PATH"; exit 1; }

cd "$REPO"
mkdir -p "$OUTDIR"

echo ">> TokLIP-S semantic linear probing"
echo "   dataset=$DATASET"
echo "   out=$OUTDIR"
echo "   ckpt=$CKPT_PATH"
echo "   vq=$VQ_CKPT_PATH"
echo "   batch_size=$BATCH_SIZE epochs=$EPOCHS epoch_length=$EPOCH_LENGTH eval_period=$EVAL_PERIOD_ITERATIONS"
echo "   extra args=$*"

PYTHONPATH=. torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" knn_tools/run_linear_toklip.py \
    --variant s \
    --feature semantic \
    --output-dir "$OUTDIR" \
    --toklip-path "$TOKLIP_PATH" \
    --ckpt-path "$CKPT_PATH" \
    --vq-ckpt-path "$VQ_CKPT_PATH" \
    --train-dataset "$TRAIN_DATASET" \
    --val-dataset "$VAL_DATASET" \
    "${TEST_DATASET_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "$@"

echo ">> Results -> $OUTDIR/results_eval_linear.json"
