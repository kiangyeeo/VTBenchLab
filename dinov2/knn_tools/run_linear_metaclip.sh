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
MODEL="${MODEL:-vit_base_patch16_clip_224.metaclip_2pt5b}"
CKPT_PATH="${CKPT_PATH:-/cache/ma-user/VTBenchLab/TokBench/tokenizer_modelzoo/MetaCLIP/vit_base_patch16_clip_224.metaclip_2pt5b}"
LEGACY_OUTDIR="/cache/ma-user/VTBenchLab/outputs/vae_linear_probing/metaclip_b16_2pt5b"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

source "$REPO/knn_tools/linear_dataset_common.sh"
setup_linear_dataset_args "metaclip" "$LEGACY_OUTDIR" "$@"
setup_linear_training_args "$@"

[ -e "$CKPT_PATH" ] || { echo "!! missing MetaCLIP checkpoint path: $CKPT_PATH"; exit 1; }

cd "$REPO"
mkdir -p "$OUTDIR"

echo ">> MetaCLIP linear probing"
echo "   dataset=$DATASET"
echo "   model=$MODEL"
echo "   out=$OUTDIR"
echo "   ckpt=$CKPT_PATH"
echo "   batch_size=$BATCH_SIZE epochs=$EPOCHS epoch_length=$EPOCH_LENGTH eval_period=$EVAL_PERIOD_ITERATIONS"
echo "   extra args=$*"

PYTHONPATH=. torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" knn_tools/run_linear_metaclip.py \
    --model "$MODEL" \
    --output-dir "$OUTDIR" \
    --checkpoint-path "$CKPT_PATH" \
    --train-dataset "$TRAIN_DATASET" \
    --val-dataset "$VAL_DATASET" \
    "${TEST_DATASET_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "$@"

echo ">> Results -> $OUTDIR/results_eval_linear.json"
