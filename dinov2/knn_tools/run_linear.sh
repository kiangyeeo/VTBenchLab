#!/usr/bin/env bash
# DINOv2 linear-probing evaluation on ImageNet-1k (single A100-80G).
#
# Usage:
#   bash knn_tools/run_linear.sh <model> [extra args passed to linear.py]
#   <model> = vits14 | vitb14 | vitl14 | vitg14  (append _reg4 for register variants)
# Examples:
#   bash knn_tools/run_linear.sh vitb14
#   bash knn_tools/run_linear.sh vitl14 --epochs 40        # longer schedule -> closer to paper
set -euo pipefail

MODEL="${1:?usage: run_linear.sh <vits14|vitb14|vitl14|vitg14[ _reg4]> [extra args]}"
shift || true

REPO=/cache/ma-user/VTBenchLab/dinov2
DATA=/cache/ma-user/VTBenchLab/data/imagenet1k
EXTRA=$DATA/extra
WEIGHTS=/cache/ma-user/VTBenchLab/checkpoints/dinov2/dinov2_${MODEL}_pretrain.pth
CONFIG=$REPO/dinov2/configs/eval/${MODEL}_pretrain.yaml
OUTDIR=/cache/ma-user/VTBenchLab/outputs/dinov2_linear_${MODEL}

[ -f "$WEIGHTS" ] || { echo "!! missing weights: $WEIGHTS (run download_model.sh $MODEL first)"; exit 1; }
[ -f "$CONFIG" ]  || { echo "!! missing config: $CONFIG"; exit 1; }

cd "$REPO"
mkdir -p "$OUTDIR"

# Default to a long (100-epoch) schedule unless the caller overrides --epochs.
EPOCHS_ARG=()
case " $* " in *" --epochs "*) ;; *) EPOCHS_ARG=(--epochs 100) ;; esac
echo ">> linear  model=$MODEL  out=$OUTDIR  epochs=${EPOCHS_ARG[*]:-(from extra args)}  extra='$*'"

# --standalone -> torchrun picks a free rendezvous port (safe for concurrent launches).
PYTHONPATH=. torchrun --standalone --nproc_per_node=1 dinov2/eval/linear.py \
    --config-file "$CONFIG" \
    --pretrained-weights "$WEIGHTS" \
    --output-dir "$OUTDIR" \
    --train-dataset "ImageNet:split=TRAIN:root=$DATA:extra=$EXTRA" \
    --val-dataset   "ImageNet:split=VAL:root=$DATA:extra=$EXTRA" \
    "${EPOCHS_ARG[@]}" "$@"

echo ">> Results (best_classifier accuracy) -> $OUTDIR/results_eval_linear.json"
