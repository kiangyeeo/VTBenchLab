#!/usr/bin/env bash
# Run DINOv2 k-NN evaluation on ImageNet-1k (single A100-80G).
#
# Usage:
#   bash knn_tools/run_knn.sh <model> [batch_size]
#   <model> = vits14 | vitb14 | vitl14 | vitg14   (append _reg4 for register variants)
# Examples:
#   bash knn_tools/run_knn.sh vits14
#   bash knn_tools/run_knn.sh vitg14          # auto-uses batch 128 (1.1B params)
set -euo pipefail

MODEL="${1:?usage: run_knn.sh <vits14|vitb14|vitl14|vitg14[ _reg4]> [batch_size]}"

REPO=/cache/ma-user/VTBenchLab/dinov2
DATA=/cache/ma-user/VTBenchLab/data/imagenet1k
EXTRA=$DATA/extra
WEIGHTS=/cache/ma-user/VTBenchLab/checkpoints/dinov2/dinov2_${MODEL}_pretrain.pth
CONFIG=$REPO/dinov2/configs/eval/${MODEL}_pretrain.yaml
OUTDIR=/cache/ma-user/VTBenchLab/outputs/dinov2_knn_${MODEL}

# Bigger models -> smaller default batch to stay within 80 GB during feature extraction.
case "$MODEL" in
  vitg14*) DEFAULT_BS=128 ;;
  *)       DEFAULT_BS=256 ;;
esac
BS="${2:-$DEFAULT_BS}"

[ -f "$WEIGHTS" ] || { echo "!! missing weights: $WEIGHTS (run download_model.sh $MODEL first)"; exit 1; }
[ -f "$CONFIG" ]  || { echo "!! missing config: $CONFIG"; exit 1; }

cd "$REPO"
mkdir -p "$OUTDIR"
echo ">> model=$MODEL  batch=$BS  out=$OUTDIR"

# Call dinov2/eval/knn.py directly (run/eval/knn.py is SLURM/submitit only).
# torchrun --nproc_per_node=1 = single GPU; bump it for multi-GPU on this node.
PYTHONPATH=. torchrun --nproc_per_node=1 dinov2/eval/knn.py \
    --config-file "$CONFIG" \
    --pretrained-weights "$WEIGHTS" \
    --output-dir "$OUTDIR" \
    --train-dataset "ImageNet:split=TRAIN:root=$DATA:extra=$EXTRA" \
    --val-dataset   "ImageNet:split=VAL:root=$DATA:extra=$EXTRA" \
    --nb_knn 10 20 100 200 \
    --temperature 0.07 \
    --batch-size "$BS"

echo ">> Results written to $OUTDIR/results_eval_knn.json"
