#!/usr/bin/env bash
# Download a DINOv2 pretrained backbone (.pth) used by k-NN eval.
#
# NOTE: this repo's eval code loads the ORIGINAL Meta backbone checkpoint
# (.pth), NOT the HuggingFace `facebook/dinov2-*` transformers format.
# The original .pth lives on dl.fbaipublicfiles.com (directly reachable).
#
# Usage:
#   bash knn_tools/download_model.sh <model> [dest_dir]
#   <model> = vits14 | vitb14 | vitl14 | vitg14   (append _reg4 for register variants)
# Examples:
#   bash knn_tools/download_model.sh vits14
#   bash knn_tools/download_model.sh vitg14
set -euo pipefail

MODEL="${1:?usage: download_model.sh <vits14|vitb14|vitl14|vitg14[ _reg4]> [dest_dir]}"
DEST="${2:-/cache/ma-user/VTBenchLab/checkpoints/dinov2}"
mkdir -p "$DEST"

ARCH="${MODEL%_reg4}"   # registers share the same arch download directory
URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_${ARCH}/dinov2_${MODEL}_pretrain.pth"
OUT="$DEST/dinov2_${MODEL}_pretrain.pth"

echo ">> Downloading $MODEL backbone"
echo "   $URL"
# Resume a partial download if re-run. Prefer wget, fall back to curl.
if command -v wget >/dev/null 2>&1; then
  wget -c -O "$OUT" "$URL"
else
  curl -fL -C - -o "$OUT" "$URL"
fi

echo ">> Done. Size:"
ls -lh "$OUT"
