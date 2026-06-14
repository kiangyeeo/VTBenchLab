#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

ROOT="${MODEL_ROOT:-../checkpoints/dinov2_baseline}"
mkdir -p "$ROOT"

download() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "[skip] $out"
  else
    echo "[official] $url -> $out"
    wget -c "$url" -O "$out"
  fi
}

download https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth "$ROOT/dinov2_vits14_pretrain.pth"
download https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth "$ROOT/dinov2_vitb14_pretrain.pth"
download https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth "$ROOT/dinov2_vitl14_pretrain.pth"
download https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth "$ROOT/dinov2_vitg14_pretrain.pth"
download https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16/checkpoint_teacher.pth "$ROOT/ibot_vitl16_checkpoint_teacher.pth"

echo "For MAE ViT-H/14, run with --allow-download once or pre-populate $ROOT/mae_vith14 from Hugging Face facebook/vit-mae-huge."
echo "For the paper OpenCLIP ViT-G/14 LAION-2B alias, place your explicit checkpoint at $ROOT/openclip_vitg14_laion2b.pt."
