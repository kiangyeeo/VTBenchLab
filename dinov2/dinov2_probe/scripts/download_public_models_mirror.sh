#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

ROOT="${MODEL_ROOT:-../checkpoints/dinov2_baseline}"
MIRROR="${MODEL_MIRROR_BASE:-}"

mkdir -p "$ROOT"

if [[ -z "$MIRROR" ]]; then
  echo "MODEL_MIRROR_BASE is not set. Example:"
  echo "  MODEL_MIRROR_BASE=https://your.mirror/dinov2_baseline bash $0"
  echo "Falling back is intentionally not done here; run download_public_models_official.sh for official sources."
  exit 0
fi

download() {
  local name="$1"
  local url="$MIRROR/$name"
  local out="$ROOT/$name"
  if [[ -f "$out" ]]; then
    echo "[skip] $out"
  else
    echo "[mirror] $url -> $out"
    wget -c "$url" -O "$out"
  fi
}

download dinov2_vits14_pretrain.pth
download dinov2_vitb14_pretrain.pth
download dinov2_vitl14_pretrain.pth
download dinov2_vitg14_pretrain.pth
download ibot_vitl16_checkpoint_teacher.pth

echo "MAE/OpenCLIP local checkpoints can also be placed under $ROOT if your mirror provides them."
