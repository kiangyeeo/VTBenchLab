#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

ROOT="${MODEL_ROOT:-../checkpoints/dinov2_baseline}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_MIRROR_UNSET_PROXY="${HF_MIRROR_UNSET_PROXY:-1}"
FILE_MIRROR="${MODEL_FILE_MIRROR_BASE:-${MODEL_MIRROR_BASE:-}}"

mkdir -p "$ROOT"

echo "Model root: $ROOT"
echo "Hugging Face endpoint: $HF_ENDPOINT"
if [[ "$HF_MIRROR_UNSET_PROXY" == "1" ]]; then
  echo "HF mirror downloads will run with HTTP(S) proxy environment variables unset."
fi

run_hf_cli() {
  if [[ "$HF_MIRROR_UNSET_PROXY" == "1" ]]; then
    env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
      HF_ENDPOINT="$HF_ENDPOINT" huggingface-cli "$@"
  else
    HF_ENDPOINT="$HF_ENDPOINT" huggingface-cli "$@"
  fi
}

hf_download() {
  local repo="$1"
  local out="$2"
  if [[ -d "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
    echo "[skip] $out"
  else
    echo "[hf-mirror] $repo -> $out"
    run_hf_cli download "$repo" --local-dir "$out" --resume-download
  fi
}

hf_download facebook/vit-mae-huge "$ROOT/mae_vith14"

if [[ -z "$FILE_MIRROR" ]]; then
  cat <<EOF

HF mirror only covers Hugging Face Hub models. These baseline weights are not downloaded here:
  - DINOv2 S/B/L/g official .pth files from dl.fbaipublicfiles.com
  - iBOT ViT-L/16 official .pth from the ByteDance public URL
  - Paper OpenCLIP ViT-G/14 LAION-2B explicit local checkpoint

To download the non-HF public weights from official sources, run that separate script
with whatever proxy settings your direct network requires:
  bash dinov2_probe/scripts/download_public_models_official.sh

If you also have a plain file mirror containing the exact .pth files, run:
  MODEL_FILE_MIRROR_BASE=https://your.file.mirror/dinov2_baseline bash $0
EOF
  exit 0
fi

download() {
  local name="$1"
  local url="$FILE_MIRROR/$name"
  local out="$ROOT/$name"
  if [[ -f "$out" ]]; then
    echo "[skip] $out"
  else
    echo "[file-mirror] $url -> $out"
    wget -c "$url" -O "$out"
  fi
}

download dinov2_vits14_pretrain.pth
download dinov2_vitb14_pretrain.pth
download dinov2_vitl14_pretrain.pth
download dinov2_vitg14_pretrain.pth
download ibot_vitl16_checkpoint_teacher.pth

echo "Place the paper OpenCLIP ViT-G/14 LAION-2B checkpoint at $ROOT/openclip_vitg14_laion2b.pt if you have it."
