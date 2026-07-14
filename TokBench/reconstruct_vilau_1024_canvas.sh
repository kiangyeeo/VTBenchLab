#!/usr/bin/env bash
# Reconstruct TokBench images through the native-256 VILA-U tokenizer while
# using a 1024x1024 TokBench padding canvas.
#
# This is a canvas/interpolation experiment. VILA-U still receives 256x256
# inputs internally; this script does not turn it into a native-1024 tokenizer.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_SCRIPT_DIR="$REPO_ROOT/tokenzier_vae_scripts/image_scripts"

export PADDING_SIZES="1024"

exec bash "$IMAGE_SCRIPT_DIR/vilau_7b_256.sh"
