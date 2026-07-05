#!/bin/bash
# Reconstruction from TokLIP-L semantic transformer tokens via nearest VQ-code projection.
# Usage:
#   CUDA_VISIBLE_DEVICES=0 PADDING_SIZES="384" bash toklip_l_semantic_nn.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LATENT_SOURCE="${LATENT_SOURCE:-semantic_nn}"
export MODEL_NAME="${MODEL_NAME:-toklip_l_semantic_nn}"
if [ ! "${PADDING_SIZES+x}" ]; then
    export PADDING_SIZES="384"
fi

exec bash "$SCRIPT_DIR/toklip_l.sh" "$@"
