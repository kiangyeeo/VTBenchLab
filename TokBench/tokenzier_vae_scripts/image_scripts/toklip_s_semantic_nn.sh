#!/bin/bash
# Reconstruction from TokLIP-S semantic transformer tokens via nearest VQ-code projection.
# Usage:
#   CUDA_VISIBLE_DEVICES=0 PADDING_SIZES="256" bash toklip_s_semantic_nn.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LATENT_SOURCE="${LATENT_SOURCE:-semantic_nn}"
export MODEL_NAME="${MODEL_NAME:-toklip_s_semantic_nn}"
if [ ! "${PADDING_SIZES+x}" ]; then
    export PADDING_SIZES="256"
fi

exec bash "$SCRIPT_DIR/toklip_s.sh" "$@"
