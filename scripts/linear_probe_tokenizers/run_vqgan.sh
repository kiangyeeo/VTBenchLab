#!/usr/bin/env bash
# Taming VQGAN ImageNet-f16 (16,384-entry, 256-dimensional codebook).
# The probe loader does not require PyTorch Lightning.
# Override VQGAN_FEATURE_MICROBATCH_SIZE or pass --feature-microbatch-size
# explicitly if the default does not fit or underutilizes the visible GPU.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_common.sh"
VQGAN_FEATURE_MICROBATCH_SIZE="${VQGAN_FEATURE_MICROBATCH_SIZE:-16}"
run_tokenizer_linear_probe vqgan \
    --feature-microbatch-size "$VQGAN_FEATURE_MICROBATCH_SIZE" \
    "$@"
