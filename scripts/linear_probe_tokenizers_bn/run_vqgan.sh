#!/usr/bin/env bash
# Taming VQGAN ImageNet-f16 (16,384-entry, 256-dimensional codebook).
# Keep the frozen-backbone microbatch identical to the no-BN launcher; the
# extracted FP32 features are concatenated into batch 1024 before BatchNorm.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_common.sh"
FEATURE_MICROBATCH_SIZE="${FEATURE_MICROBATCH_SIZE:-256}"
run_tokenizer_linear_probe_bn vqgan \
    --feature-microbatch-size "$FEATURE_MICROBATCH_SIZE" \
    "$@"
