#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_common.sh"
FEATURE_MICROBATCH_SIZE="${FEATURE_MICROBATCH_SIZE:-1024}"
STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-1}"
run_tokenizer_linear_probe dinov3_vitl16_lvd1689m \
    --feature-microbatch-size "$FEATURE_MICROBATCH_SIZE" \
    --stop-after-epoch "$STOP_AFTER_EPOCH" \
    "$@"
