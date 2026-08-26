#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_common.sh"
FEATURE_MICROBATCH_SIZE="${FEATURE_MICROBATCH_SIZE:-16}"
run_tokenizer_linear_probe dinov2_large --feature-microbatch-size "$FEATURE_MICROBATCH_SIZE" "$@"
