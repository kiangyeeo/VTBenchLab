#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_common.sh"
FEATURE_MICROBATCH_SIZE="${FEATURE_MICROBATCH_SIZE:-8}"
run_tokenizer_linear_probe pe_core_l14_336 --feature-microbatch-size "$FEATURE_MICROBATCH_SIZE" "$@"
