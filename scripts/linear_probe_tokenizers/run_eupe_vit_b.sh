#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_common.sh"
run_tokenizer_linear_probe eupe_vit_b --feature-microbatch-size 1024 "$@"
