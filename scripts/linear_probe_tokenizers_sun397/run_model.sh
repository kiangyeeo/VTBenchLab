#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_common.sh"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 MODEL [linear_probe.py arguments...]" >&2
    exit 2
fi

model="$1"
shift
run_tokenizer_sun397_probe "$model" "$@"
