#!/usr/bin/env bash
set -euo pipefail

MATCH_BUDGET_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$MATCH_BUDGET_SCRIPT_DIR/run_common.sh"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 MODEL [linear_probe.py arguments...]" >&2
    exit 2
fi

requested_model="$1"
shift
resolved_model=""
while IFS=$'\t' read -r rank label model head; do
    [[ -n "$rank" && "$rank" != \#* ]] || continue
    if [[ "$requested_model" == "$model" || "$requested_model" == "$rank" ]]; then
        resolved_model="$model"
        break
    fi
done < "$MATCH_BUDGET_SCRIPT_DIR/tokenizers.tsv"

if [[ -z "$resolved_model" ]]; then
    echo "!! '$requested_model' is not one of the 42 configured tokenizers" >&2
    exit 2
fi

for argument in "$@"; do
    if [[ "$argument" == "--model" || "$argument" == --model=* ]]; then
        echo "!! --model cannot be overridden through passthrough arguments" >&2
        exit 2
    fi
done

run_match_budget_probe "$resolved_model" "$@"
