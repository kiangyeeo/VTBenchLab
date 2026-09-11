#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_common.sh"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 MODEL_OR_RANK [linear_probe.py arguments...]" >&2
    exit 2
fi

requested="$1"
shift
resolved=""
while IFS=$'\t' read -r rank label model head; do
    [[ -n "$rank" && "$rank" != \#* ]] || continue
    if [[ "$requested" == "$rank" || "$requested" == "$label" || "$requested" == "$model" ]]; then
        resolved="$model"
        break
    fi
done < "$SCRIPT_DIR/tokenizers.tsv"

if [[ -z "$resolved" ]]; then
    echo "!! '$requested' is not one of the 64 configured tokenizers" >&2
    exit 2
fi

for argument in "$@"; do
    if [[ "$argument" == "--model" || "$argument" == --model=* ]]; then
        echo "!! --model cannot be overridden through passthrough arguments" >&2
        exit 2
    fi
done

run_match_budget_5shot_probe "$resolved" "$@"
