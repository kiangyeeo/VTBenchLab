#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_common.sh"
MANIFEST="$SCRIPT_DIR/../linear_probe_match_budget_5shot_noaug_cached_10epoch/tokenizers.tsv"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 MODEL_OR_RANK [linear_probe_sweep.py arguments...]" >&2
    exit 2
fi
requested="$1"
shift
resolved=""
while IFS=$'\t' read -r rank requested_id model head; do
    [[ -n "$rank" && "$rank" != \#* ]] || continue
    if [[ "$requested" == "$rank" || "$requested" == "$requested_id" || "$requested" == "$model" ]]; then
        resolved="$model"
        break
    fi
done < "$MANIFEST"
if [[ -z "$resolved" ]]; then
    echo "!! unknown tokenizer '$requested'" >&2
    exit 2
fi
run_sweep_probe "$resolved" "$@"
