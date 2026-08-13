#!/usr/bin/env bash
set -euo pipefail

FOOD101_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$FOOD101_SCRIPT_DIR/run_common.sh"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 TOKENIZER_SETUP_ID_OR_MODEL [linear_probe.py arguments...]" >&2
    exit 2
fi

requested_model="$1"
shift
resolved_model=""
resolved_setup_id=""
while IFS=$'\t' read -r setup_id probe_model; do
    [[ -n "$setup_id" && "$setup_id" != \#* ]] || continue
    if [[ "$requested_model" == "$setup_id" || "$requested_model" == "$probe_model" ]]; then
        resolved_model="$probe_model"
        resolved_setup_id="$setup_id"
        break
    fi
done < "$FOOD101_SCRIPT_DIR/tokenizers_from_setup.tsv"

if [[ -z "$resolved_model" ]]; then
    echo "!! '$requested_model' is not one of the 45 Tokenizer_set_up.md configurations" >&2
    exit 2
fi

if [[ "$resolved_setup_id" != "$resolved_model" ]]; then
    echo ">> Tokenizer_set_up.md id $resolved_setup_id maps to probe model $resolved_model"
fi
run_food101_tokenizer_probe "$resolved_model" "$@"
