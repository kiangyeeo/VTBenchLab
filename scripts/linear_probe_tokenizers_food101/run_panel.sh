#!/usr/bin/env bash
set -euo pipefail

FOOD101_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$FOOD101_SCRIPT_DIR/run_common.sh"

FOOD101_PANEL_SHARD_COUNT="${FOOD101_PANEL_SHARD_COUNT:-1}"
FOOD101_PANEL_SHARD_INDEX="${FOOD101_PANEL_SHARD_INDEX:-0}"
FOOD101_PANEL_DRY_RUN="${FOOD101_PANEL_DRY_RUN:-0}"
if [[ ! "$FOOD101_PANEL_SHARD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "!! FOOD101_PANEL_SHARD_COUNT must be a positive integer" >&2
    exit 2
fi
if [[ ! "$FOOD101_PANEL_SHARD_INDEX" =~ ^[0-9]+$ ]] \
    || (( FOOD101_PANEL_SHARD_INDEX >= FOOD101_PANEL_SHARD_COUNT )); then
    echo "!! FOOD101_PANEL_SHARD_INDEX must be in [0, FOOD101_PANEL_SHARD_COUNT)" >&2
    exit 2
fi

echo ">> Running Tokenizer_set_up.md shard $FOOD101_PANEL_SHARD_INDEX/$FOOD101_PANEL_SHARD_COUNT"
row_index=0
while IFS=$'\t' read -r setup_id probe_model; do
    [[ -n "$setup_id" && "$setup_id" != \#* ]] || continue
    if (( row_index % FOOD101_PANEL_SHARD_COUNT == FOOD101_PANEL_SHARD_INDEX )); then
        echo ">> panel[$row_index] setup_id=$setup_id model=$probe_model"
        if [[ "$FOOD101_PANEL_DRY_RUN" != "1" ]]; then
            run_food101_tokenizer_probe "$probe_model" "$@"
        fi
    fi
    ((row_index += 1))
done < "$FOOD101_SCRIPT_DIR/tokenizers_from_setup.tsv"
