#!/usr/bin/env bash
set -euo pipefail

FOOD101_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$FOOD101_SCRIPT_DIR/run_common.sh"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 TARGET_EPOCH [linear_probe.py arguments...]" >&2
    exit 2
fi

target_epoch="$1"
shift
FOOD101_TOTAL_EPOCHS="${FOOD101_TOTAL_EPOCHS:-10}"
FOOD101_PANEL_SHARD_COUNT="${FOOD101_PANEL_SHARD_COUNT:-1}"
FOOD101_PANEL_SHARD_INDEX="${FOOD101_PANEL_SHARD_INDEX:-0}"
FOOD101_PANEL_DRY_RUN="${FOOD101_PANEL_DRY_RUN:-0}"
FOOD101_SEED="${FOOD101_SEED:-0}"

for value_name in target_epoch FOOD101_TOTAL_EPOCHS FOOD101_PANEL_SHARD_COUNT; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "!! $value_name must be a positive integer, got '$value'" >&2
        exit 2
    fi
done
if [[ ! "$FOOD101_PANEL_SHARD_INDEX" =~ ^[0-9]+$ ]] \
    || (( FOOD101_PANEL_SHARD_INDEX >= FOOD101_PANEL_SHARD_COUNT )); then
    echo "!! FOOD101_PANEL_SHARD_INDEX must be in [0, FOOD101_PANEL_SHARD_COUNT)" >&2
    exit 2
fi
if (( target_epoch > FOOD101_TOTAL_EPOCHS )); then
    echo "!! target epoch $target_epoch exceeds total epochs $FOOD101_TOTAL_EPOCHS" >&2
    exit 2
fi
if [[ "$FOOD101_PANEL_DRY_RUN" != "0" && "$FOOD101_PANEL_DRY_RUN" != "1" ]]; then
    echo "!! FOOD101_PANEL_DRY_RUN must be 0 or 1" >&2
    exit 2
fi
if [[ ! "$FOOD101_SEED" =~ ^[0-9]+$ ]]; then
    echo "!! FOOD101_SEED must be a non-negative integer" >&2
    exit 2
fi

for argument in "$@"; do
    case "$argument" in
        -h|--help|--model|--model=*|--epochs|--epochs=*|--stop-after-epoch|--stop-after-epoch=*|--seed|--seed=*|--no-resume|--output-dir|--output-dir=*|--output-root|--output-root=*)
            echo "!! panel scheduling owns $argument; it cannot be passed through" >&2
            exit 2
            ;;
    esac
done

echo ">> epoch $target_epoch/$FOOD101_TOTAL_EPOCHS shard $FOOD101_PANEL_SHARD_INDEX/$FOOD101_PANEL_SHARD_COUNT"
row_index=0
selected_count=0
while IFS=$'\t' read -r setup_id probe_model; do
    [[ -n "$setup_id" && "$setup_id" != \#* ]] || continue
    if (( row_index % FOOD101_PANEL_SHARD_COUNT == FOOD101_PANEL_SHARD_INDEX )); then
        echo ">> epoch=$target_epoch panel[$row_index] setup_id=$setup_id model=$probe_model"
        ((selected_count += 1))
        if [[ "$FOOD101_PANEL_DRY_RUN" != "1" ]]; then
            run_food101_tokenizer_probe \
                "$probe_model" \
                --epochs "$FOOD101_TOTAL_EPOCHS" \
                --stop-after-epoch "$target_epoch" \
                --seed "$FOOD101_SEED" \
                "$@"
        fi
    fi
    ((row_index += 1))
done < "$FOOD101_SCRIPT_DIR/tokenizers_from_setup.tsv"

if (( row_index != 45 )); then
    echo "!! expected 45 manifest rows, found $row_index" >&2
    exit 1
fi
echo ">> epoch $target_epoch shard $FOOD101_PANEL_SHARD_INDEX complete ($selected_count tokenizers)"
