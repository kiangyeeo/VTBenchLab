#!/usr/bin/env bash
set -euo pipefail

FOOD101_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FOOD101_TOTAL_EPOCHS="${FOOD101_TOTAL_EPOCHS:-10}"
FOOD101_PANEL_START_EPOCH="${FOOD101_PANEL_START_EPOCH:-1}"
FOOD101_PANEL_END_EPOCH="${FOOD101_PANEL_END_EPOCH:-$FOOD101_TOTAL_EPOCHS}"
FOOD101_PANEL_GPUS="${FOOD101_PANEL_GPUS:-}"
FOOD101_PANEL_DRY_RUN="${FOOD101_PANEL_DRY_RUN:-0}"

for value_name in FOOD101_TOTAL_EPOCHS FOOD101_PANEL_START_EPOCH FOOD101_PANEL_END_EPOCH; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "!! $value_name must be a positive integer, got '$value'" >&2
        exit 2
    fi
done
if (( FOOD101_PANEL_START_EPOCH > FOOD101_PANEL_END_EPOCH )); then
    echo "!! FOOD101_PANEL_START_EPOCH must not exceed FOOD101_PANEL_END_EPOCH" >&2
    exit 2
fi
if (( FOOD101_PANEL_END_EPOCH > FOOD101_TOTAL_EPOCHS )); then
    echo "!! FOOD101_PANEL_END_EPOCH must not exceed FOOD101_TOTAL_EPOCHS" >&2
    exit 2
fi

gpu_ids=()
if [[ -n "$FOOD101_PANEL_GPUS" ]]; then
    if [[ "$FOOD101_PANEL_GPUS" == ,* \
        || "$FOOD101_PANEL_GPUS" == *, \
        || "$FOOD101_PANEL_GPUS" == *,,* ]]; then
        echo "!! FOOD101_PANEL_GPUS contains an empty GPU id: '$FOOD101_PANEL_GPUS'" >&2
        exit 2
    fi
    IFS=',' read -r -a gpu_ids <<< "$FOOD101_PANEL_GPUS"
    declare -A seen_gpu_ids=()
    for gpu_id in "${gpu_ids[@]}"; do
        if [[ -z "$gpu_id" || ! "$gpu_id" =~ ^[-/[:alnum:]_.:]+$ ]]; then
            echo "!! invalid GPU id in FOOD101_PANEL_GPUS: '$gpu_id'" >&2
            exit 2
        fi
        if [[ -n "${seen_gpu_ids[$gpu_id]:-}" ]]; then
            echo "!! duplicate GPU id in FOOD101_PANEL_GPUS: '$gpu_id'" >&2
            exit 2
        fi
        seen_gpu_ids[$gpu_id]=1
    done
else
    gpu_ids=("")
fi

worker_pids=()
terminate_workers() {
    local pid
    for pid in "${worker_pids[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap 'terminate_workers; exit 130' INT TERM

worker_count="${#gpu_ids[@]}"
echo ">> Food-101 global epoch barrier: epochs $FOOD101_PANEL_START_EPOCH-$FOOD101_PANEL_END_EPOCH/$FOOD101_TOTAL_EPOCHS, workers=$worker_count"

for ((epoch = FOOD101_PANEL_START_EPOCH; epoch <= FOOD101_PANEL_END_EPOCH; epoch += 1)); do
    echo ">> ===== starting global epoch $epoch/$FOOD101_TOTAL_EPOCHS ====="
    worker_pids=()
    for ((worker_index = 0; worker_index < worker_count; worker_index += 1)); do
        gpu_id="${gpu_ids[$worker_index]}"
        if [[ -n "$gpu_id" ]]; then
            CUDA_VISIBLE_DEVICES="$gpu_id" \
            FOOD101_TOTAL_EPOCHS="$FOOD101_TOTAL_EPOCHS" \
            FOOD101_PANEL_SHARD_COUNT="$worker_count" \
            FOOD101_PANEL_SHARD_INDEX="$worker_index" \
            FOOD101_PANEL_DRY_RUN="$FOOD101_PANEL_DRY_RUN" \
                bash "$FOOD101_SCRIPT_DIR/run_panel_epoch.sh" "$epoch" "$@" &
        else
            FOOD101_TOTAL_EPOCHS="$FOOD101_TOTAL_EPOCHS" \
            FOOD101_PANEL_SHARD_COUNT=1 \
            FOOD101_PANEL_SHARD_INDEX=0 \
            FOOD101_PANEL_DRY_RUN="$FOOD101_PANEL_DRY_RUN" \
                bash "$FOOD101_SCRIPT_DIR/run_panel_epoch.sh" "$epoch" "$@" &
        fi
        worker_pids+=("$!")
    done

    round_failed=0
    for worker_index in "${!worker_pids[@]}"; do
        if ! wait "${worker_pids[$worker_index]}"; then
            echo "!! epoch $epoch worker $worker_index failed" >&2
            round_failed=1
        fi
    done
    worker_pids=()
    if (( round_failed != 0 )); then
        echo "!! global epoch $epoch failed; epoch $((epoch + 1)) will not start" >&2
        exit 1
    fi
    echo ">> ===== global epoch $epoch complete for all 45 tokenizers ====="
done

echo ">> requested Food-101 epoch barriers completed"
