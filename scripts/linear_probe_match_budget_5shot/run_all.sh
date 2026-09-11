#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MATCH_BUDGET_5SHOT_GPUS="${MATCH_BUDGET_5SHOT_GPUS:-}"
MATCH_BUDGET_5SHOT_DRY_RUN="${MATCH_BUDGET_5SHOT_DRY_RUN:-0}"

all_labels=()
all_models=()
while IFS=$'\t' read -r rank label model head; do
    [[ -n "$rank" && "$rank" != \#* ]] || continue
    all_labels+=("$label")
    all_models+=("$model")
done < "$SCRIPT_DIR/tokenizers.tsv"

if [[ ${#all_models[@]} -ne 64 ]]; then
    echo "!! expected 64 unique models in tokenizers.tsv, found ${#all_models[@]}" >&2
    exit 2
fi

if [[ $# -gt 0 ]]; then
    models=()
    for requested in "$@"; do
        resolved=""
        for ((index = 0; index < ${#all_models[@]}; index += 1)); do
            if [[ "$requested" == "${all_models[$index]}" || \
                  "$requested" == "${all_labels[$index]}" || \
                  "$requested" == "$((index + 1))" ]]; then
                resolved="${all_models[$index]}"
                break
            fi
        done
        if [[ -z "$resolved" ]]; then
            echo "!! unknown 5-shot tokenizer '$requested'" >&2
            exit 2
        fi
        models+=("$resolved")
    done
else
    models=("${all_models[@]}")
fi

gpu_ids=()
if [[ -n "$MATCH_BUDGET_5SHOT_GPUS" ]]; then
    IFS=',' read -r -a gpu_ids <<< "$MATCH_BUDGET_5SHOT_GPUS"
    for gpu_id in "${gpu_ids[@]}"; do
        if [[ -z "$gpu_id" || ! "$gpu_id" =~ ^[0-9]+$ ]]; then
            echo "!! MATCH_BUDGET_5SHOT_GPUS must be comma-separated numeric ids" >&2
            exit 2
        fi
    done
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
        IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
    else
        gpu_ids=("inherit")
    fi
else
    gpu_ids=("0")
fi

echo ">> 5-shot match-budget panel: models=${#models[@]}, workers=${#gpu_ids[@]}, GPUs=${MATCH_BUDGET_5SHOT_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
echo ">> protocol: ImageNet-1K class-balanced 5-shot, exactly one epoch"

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
for ((worker_index = 0; worker_index < worker_count; worker_index += 1)); do
    gpu_id="${gpu_ids[$worker_index]}"
    (
        for ((model_index = worker_index; model_index < ${#models[@]}; model_index += worker_count)); do
            model="${models[$model_index]}"
            if [[ "$gpu_id" == "inherit" ]]; then
                MATCH_BUDGET_5SHOT_DRY_RUN="$MATCH_BUDGET_5SHOT_DRY_RUN" \
                    bash "$SCRIPT_DIR/run_model.sh" "$model"
            else
                CUDA_VISIBLE_DEVICES="$gpu_id" \
                MATCH_BUDGET_5SHOT_DRY_RUN="$MATCH_BUDGET_5SHOT_DRY_RUN" \
                    bash "$SCRIPT_DIR/run_model.sh" "$model"
            fi
        done
    ) &
    worker_pids+=("$!")
done

failed=0
for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
worker_pids=()

if ((failed)); then
    echo "!! one or more 5-shot workers failed" >&2
    exit 1
fi
echo ">> all requested 5-shot probes completed"
