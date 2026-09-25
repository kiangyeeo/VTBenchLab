#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CACHED_10E_GPUS="${CACHED_10E_GPUS:-}"
CACHED_10E_DRY_RUN="${CACHED_10E_DRY_RUN:-0}"

all_labels=()
all_models=()
while IFS=$'\t' read -r rank label model head; do
    [[ -n "$rank" && "$rank" != \#* ]] || continue
    all_labels+=("$label")
    all_models+=("$model")
done < "$SCRIPT_DIR/tokenizers.tsv"

if [[ ${#all_models[@]} -ne 70 ]]; then
    echo "!! expected 70 unique models, found ${#all_models[@]}" >&2
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
            echo "!! unknown tokenizer '$requested'" >&2
            exit 2
        fi
        models+=("$resolved")
    done
else
    models=("${all_models[@]}")
fi

gpu_ids=()
if [[ -n "$CACHED_10E_GPUS" ]]; then
    IFS=',' read -r -a gpu_ids <<< "$CACHED_10E_GPUS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
        IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
    else
        gpu_ids=("inherit")
    fi
else
    gpu_ids=("0")
fi
for gpu_id in "${gpu_ids[@]}"; do
    if [[ "$gpu_id" != "inherit" && ! "$gpu_id" =~ ^[0-9]+$ ]]; then
        echo "!! CACHED_10E_GPUS must be comma-separated numeric ids" >&2
        exit 2
    fi
done

echo ">> no-augmentation cached 5-shot/10-epoch panel: models=${#models[@]}, workers=${#gpu_ids[@]}"

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
                CACHED_10E_DRY_RUN="$CACHED_10E_DRY_RUN" bash "$SCRIPT_DIR/run_model.sh" "$model"
            else
                CUDA_VISIBLE_DEVICES="$gpu_id" CACHED_10E_DRY_RUN="$CACHED_10E_DRY_RUN" \
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
    echo "!! one or more cached 10-epoch workers failed" >&2
    exit 1
fi
echo ">> all requested cached 10-epoch probes completed"
