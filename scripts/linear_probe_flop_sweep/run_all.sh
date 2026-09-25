#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/../linear_probe_match_budget_5shot_noaug_cached_10epoch/tokenizers.tsv"
SWEEP_GPUS="${SWEEP_GPUS:-0}"

all_ids=()
all_models=()
while IFS=$'\t' read -r rank requested_id model head; do
    [[ -n "$rank" && "$rank" != \#* ]] || continue
    all_ids+=("$requested_id")
    all_models+=("$model")
done < "$MANIFEST"
if [[ ${#all_models[@]} -ne 70 ]]; then
    echo "!! expected 70 tokenizers, found ${#all_models[@]}" >&2
    exit 2
fi

if [[ $# -gt 0 ]]; then
    models=()
    ids=()
    for requested in "$@"; do
        found=""
        for ((i = 0; i < ${#all_models[@]}; i += 1)); do
            if [[ "$requested" == "$((i + 1))" || "$requested" == "${all_ids[$i]}" || "$requested" == "${all_models[$i]}" ]]; then
                models+=("${all_models[$i]}")
                ids+=("${all_ids[$i]}")
                found=1
                break
            fi
        done
        [[ -n "$found" ]] || { echo "!! unknown tokenizer '$requested'" >&2; exit 2; }
    done
else
    models=("${all_models[@]}")
    ids=("${all_ids[@]}")
fi

IFS=',' read -r -a gpu_ids <<< "$SWEEP_GPUS"
for gpu in "${gpu_ids[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "!! SWEEP_GPUS must contain numeric GPU ids" >&2; exit 2; }
done

echo ">> FLOP sweep: tokenizers=${#models[@]} GPUs=${gpu_ids[*]}"
probe_args=()
if [[ -n "${SWEEP_CAP_SHOTS:-}" ]]; then
    probe_args+=(--cap-shots "$SWEEP_CAP_SHOTS")
    echo ">> requested cap shots: $SWEEP_CAP_SHOTS"
fi
if [[ -n "${SWEEP_VALIDATION_SAMPLES:-}" ]]; then
    probe_args+=(--validation-samples "$SWEEP_VALIDATION_SAMPLES")
    echo ">> validation samples: $SWEEP_VALIDATION_SAMPLES"
fi
if [[ -n "${SWEEP_VALIDATION_SEED:-}" ]]; then
    probe_args+=(--validation-seed "$SWEEP_VALIDATION_SEED")
    echo ">> validation seed: $SWEEP_VALIDATION_SEED"
fi
worker_pids=()
terminate_workers() {
    local pid
    for pid in "${worker_pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap 'terminate_workers; exit 130' INT TERM

worker_count=${#gpu_ids[@]}
for ((worker = 0; worker < worker_count; worker += 1)); do
    gpu=${gpu_ids[$worker]}
    (
        for ((i = worker; i < ${#models[@]}; i += worker_count)); do
            echo ">> tokenizer $((i + 1))/${#models[@]} ${ids[$i]} on GPU $gpu"
            CUDA_VISIBLE_DEVICES="$gpu" \
            SWEEP_TOKENIZER_INDEX="$((i + 1))" \
            SWEEP_TOKENIZER_TOTAL="${#models[@]}" \
                bash "$SCRIPT_DIR/run_model.sh" "${models[$i]}" "${probe_args[@]}"
        done
    ) &
    worker_pids+=("$!")
done

failed=0
for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
worker_pids=()
((failed == 0)) || { echo "!! one or more sweep workers failed" >&2; exit 1; }
echo ">> all FLOP sweep tokenizers completed"
