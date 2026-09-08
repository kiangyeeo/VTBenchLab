#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-/cache/ma-user/VTBenchLab}"
COCO_OUT_ROOT="${OUT_ROOT:-$WORKSPACE/outputs/coco2014_multilabel_linear_probing}"
COCO_GPUS="${COCO_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
COCO_MODELS="${COCO_MODELS:-}"
COCO_DRY_RUN="${COCO_DRY_RUN:-0}"
COCO_SKIP_COMPLETED="${COCO_SKIP_COMPLETED:-1}"

if [[ -z "$COCO_GPUS" ]]; then
    echo "!! set COCO_GPUS to a comma-separated GPU list, for example COCO_GPUS=0,1,2,3" >&2
    exit 2
fi
for value_name in COCO_DRY_RUN COCO_SKIP_COMPLETED; do
    value="${!value_name}"
    if [[ "$value" != "0" && "$value" != "1" ]]; then
        echo "!! $value_name must be 0 or 1, got '$value'" >&2
        exit 2
    fi
done
for argument in "$@"; do
    case "$argument" in
        -h|--help|--model|--model=*|--epochs|--epochs=*|--stop-after-epoch|--stop-after-epoch=*|--output-dir|--output-dir=*|--output-root|--output-root=*|--no-resume)
            echo "!! the all-model runner owns $argument; it cannot be passed through" >&2
            exit 2
            ;;
    esac
done

if [[ "$COCO_GPUS" == ,* || "$COCO_GPUS" == *, || "$COCO_GPUS" == *,,* ]]; then
    echo "!! COCO_GPUS contains an empty GPU id: '$COCO_GPUS'" >&2
    exit 2
fi
IFS=',' read -r -a gpu_ids <<< "$COCO_GPUS"
declare -A seen_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
    if [[ -z "$gpu_id" || ! "$gpu_id" =~ ^[-/[:alnum:]_.:]+$ ]]; then
        echo "!! invalid GPU id in COCO_GPUS: '$gpu_id'" >&2
        exit 2
    fi
    if [[ -n "${seen_gpu_ids[$gpu_id]:-}" ]]; then
        echo "!! duplicate GPU id in COCO_GPUS: '$gpu_id'" >&2
        exit 2
    fi
    seen_gpu_ids[$gpu_id]=1
done

models=()
if [[ -n "$COCO_MODELS" ]]; then
    if [[ "$COCO_MODELS" == ,* || "$COCO_MODELS" == *, || "$COCO_MODELS" == *,,* ]]; then
        echo "!! COCO_MODELS contains an empty model name" >&2
        exit 2
    fi
    IFS=',' read -r -a models <<< "$COCO_MODELS"
else
    while IFS= read -r runner; do
        model="$(basename "$runner")"
        model="${model#run_}"
        model="${model%.sh}"
        case "$model" in
            common|coco_multilabel|coco_multilabel_all) continue ;;
        esac
        models+=("$model")
    done < <(find "$SCRIPT_DIR" -maxdepth 1 -type f -name 'run_*.sh' -print | sort)
fi
if (( ${#models[@]} == 0 )); then
    echo "!! no tokenizer runners found" >&2
    exit 1
fi
for model in "${models[@]}"; do
    if [[ ! -f "$SCRIPT_DIR/run_${model}.sh" ]]; then
        echo "!! no existing tokenizer runner for '$model': $SCRIPT_DIR/run_${model}.sh" >&2
        exit 2
    fi
done

mkdir -p "$COCO_OUT_ROOT"
status_dir="$COCO_OUT_ROOT/_all_models_status"
mkdir -p "$status_dir"

declare -A completed_models=()
if [[ "$COCO_SKIP_COMPLETED" == "1" ]]; then
    while IFS= read -r model; do
        [[ -n "$model" ]] && completed_models[$model]=1
    done < <(python - "$COCO_OUT_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
for protocol_path in root.glob("*/protocol.json"):
    result_path = protocol_path.parent / "results_eval_multilabel.json"
    if not result_path.is_file():
        continue
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if (
        protocol.get("version") == "tokenizer_linear_probe_coco2014_multilabel_v1"
        and protocol.get("epochs") == 1
        and result.get("protocol_version") == protocol.get("version")
        and result.get("iteration") == protocol.get("max_updates")
        and result.get("best_classifier", {}).get("mAP") is not None
    ):
        print(protocol["model"])
PY
    )
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
echo ">> COCO multi-label all-model panel: ${#models[@]} tokenizers, $worker_count GPUs"
echo "   GPUs=$COCO_GPUS; epochs=1; skip_completed=$COCO_SKIP_COMPLETED; dry_run=$COCO_DRY_RUN"
echo "   output_root=$COCO_OUT_ROOT"

for ((worker_index = 0; worker_index < worker_count; worker_index += 1)); do
    (
        gpu_id="${gpu_ids[$worker_index]}"
        failure_file="$status_dir/failures_worker_${worker_index}.txt"
        success_file="$status_dir/success_worker_${worker_index}.txt"
        : > "$failure_file"
        : > "$success_file"
        selected_count=0
        failed_count=0
        skipped_count=0
        for model_index in "${!models[@]}"; do
            if (( model_index % worker_count != worker_index )); then
                continue
            fi
            model="${models[$model_index]}"
            if [[ -n "${completed_models[$model]:-}" ]]; then
                echo ">> [GPU $gpu_id] skip completed $model"
                ((skipped_count += 1))
                continue
            fi
            ((selected_count += 1))
            model_log="$status_dir/${model}.launcher.log"
            echo ">> [GPU $gpu_id] start $model ($((model_index + 1))/${#models[@]})"
            if [[ "$COCO_DRY_RUN" == "1" ]]; then
                echo "CUDA_VISIBLE_DEVICES=$gpu_id bash $SCRIPT_DIR/run_coco_multilabel.sh $model --epochs 1 --stop-after-epoch 1 $*"
                continue
            fi
            if CUDA_VISIBLE_DEVICES="$gpu_id" OUT_ROOT="$COCO_OUT_ROOT" \
                bash "$SCRIPT_DIR/run_coco_multilabel.sh" "$model" \
                --epochs 1 --stop-after-epoch 1 "$@" > "$model_log" 2>&1; then
                echo "$model" >> "$success_file"
                echo ">> [GPU $gpu_id] done $model"
            else
                status=$?
                echo -e "$model\texit=$status\tlog=$model_log" >> "$failure_file"
                echo "!! [GPU $gpu_id] failed $model (exit $status); continuing; log=$model_log" >&2
                ((failed_count += 1))
            fi
        done
        echo ">> [GPU $gpu_id] worker complete: attempted=$selected_count skipped=$skipped_count failed=$failed_count"
        ((failed_count == 0))
    ) &
    worker_pids+=("$!")
done

panel_failed=0
for worker_index in "${!worker_pids[@]}"; do
    if ! wait "${worker_pids[$worker_index]}"; then
        panel_failed=1
    fi
done
worker_pids=()

if (( panel_failed != 0 )); then
    echo "!! panel completed with failures; inspect $status_dir/failures_worker_*.txt" >&2
    exit 1
fi
echo ">> all requested COCO multi-label tokenizer probes are complete"
