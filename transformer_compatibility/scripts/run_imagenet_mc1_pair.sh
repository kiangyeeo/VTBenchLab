#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(cd -- "$EXPERIMENT_ROOT/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-dino}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/data/imagenet1k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EXPERIMENT_ROOT/outputs/imagenet1k}"
MODELS="${MODELS:-mc1_b16_224_2.5b mc1_b16_224_400m}"
READOUTS="${READOUTS:-gap_linear gap_mlp transformer}"
SEEDS="${SEEDS:-0 1 2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bfloat16}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    PYTHON_CMD=(python)
else
    PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV" python)
fi

for model in $MODELS; do
    for readout in $READOUTS; do
        for seed in $SEEDS; do
            run_dir="$OUTPUT_ROOT/$model/$readout/seed$seed"
            if [[ -f "$run_dir/summary.json" ]]; then
                echo ">> skip completed run: $model / $readout / seed$seed"
                continue
            fi
            echo ">> run: $model / $readout / seed$seed"
            "${PYTHON_CMD[@]}" "$EXPERIMENT_ROOT/train_imagenet.py" \
                --model "$model" \
                --readout "$readout" \
                --seed "$seed" \
                --data-root "$DATA_ROOT" \
                --output-root "$OUTPUT_ROOT" \
                --micro-batch-size "$MICRO_BATCH_SIZE" \
                --eval-batch-size "$EVAL_BATCH_SIZE" \
                --num-workers "$NUM_WORKERS" \
                --precision "$PRECISION"
        done
    done
done

"${PYTHON_CMD[@]}" "$EXPERIMENT_ROOT/summarize_imagenet.py" \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "${OUTPUT_ROOT}_summary"

