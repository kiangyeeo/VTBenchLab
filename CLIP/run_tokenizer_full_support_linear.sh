#!/usr/bin/env bash
# Run K-shot-aligned ImageNet full-support linear probing.
#
# Usage:
#   bash run_tokenizer_full_support_linear.sh                 # all five models
#   bash run_tokenizer_full_support_linear.sh unitok metaclip # selected models
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/cache/ma-user/VTBenchLab/data/imagenet1k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cache/ma-user/VTBenchLab/outputs/imagenet_kshot_linear_clip_paper_v1}"
CONDA_ENV="${CONDA_ENV:-TokBench}"
SEED="${SEED:-0}"
SELECTION_SEED="${SELECTION_SEED:-0}"
SELECTION_FRACTION="${SELECTION_FRACTION:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_ITER="${MAX_ITER:-1000}"
TOL="${TOL:-1e-4}"
LOGREG_VERBOSE="${LOGREG_VERBOSE:-1}"
FIXED_C="${FIXED_C:-}"
FEATURES_ONLY="${FEATURES_ONLY:-0}"
PROBE_ONLY="${PROBE_ONLY:-0}"
OVERWRITE_PROBE="${OVERWRITE_PROBE:-0}"
OVERWRITE_FEATURES="${OVERWRITE_FEATURES:-0}"

if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=(unitok toklips toklipl vilau metaclip)
fi

EXTRA_ARGS=()
if [[ -n "$FIXED_C" ]]; then
    EXTRA_ARGS+=(--fixed-c "$FIXED_C")
fi
if [[ "$FEATURES_ONLY" == "1" ]]; then
    EXTRA_ARGS+=(--features-only)
fi
if [[ "$PROBE_ONLY" == "1" ]]; then
    EXTRA_ARGS+=(--probe-only)
fi
if [[ "$OVERWRITE_PROBE" == "1" ]]; then
    EXTRA_ARGS+=(--overwrite-probe)
fi
if [[ "$OVERWRITE_FEATURES" == "1" ]]; then
    EXTRA_ARGS+=(--overwrite-features)
fi

for model in "${MODELS[@]}"; do
    echo ">> ImageNet full-support probe: model=$model selection_fraction=$SELECTION_FRACTION batch=$BATCH_SIZE"
    conda run --no-capture-output -n "$CONDA_ENV" \
        python "$SCRIPT_DIR/linear_probe_tokenizers_full_support.py" \
        --model "$model" \
        --data-root "$DATA_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --seed "$SEED" \
        --selection-seed "$SELECTION_SEED" \
        --selection-fraction "$SELECTION_FRACTION" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --max-iter "$MAX_ITER" \
        --tol "$TOL" \
        --logreg-verbose "$LOGREG_VERBOSE" \
        "${EXTRA_ARGS[@]}"
done

if [[ "$FEATURES_ONLY" != "1" ]]; then
    conda run --no-capture-output -n "$CONDA_ENV" \
        python "$SCRIPT_DIR/summarize_tokenizer_full_support.py" \
        --output-root "$OUTPUT_ROOT"
fi
