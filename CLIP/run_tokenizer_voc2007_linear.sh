#!/usr/bin/env bash
# Run the Kornblith-style PASCAL VOC 2007 tokenizer linear probe.
#
# Usage:
#   bash run_tokenizer_voc2007_linear.sh
#   bash run_tokenizer_voc2007_linear.sh unitok toklipl
#
# Environment overrides:
#   DATA_ROOT, OUTPUT_ROOT, CONDA_ENV, BATCH_SIZE, NUM_WORKERS, MAX_ITER, TOL,
#   SEED, LOGREG_VERBOSE, SMOKE_TEST, PROBE_ONLY, OVERWRITE_PROBE,
#   OVERWRITE_FEATURES
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/cache/ma-user/VTBenchLab/data/voc2007/VOCdevkit/VOC2007}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cache/ma-user/VTBenchLab/outputs/voc2007_multilabel_linear_kornblith_v1}"
CONDA_ENV="${CONDA_ENV:-TokBench}"
BATCH_SIZE="${BATCH_SIZE:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_ITER="${MAX_ITER:-1000}"
TOL="${TOL:-1e-4}"
SEED="${SEED:-0}"
LOGREG_VERBOSE="${LOGREG_VERBOSE:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"
PROBE_ONLY="${PROBE_ONLY:-0}"
OVERWRITE_PROBE="${OVERWRITE_PROBE:-0}"
OVERWRITE_FEATURES="${OVERWRITE_FEATURES:-0}"

if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=(unitok vilau metaclip toklips toklipl)
fi

EXTRA_ARGS=()
if [[ "$SMOKE_TEST" == "1" ]]; then
    EXTRA_ARGS+=(--smoke-test)
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
    echo ">> VOC2007 multi-label linear probe: model=$model batch=$BATCH_SIZE max_iter=$MAX_ITER tol=$TOL"
    conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT_DIR/linear_probe_voc2007.py" \
        --model "$model" \
        --data-root "$DATA_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --max-iter "$MAX_ITER" \
        --tol "$TOL" \
        --seed "$SEED" \
        --logreg-verbose "$LOGREG_VERBOSE" \
        "${EXTRA_ARGS[@]}"
done

if [[ "$SMOKE_TEST" != "1" ]]; then
    conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT_DIR/summarize_tokenizer_voc2007.py" \
        --output-root "$OUTPUT_ROOT"
fi
