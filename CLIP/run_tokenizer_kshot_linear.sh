#!/usr/bin/env bash
# Run the CLIP-paper-aligned ImageNet k-shot tokenizer linear probe.
#
# Usage:
#   bash run_tokenizer_kshot_linear.sh                 # all five models
#   bash run_tokenizer_kshot_linear.sh unitok metaclip # selected models
#
# Environment overrides:
#   DATA_ROOT, OUTPUT_ROOT, CONDA_ENV, PROTOCOL, SEEDS, SHOTS, BATCH_SIZE,
#   NUM_WORKERS, SELECTION_SEED, SELECTION_FRACTION, MAX_ITER, TOL,
#   LOGREG_VERBOSE, PROBE_ONLY, OVERWRITE_PROBE, OVERWRITE_FEATURES
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/cache/ma-user/VTBenchLab/data/imagenet1k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cache/ma-user/VTBenchLab/outputs/imagenet_kshot_linear_clip_paper_v1}"
CONDA_ENV="${CONDA_ENV:-TokBench}"
PROTOCOL="${PROTOCOL:-clip-paper-v1}"
SEEDS="${SEEDS:-0 1 2}"
SHOTS="${SHOTS:-1 2 4 8 16}"
BATCH_SIZE="${BATCH_SIZE:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SELECTION_SEED="${SELECTION_SEED:-0}"
SELECTION_FRACTION="${SELECTION_FRACTION:-0.1}"
C="${C:-0.316}"
MAX_ITER="${MAX_ITER:-1000}"
TOL="${TOL:-1e-4}"
LOGREG_VERBOSE="${LOGREG_VERBOSE:-1}"
PROBE_ONLY="${PROBE_ONLY:-0}"
OVERWRITE_PROBE="${OVERWRITE_PROBE:-0}"
OVERWRITE_FEATURES="${OVERWRITE_FEATURES:-0}"

read -r -a SEED_VALUES <<< "$SEEDS"
read -r -a SHOT_VALUES <<< "$SHOTS"

if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=(unitok toklips toklipl vilau metaclip)
fi

EXTRA_ARGS=()
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
    for seed in "${SEED_VALUES[@]}"; do
        echo ">> ImageNet k-shot linear probe: protocol=$PROTOCOL model=$model seed=$seed shots=$SHOTS batch=$BATCH_SIZE"
        conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT_DIR/linear_probe_tokenizers.py" \
            --model "$model" \
            --data-root "$DATA_ROOT" \
            --output-root "$OUTPUT_ROOT" \
            --protocol "$PROTOCOL" \
            --seed "$seed" \
            --selection-seed "$SELECTION_SEED" \
            --selection-fraction "$SELECTION_FRACTION" \
            --shots "${SHOT_VALUES[@]}" \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --c "$C" \
            --max-iter "$MAX_ITER" \
            --tol "$TOL" \
            --logreg-verbose "$LOGREG_VERBOSE" \
            "${EXTRA_ARGS[@]}"
    done
done

conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT_DIR/summarize_tokenizer_kshot.py" \
    --output-root "$OUTPUT_ROOT" \
    --seeds "${SEED_VALUES[@]}"
