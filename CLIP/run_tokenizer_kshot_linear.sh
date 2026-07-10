#!/usr/bin/env bash
# Run the OpenAI-CLIP-style ImageNet k-shot linear probe for VTBench tokenizers.
#
# Usage:
#   bash run_tokenizer_kshot_linear.sh                 # all five models
#   bash run_tokenizer_kshot_linear.sh unitok metaclip # selected models
#
# Environment overrides:
#   DATA_ROOT, OUTPUT_ROOT, CONDA_ENV, SEED, SHOTS, BATCH_SIZE,
#   NUM_WORKERS, C, MAX_ITER, TOL
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/cache/ma-user/VTBenchLab/data/imagenet1k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cache/ma-user/VTBenchLab/outputs/imagenet_kshot_linear_clip}"
CONDA_ENV="${CONDA_ENV:-TokBench}"
SEED="${SEED:-0}"
SHOTS="${SHOTS:-1 2 4 8 16}"
BATCH_SIZE="${BATCH_SIZE:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
C="${C:-0.316}"
MAX_ITER="${MAX_ITER:-1000}"
TOL="${TOL:-1e-4}"

if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=(unitok toklips toklipl vilau metaclip)
fi

for model in "${MODELS[@]}"; do
    echo ">> CLIP-style ImageNet k-shot linear probe: model=$model seed=$SEED shots=$SHOTS batch=$BATCH_SIZE"
    conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT_DIR/linear_probe_tokenizers.py" \
        --model "$model" \
        --data-root "$DATA_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --seed "$SEED" \
        --shots $SHOTS \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --c "$C" \
        --max-iter "$MAX_ITER" \
        --tol "$TOL"
done

conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT_DIR/summarize_tokenizer_kshot.py" \
    --output-root "$OUTPUT_ROOT" \
    --seed "$SEED"
