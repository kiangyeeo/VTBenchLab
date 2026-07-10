#!/usr/bin/env bash
# Run the OpenAI-CLIP-style ImageNet k-shot linear probe for VTBench tokenizers.
#
# Usage:
#   bash run_tokenizer_kshot_linear.sh                 # all five models
#   bash run_tokenizer_kshot_linear.sh unitok metaclip # selected models
#
# Environment overrides:
#   DATA_ROOT, OUTPUT_ROOT, CONDA_ENV, SEED, SHOTS, NUM_WORKERS, C, MAX_ITER
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/cache/ma-user/VTBenchLab/data/imagenet1k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cache/ma-user/VTBenchLab/outputs/imagenet_kshot_linear_clip}"
CONDA_ENV="${CONDA_ENV:-TokBench}"
SEED="${SEED:-0}"
SHOTS="${SHOTS:-1 2 4 8 16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
C="${C:-0.316}"
MAX_ITER="${MAX_ITER:-1000}"

if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=(unitok toklips toklipl vilau metaclip)
fi

batch_size_for() {
    case "$1" in
        unitok) printf '%s\n' "${UNITOK_BATCH_SIZE:-64}" ;;
        toklips) printf '%s\n' "${TOKLIPS_BATCH_SIZE:-64}" ;;
        toklipl) printf '%s\n' "${TOKLIPL_BATCH_SIZE:-32}" ;;
        vilau) printf '%s\n' "${VILAU_BATCH_SIZE:-64}" ;;
        metaclip) printf '%s\n' "${METACLIP_BATCH_SIZE:-256}" ;;
        *) echo "Unknown model: $1" >&2; exit 2 ;;
    esac
}

for model in "${MODELS[@]}"; do
    batch_size="$(batch_size_for "$model")"
    echo ">> CLIP-style ImageNet k-shot linear probe: model=$model seed=$SEED shots=$SHOTS batch=$batch_size"
    conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT_DIR/linear_probe_tokenizers.py" \
        --model "$model" \
        --data-root "$DATA_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --seed "$SEED" \
        --shots $SHOTS \
        --batch-size "$batch_size" \
        --num-workers "$NUM_WORKERS" \
        --c "$C" \
        --max-iter "$MAX_ITER"
done

conda run --no-capture-output -n "$CONDA_ENV" python "$SCRIPT_DIR/summarize_tokenizer_kshot.py" \
    --output-root "$OUTPUT_ROOT" \
    --seed "$SEED"
