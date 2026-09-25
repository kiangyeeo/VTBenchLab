#!/usr/bin/env bash
set -euo pipefail

SWEEP_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_WORKSPACE="${SWEEP_WORKSPACE:-/cache/ma-user/VTBenchLab}"
SWEEP_DINO_REPO="${SWEEP_DINO_REPO:-$SWEEP_WORKSPACE/dinov2}"
SWEEP_DATA="${SWEEP_DATA:-$SWEEP_WORKSPACE/data/imagenet1k}"
SWEEP_EXTRA="${SWEEP_EXTRA:-$SWEEP_DATA/extra}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-$SWEEP_WORKSPACE/outputs/vae_linear_probing_flop_sweep_noaug_cached_1epoch_allbn}"
SWEEP_CACHE_ROOT="${SWEEP_CACHE_ROOT:-$SWEEP_WORKSPACE/outputs/vae_linear_probing_flop_sweep_noaug_cached_5epoch_allbn/_feature_cache}"
SWEEP_NUM_WORKERS="${SWEEP_NUM_WORKERS:-8}"
SWEEP_CONDA_ENV="${SWEEP_CONDA_ENV:-dino}"
SWEEP_SEED="${SWEEP_SEED:-0}"
SWEEP_SUPPORT_SEED="${SWEEP_SUPPORT_SEED:-0}"

run_sweep_probe() {
    local model="$1"
    shift
    local python_command=(python)
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$SWEEP_CONDA_ENV" ]]; then
        python_command=(conda run --no-capture-output -n "$SWEEP_CONDA_ENV" python)
    fi
    PYTHONPATH="$SWEEP_DINO_REPO${PYTHONPATH:+:$PYTHONPATH}" \
        "${python_command[@]}" "$SWEEP_SCRIPT_DIR/linear_probe_sweep.py" \
        --model "$model" \
        --data-root "$SWEEP_DATA" \
        --extra-root "$SWEEP_EXTRA" \
        --output-root "$SWEEP_OUTPUT_ROOT" \
        --cache-root "$SWEEP_CACHE_ROOT" \
        --num-workers "$SWEEP_NUM_WORKERS" \
        --seed "$SWEEP_SEED" \
        --support-seed "$SWEEP_SUPPORT_SEED" \
        "$@"
}
