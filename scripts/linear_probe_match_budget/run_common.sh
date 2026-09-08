#!/usr/bin/env bash
set -euo pipefail

MATCH_BUDGET_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MATCH_BUDGET_WORKSPACE="${MATCH_BUDGET_WORKSPACE:-/cache/ma-user/VTBenchLab}"
MATCH_BUDGET_DINO_REPO="${MATCH_BUDGET_DINO_REPO:-$MATCH_BUDGET_WORKSPACE/dinov2}"
MATCH_BUDGET_DATA="${MATCH_BUDGET_DATA:-$MATCH_BUDGET_WORKSPACE/data/imagenet1k}"
MATCH_BUDGET_EXTRA="${MATCH_BUDGET_EXTRA:-$MATCH_BUDGET_DATA/extra}"
MATCH_BUDGET_OUT_ROOT="${MATCH_BUDGET_OUT_ROOT:-$MATCH_BUDGET_WORKSPACE/outputs/vae_linear_probing_match_budget_4shot}"
MATCH_BUDGET_NUM_WORKERS="${MATCH_BUDGET_NUM_WORKERS:-8}"
MATCH_BUDGET_CONDA_ENV="${MATCH_BUDGET_CONDA_ENV:-dino}"
MATCH_BUDGET_SEED="${MATCH_BUDGET_SEED:-0}"
MATCH_BUDGET_SUPPORT_SEED="${MATCH_BUDGET_SUPPORT_SEED:-0}"
MATCH_BUDGET_DRY_RUN="${MATCH_BUDGET_DRY_RUN:-0}"

match_budget_output_name() {
    local model="$1"
    case "$model" in
        pixio_vitb16|pixio_vitl16|pixio_vith16)
            echo "${model}_mae_bn"
            ;;
        webssl_mae3b_full2b_224)
            echo "${model}_cls"
            ;;
        *)
            echo "$model"
            ;;
    esac
}

run_match_budget_probe() {
    local model="$1"
    shift

    local output_name result_path
    output_name="$(match_budget_output_name "$model")"
    result_path="$MATCH_BUDGET_OUT_ROOT/$output_name/results_eval_linear.json"
    if python "$MATCH_BUDGET_SCRIPT_DIR/check_complete.py" "$result_path"; then
        echo ">> skip completed match-budget probe: $model"
        return 0
    fi

    echo ">> ImageNet-1K 4-shot/1-epoch tokenizer probing: $model"
    echo "   support=4/class (4,000 total); optimization batch=1,000; updates=4"
    echo "   Pixio/Web-SSL-MAE head=non-affine BN -> Linear; all others=Linear"
    echo "   support_seed=$MATCH_BUDGET_SUPPORT_SEED; train_seed=$MATCH_BUDGET_SEED"
    echo "   output_root=$MATCH_BUDGET_OUT_ROOT"

    if [[ "$MATCH_BUDGET_DRY_RUN" == "1" ]]; then
        echo ">> dry-run command: conda run -n $MATCH_BUDGET_CONDA_ENV python $MATCH_BUDGET_SCRIPT_DIR/linear_probe.py --model $model"
        return 0
    fi

    [[ -d "$MATCH_BUDGET_DINO_REPO/dinov2" ]] || {
        echo "!! missing DINOv2 repository: $MATCH_BUDGET_DINO_REPO" >&2
        return 1
    }
    [[ -d "$MATCH_BUDGET_DATA" ]] || {
        echo "!! missing ImageNet root: $MATCH_BUDGET_DATA" >&2
        return 1
    }
    [[ -d "$MATCH_BUDGET_EXTRA" ]] || {
        echo "!! missing ImageNet extra directory: $MATCH_BUDGET_EXTRA" >&2
        return 1
    }

    local python_command=(python)
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$MATCH_BUDGET_CONDA_ENV" ]]; then
        python_command=(conda run --no-capture-output -n "$MATCH_BUDGET_CONDA_ENV" python)
    fi

    PYTHONPATH="$MATCH_BUDGET_DINO_REPO${PYTHONPATH:+:$PYTHONPATH}" \
        "${python_command[@]}" "$MATCH_BUDGET_SCRIPT_DIR/linear_probe.py" \
        --model "$model" \
        --data-root "$MATCH_BUDGET_DATA" \
        --extra-root "$MATCH_BUDGET_EXTRA" \
        --output-root "$MATCH_BUDGET_OUT_ROOT" \
        --num-workers "$MATCH_BUDGET_NUM_WORKERS" \
        --seed "$MATCH_BUDGET_SEED" \
        --support-seed "$MATCH_BUDGET_SUPPORT_SEED" \
        --stop-after-epoch 1 \
        "$@"
}
