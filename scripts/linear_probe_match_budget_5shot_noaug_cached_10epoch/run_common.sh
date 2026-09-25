#!/usr/bin/env bash
set -euo pipefail

CACHED_10E_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CACHED_10E_WORKSPACE="${CACHED_10E_WORKSPACE:-/cache/ma-user/VTBenchLab}"
CACHED_10E_DINO_REPO="${CACHED_10E_DINO_REPO:-$CACHED_10E_WORKSPACE/dinov2}"
CACHED_10E_DATA="${CACHED_10E_DATA:-$CACHED_10E_WORKSPACE/data/imagenet1k}"
CACHED_10E_EXTRA="${CACHED_10E_EXTRA:-$CACHED_10E_DATA/extra}"
CACHED_10E_OUT_ROOT="${CACHED_10E_OUT_ROOT:-$CACHED_10E_WORKSPACE/outputs/vae_linear_probing_match_budget_5shot_noaug_cached_10epoch}"
CACHED_10E_CACHE_ROOT="${CACHED_10E_CACHE_ROOT:-$CACHED_10E_OUT_ROOT/_feature_cache}"
CACHED_10E_NUM_WORKERS="${CACHED_10E_NUM_WORKERS:-8}"
CACHED_10E_CONDA_ENV="${CACHED_10E_CONDA_ENV:-dino}"
CACHED_10E_SEED="${CACHED_10E_SEED:-0}"
CACHED_10E_SUPPORT_SEED="${CACHED_10E_SUPPORT_SEED:-0}"
CACHED_10E_DRY_RUN="${CACHED_10E_DRY_RUN:-0}"

cached_10e_output_name() {
    local model="$1"
    case "$model" in
        pixio_vitb16|pixio_vitl16|pixio_vith16)
            echo "${model}_mae_bn"
            ;;
        webssl_mae*_full2b_224)
            echo "${model}_cls"
            ;;
        vilau)
            echo "vilau_7b_256_semantic_penultimate"
            ;;
        toklip_s)
            echo "toklip_s_semantic_256"
            ;;
        toklip_l)
            echo "toklip_l_semantic_384"
            ;;
        uniar_bsq)
            echo "uniar_bsq_final27_256"
            ;;
        *)
            echo "$model"
            ;;
    esac
}

run_cached_10e_probe() {
    local model="$1"
    shift

    local output_name result_path
    output_name="$(cached_10e_output_name "$model")"
    result_path="$CACHED_10E_OUT_ROOT/$output_name/results_eval_linear.json"
    if python "$CACHED_10E_SCRIPT_DIR/check_complete.py" "$result_path"; then
        echo ">> skip completed cached 10-epoch probe: $model"
        return 0
    fi

    echo ">> ImageNet-1K deterministic 5-shot cached 10-epoch probe: $model"
    echo "   cache train=5,000 and val=50,000 frozen features exactly once"
    echo "   no augmentation; optimization batch=1,000; epochs=10; updates=50"
    echo "   unified head for all 70 models: non-affine BN -> Linear"
    echo "   output_root=$CACHED_10E_OUT_ROOT"

    if [[ "$CACHED_10E_DRY_RUN" == "1" ]]; then
        echo ">> dry-run command: python $CACHED_10E_SCRIPT_DIR/linear_probe.py --model $model"
        return 0
    fi

    [[ -d "$CACHED_10E_DINO_REPO/dinov2" ]] || {
        echo "!! missing DINOv2 repository: $CACHED_10E_DINO_REPO" >&2
        return 1
    }
    [[ -d "$CACHED_10E_DATA" ]] || {
        echo "!! missing ImageNet root: $CACHED_10E_DATA" >&2
        return 1
    }
    [[ -d "$CACHED_10E_EXTRA" ]] || {
        echo "!! missing ImageNet extra directory: $CACHED_10E_EXTRA" >&2
        return 1
    }

    local python_command=(python)
    if [[ "${CONDA_DEFAULT_ENV:-}" != "$CACHED_10E_CONDA_ENV" ]]; then
        python_command=(conda run --no-capture-output -n "$CACHED_10E_CONDA_ENV" python)
    fi

    PYTHONPATH="$CACHED_10E_DINO_REPO${PYTHONPATH:+:$PYTHONPATH}" \
        "${python_command[@]}" "$CACHED_10E_SCRIPT_DIR/linear_probe.py" \
        --model "$model" \
        --data-root "$CACHED_10E_DATA" \
        --extra-root "$CACHED_10E_EXTRA" \
        --output-root "$CACHED_10E_OUT_ROOT" \
        --cache-root "$CACHED_10E_CACHE_ROOT" \
        --num-workers "$CACHED_10E_NUM_WORKERS" \
        --seed "$CACHED_10E_SEED" \
        --support-seed "$CACHED_10E_SUPPORT_SEED" \
        --stop-after-epoch 10 \
        "$@"
}
