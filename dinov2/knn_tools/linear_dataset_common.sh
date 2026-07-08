#!/usr/bin/env bash

_has_arg() {
    local needle="$1"
    shift
    for arg in "$@"; do
        if [[ "$arg" == "$needle" || "$arg" == "$needle="* ]]; then
            return 0
        fi
    done
    return 1
}

_arg_value() {
    local needle="$1"
    shift

    while [[ $# -gt 0 ]]; do
        case "$1" in
            "$needle")
                shift
                if [[ $# -gt 0 ]]; then
                    printf "%s\n" "$1"
                    return 0
                fi
                return 1
                ;;
            "$needle="*)
                printf "%s\n" "${1#"$needle="}"
                return 0
                ;;
        esac
        shift
    done
    return 1
}

setup_linear_training_args() {
    BATCH_SIZE="$(_arg_value "--batch-size" "$@" || printf "%s" "${BATCH_SIZE:-128}")"

    case "${DATASET,,}" in
        cifar100)
            DEFAULT_TRAIN_SAMPLES=50000
            DEFAULT_EPOCHS=20
            DEFAULT_EVAL_PERIOD_ITERATIONS=0
            ;;
        food101)
            DEFAULT_TRAIN_SAMPLES=75750
            DEFAULT_EPOCHS=20
            DEFAULT_EVAL_PERIOD_ITERATIONS=0
            ;;
        oxford_pets)
            DEFAULT_TRAIN_SAMPLES=3680
            DEFAULT_EPOCHS=100
            DEFAULT_EVAL_PERIOD_ITERATIONS=0
            ;;
        flowers102)
            DEFAULT_TRAIN_SAMPLES=1020
            DEFAULT_EPOCHS=200
            DEFAULT_EVAL_PERIOD_ITERATIONS=0
            ;;
        stanford_cars)
            DEFAULT_TRAIN_SAMPLES=8144
            DEFAULT_EPOCHS=100
            DEFAULT_EVAL_PERIOD_ITERATIONS=0
            ;;
        fgvc_aircraft)
            DEFAULT_TRAIN_SAMPLES=3334
            DEFAULT_EPOCHS=100
            DEFAULT_EVAL_PERIOD_ITERATIONS=0
            ;;
        dtd)
            DEFAULT_TRAIN_SAMPLES=1880
            DEFAULT_EPOCHS=200
            DEFAULT_EVAL_PERIOD_ITERATIONS=0
            ;;
        sun397)
            DEFAULT_TRAIN_SAMPLES=76127
            DEFAULT_EPOCHS=15
            DEFAULT_EVAL_PERIOD_ITERATIONS=0
            ;;
        caltech101)
            DEFAULT_TRAIN_SAMPLES=3030
            DEFAULT_EPOCHS=100
            DEFAULT_EVAL_PERIOD_ITERATIONS=0
            ;;
        *)
            DEFAULT_TRAIN_SAMPLES=0
            DEFAULT_EPOCHS=10
            DEFAULT_EPOCH_LENGTH=1250
            DEFAULT_EVAL_PERIOD_ITERATIONS=1250
            ;;
    esac

    if [[ "${DEFAULT_TRAIN_SAMPLES:-0}" -gt 0 ]]; then
        DEFAULT_EPOCH_LENGTH=$(((DEFAULT_TRAIN_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE))
    fi

    EPOCHS="$(_arg_value "--epochs" "$@" || printf "%s" "${EPOCHS:-$DEFAULT_EPOCHS}")"
    EPOCH_LENGTH="$(_arg_value "--epoch-length" "$@" || printf "%s" "${EPOCH_LENGTH:-$DEFAULT_EPOCH_LENGTH}")"
    EVAL_PERIOD_ITERATIONS="$(_arg_value "--eval-period-iterations" "$@" || printf "%s" "${EVAL_PERIOD_ITERATIONS:-$DEFAULT_EVAL_PERIOD_ITERATIONS}")"

    TRAINING_ARGS=()
    if ! _has_arg "--batch-size" "$@"; then
        TRAINING_ARGS+=(--batch-size "$BATCH_SIZE")
    fi
    if ! _has_arg "--epochs" "$@"; then
        TRAINING_ARGS+=(--epochs "$EPOCHS")
    fi
    if ! _has_arg "--epoch-length" "$@"; then
        TRAINING_ARGS+=(--epoch-length "$EPOCH_LENGTH")
    fi
    if ! _has_arg "--eval-period-iterations" "$@"; then
        TRAINING_ARGS+=(--eval-period-iterations "$EVAL_PERIOD_ITERATIONS")
    fi
}

setup_linear_dataset_args() {
    local tokenizer_dir="$1"
    local legacy_outdir="$2"
    shift 2

    DATASET="${DATASET:-imagenet1k}"
    OUT_ROOT="${OUT_ROOT:-/cache/ma-user/VTBenchLab/outputs/vae_linear_probing}"

    TEST_DATASET_ARGS=()
    if [[ "$DATASET" == "imagenet1k" || "$DATASET" == "ImageNet" ]]; then
        DATA="${DATA:-/cache/ma-user/VTBenchLab/data/imagenet1k}"
        EXTRA="${EXTRA:-$DATA/extra}"
        [ -d "$DATA" ] || { echo "!! missing ImageNet root: $DATA"; exit 1; }
        [ -d "$EXTRA" ] || { echo "!! missing ImageNet extra dir: $EXTRA"; exit 1; }

        TRAIN_DATASET="ImageNet:split=TRAIN:root=$DATA:extra=$EXTRA"
        VAL_DATASET="ImageNet:split=VAL:root=$DATA:extra=$EXTRA"
        OUTDIR="${OUTDIR:-$legacy_outdir}"
    else
        HF_DATA_ROOT="${HF_DATA_ROOT:-/cache/ma-user/VTBenchLab/data/hf_datasets}"
        [ -d "$HF_DATA_ROOT/$DATASET" ] || { echo "!! missing HF dataset: $HF_DATA_ROOT/$DATASET"; exit 1; }

        TRAIN_DATASET="HFDataset:name=$DATASET:split=TRAIN:root=$HF_DATA_ROOT"
        VAL_DATASET="HFDataset:name=$DATASET:split=VAL:root=$HF_DATA_ROOT"
        OUTDIR="${OUTDIR:-$OUT_ROOT/$DATASET/$tokenizer_dir}"

        if ! _has_arg "--test-datasets" "$@"; then
            case "$DATASET" in
                flowers102|sun397)
                    TEST_DATASET_ARGS=(--test-datasets "HFDataset:name=$DATASET:split=TEST:root=$HF_DATA_ROOT")
                    ;;
            esac
        fi
    fi
}
