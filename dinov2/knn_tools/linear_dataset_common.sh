#!/usr/bin/env bash

_has_arg() {
    local needle="$1"
    shift
    for arg in "$@"; do
        if [[ "$arg" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

setup_linear_training_args() {
    BATCH_SIZE="${BATCH_SIZE:-128}"
    EPOCHS="${EPOCHS:-10}"
    EPOCH_LENGTH="${EPOCH_LENGTH:-1250}"

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
