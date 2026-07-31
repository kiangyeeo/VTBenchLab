#!/bin/bash

# Shared TokBench launcher for the three released RAEv2 Stage-1 variants.
# This file defines run_rae_stage1; use raev2.sh, dinov3.sh, or ijepa.sh.

run_rae_stage1() {
    local script_dir="${SCRIPT_DIR:?SCRIPT_DIR is required}"
    local model_name="${MODEL_NAME:?MODEL_NAME is required}"
    local python_entry="${PYTHON_ENTRY:?PYTHON_ENTRY is required}"
    local encoder_rel="${ENCODER_CKPT_REL:?ENCODER_CKPT_REL is required}"
    local decoder_rel="${DECODER_CKPT_REL:?DECODER_CKPT_REL is required}"
    local stats_rel="${STATS_CKPT_REL:?STATS_CKPT_REL is required}"
    local needs_dinov3="${NEEDS_DINOV3:?NEEDS_DINOV3 is required}"
    local output_name="${OUTPUT_NAME:-$model_name}"
    local -a python_extra_args=()
    if declare -p PYTHON_EXTRA_ARGS >/dev/null 2>&1; then
        python_extra_args=("${PYTHON_EXTRA_ARGS[@]}")
    fi

    local repo_root
    repo_root="$(cd "$script_dir/../.." && pwd)"

    local data_root="${DATA_ROOT:-$repo_root/tokbench_data}"
    local recon_root="${RECON_ROOT:-$repo_root/image_reconstruction_results}"
    local model_zoo="${MODEL_ZOO:-$repo_root/tokenizer_modelzoo}"
    local model_root="${RAEV2_MODEL_ROOT:-$model_zoo/RAEv2-models}"
    local raev2_path="${RAEV2_PATH:-$script_dir/RAEv2}"
    local dinov3_path="${DINOV3_PATH:-$script_dir/dinov3}"
    local batch_size="${BATCH_SIZE:-1}"

    local -a padding_sizes
    local -a text_datas
    local -a face_datas
    local -a gpu_list
    read -r -a padding_sizes <<< "${PADDING_SIZES:-256}"
    read -r -a text_datas <<< "${TEXT_DATAS:-ic13 ic15 textocr tt cord docvqa infograph sroie}"
    read -r -a face_datas <<< "${FACE_DATAS:-wflw}"

    local gpu_csv="${CUDA_VISIBLE_DEVICES:-0}"
    IFS=',' read -r -a gpu_list <<< "$gpu_csv"
    local chunks="${CHUNKS:-${#gpu_list[@]}}"

    require_dir() {
        if [ ! -d "$1" ]; then
            echo "Missing $2: $1" >&2
            return 1
        fi
    }

    require_file() {
        if [ ! -f "$1" ]; then
            echo "Missing $2: $1" >&2
            return 1
        fi
    }

    require_positive_integer() {
        if ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
            echo "$2 must be a positive integer, got: $1" >&2
            return 1
        fi
    }

    require_positive_integer "$batch_size" "BATCH_SIZE"
    require_positive_integer "$chunks" "CHUNKS"
    if [ "${#gpu_list[@]}" -eq 0 ]; then
        echo "CUDA_VISIBLE_DEVICES did not provide any GPUs" >&2
        return 1
    fi
    if [ "$chunks" -gt "${#gpu_list[@]}" ]; then
        echo "CHUNKS=$chunks exceeds the ${#gpu_list[@]} visible GPUs." >&2
        echo "Each chunk loads a large encoder and decoder; GPU oversubscription is disabled." >&2
        return 1
    fi

    if [ "${#padding_sizes[@]}" -eq 0 ]; then
        echo "PADDING_SIZES cannot be empty" >&2
        return 1
    fi
    local padding_size
    for padding_size in "${padding_sizes[@]}"; do
        if [ "$padding_size" != "256" ]; then
            echo "$model_name uses an official native-256 checkpoint." >&2
            echo "Only PADDING_SIZES=256 is supported; got $padding_size." >&2
            return 1
        fi
    done

    require_dir "$data_root/images/text_data" "TokBench text image root" || return 1
    require_dir "$data_root/images/face_data" "TokBench face image root" || return 1
    require_file "$script_dir/$python_entry" "$model_name Python entrypoint" || return 1
    require_file "$raev2_path/src/stage1/rae.py" "RAEv2 source code" || return 1
    require_file "$raev2_path/configs/decoder/ViTXL/config.json" "RAEv2 decoder config" || return 1
    require_file "$model_root/$encoder_rel" "$model_name encoder checkpoint" || return 1
    require_file "$model_root/$decoder_rel" "$model_name decoder checkpoint" || return 1
    require_file "$model_root/$stats_rel" "$model_name normalization statistics" || return 1
    if [ "$needs_dinov3" = "1" ]; then
        require_file "$dinov3_path/hubconf.py" "local DINOv3 repository" || return 1
    fi

    launch_dataset() {
        local category="$1"
        local dataset="$2"
        local size="$3"
        local input_dir="$data_root/images/${category}_data/$dataset"
        local output_base="$recon_root/$output_name/${category}_data/$dataset"
        require_dir "$input_dir" "$category dataset '$dataset'" || return 1

        echo "[$output_name] padding=$size dataset=$dataset ($category)"
        local -a pids=()
        local idx
        for ((idx = 0; idx < chunks; idx++)); do
            local gpu_idx=$((idx % ${#gpu_list[@]}))
            local gpu="${gpu_list[$gpu_idx]}"
            CUDA_VISIBLE_DEVICES="$gpu" python "$script_dir/$python_entry" \
                --image_path "$input_dir" \
                --save_path "$output_base" \
                --model_name "$model_name" \
                --raev2_path "$raev2_path" \
                --dinov3_path "$dinov3_path" \
                --model_root "$model_root" \
                --padding_size "$size" \
                --batch_size "$batch_size" \
                --num_chunks "$chunks" \
                --chunk_idx "$idx" \
                "${python_extra_args[@]}" &
            pids+=("$!")
        done

        local failed=0
        local pid
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed=1
            fi
        done
        if [ "$failed" -ne 0 ]; then
            echo "[$output_name] reconstruction failed for $category/$dataset" >&2
            return 1
        fi
    }

    cd "$script_dir"
    local dataset
    for dataset in "${text_datas[@]}"; do
        for padding_size in "${padding_sizes[@]}"; do
            launch_dataset "text" "$dataset" "$padding_size" || return 1
        done
    done
    for dataset in "${face_datas[@]}"; do
        for padding_size in "${padding_sizes[@]}"; do
            launch_dataset "face" "$dataset" "$padding_size" || return 1
        done
    done

    echo "[$output_name] image reconstruction complete -> $recon_root/$output_name"
}
