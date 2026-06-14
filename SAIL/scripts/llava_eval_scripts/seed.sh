#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

DATA_DIR=''
CHUNKS=${#GPULIST[@]}
model_path='llava_checkpoints/llava_stage2_star7XL_d1024_scale20_sequence_full'
CKPT=${model_path##*/}


output_file=$model_path/seed/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $model_path/seed/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python llava_train/eval/convert_seed_for_submission.py \
    --annotation-file $DATA_DIR/seedbench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file $model_path/seed/$CKPT.jsonl
