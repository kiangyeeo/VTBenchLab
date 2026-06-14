#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

DATA_DIR=''
model_path='llava_checkpoints/llava_stage2_star7XL_d1024_scale20_sequence_full'

SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava_train/eval/model_vqa_loader.py \
        --model-path $model_path \
        --question-file llava_train/eval/vqav2/$SPLIT.jsonl \
        --image-folder $DATA_DIR/vqav2/test2015 \
        --answers-file $model_path/vqav2/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$model_path/vqav2/$SPLIT/$model_path/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $model_path/vqav2/$SPLIT/$model_path/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava_train/eval/convert_vqav2_for_submission.py --split $SPLIT --model_path $model_path --dir $model_path/vqav2

