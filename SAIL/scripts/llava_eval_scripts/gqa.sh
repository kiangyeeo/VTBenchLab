#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
# print the gpu list
echo "GPU list: ${GPULIST[@]}"
CHUNKS=${#GPULIST[@]}

model_path='llava_checkpoints/llava_stage2_star7XL_d1024_scale20_sequence_full'
# model_path=$1
SPLIT="llava_gqa_testdev_balanced"
DATA_DIR=""

IDX=0
python llava_train/eval/model_vqa_loader.py \
    --model-path $model_path \
    --question-file llava_train/eval/gqa/$SPLIT.jsonl \
    --image-folder $DATA_DIR/images \
    --answers-file $model_path/gqa/$SPLIT/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --temperature 0 \
    --conv-mode vicuna_v1


output_file=$model_path/gqa/$SPLIT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Processing chunk $IDX"
    cat $model_path/gqa/$SPLIT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava_train/eval/convert_gqa_for_eval.py --src $output_file --dst $DATA_DIR/testdev_balanced_predictions.json

cd $DATA_DIR
python eval.py --tier testdev_balanced
