#!/bin/bash

DATA_DIR=''
model_path='llava_checkpoints/llava_stage2_star7XL_d1024_scale20_sequence_full'
# model_path=$1
answer_name=${model_path##*/}

python llava_train/eval/model_vqa_loader.py \
    --model-path $model_path \
    --question-file llava_train/eval/pope/llava_pope_test.jsonl \
    --image-folder $DATA_DIR/coco2014/val2014 \
    --answers-file $model_path/pope/$SPLIT/$answer_name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava_train/eval/eval_pope.py \
    --annotation-dir llava_train/eval/pope \
    --question-file llava_train/eval/pope/llava_pope_test.jsonl \
    --result-file $model_path/pope/$SPLIT/$answer_name.jsonl