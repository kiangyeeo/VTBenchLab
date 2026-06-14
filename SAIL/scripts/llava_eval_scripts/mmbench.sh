#!/bin/bash

DATA_DIR=''
SPLIT="mmbench_dev_20230712"


model_path='llava_checkpoints/llava_stage2_star7XL_d1024_scale20_sequence_full'
answer_name=${model_path##*/}

python llava_train/eval/model_vqa_mmbench.py \
    --model-path $model_path \
    --question-file $DATA_DIR/mmbench/$SPLIT.tsv \
    --answers-file $model_path/mmbench/$SPLIT/$answer_name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $model_path/mmbench/answers_upload/$SPLIT

python llava_train/eval/convert_mmbench_for_submission.py \
    --annotation-file $DATA_DIR/mmbench/$SPLIT.tsv \
    --result-dir $model_path/mmbench/$SPLIT \
    --upload-dir $model_path/mmbench/answers_upload/$SPLIT \
    --experiment $answer_name