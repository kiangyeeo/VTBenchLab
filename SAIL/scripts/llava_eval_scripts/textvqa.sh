#!/bin/bash

DATA_DIR=''

model_path='llava_checkpoints/llava_stage2_star7XL_d1024_scale20_sequence_full'
answer_name=${model_path##*/}

python llava_train/eval/model_vqa_loader.py \
    --model-path $model_path \
    --question-file llava_train/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $DATA_DIR/TextVQA/train_images \
    --answers-file $model_path/textvqa/$answer_name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava_train/eval/eval_textvqa.py \
    --annotation-file $DATA_DIR/TextVQA/TextVQA_0.5.1_val.json \
    --result-file $model_path/textvqa/$answer_name.jsonl