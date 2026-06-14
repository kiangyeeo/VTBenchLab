#!/bin/bash
DATA_DIR=''

model_path='llava_checkpoints/llava_stage2_star7XL_d1024_scale20_sequence_full'
answer_name=${model_path##*/}
upload_name='vizwiz_answer_upload_'$answer_name


python llava_train/eval/model_vqa_loader.py \
    --model-path $model_path \
    --question-file llava_train/eval/vizwiz/llava_test.jsonl \
    --image-folder $DATA_DIR/vizwiz/test \
    --answers-file $model_path/vizwiz/$answer_name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava_train/eval/convert_vizwiz_for_submission.py \
    --annotation-file llava_train/eval/vizwiz/llava_test.jsonl \
    --result-file $model_path/vizwiz/$answer_name.jsonl \
    --result-upload-file $model_path/vizwiz/$upload_name.json