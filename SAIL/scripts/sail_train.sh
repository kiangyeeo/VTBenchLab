#!/bin/bash
# ----------------------TRAIN SETTING------------------------

epoch_num=100
lr=1e-5
bs=32768
d=1024
width_factor=8
logit_scale=20
logit_bias=-10

text_embedding_list="data/tensor_data/text_embedding/NV-Embed-v2/yfcc15m_raw_caption data/tensor_data/text_embedding/NV-Embed-v2/dreamclipcc3m_raw_caption data/tensor_data/text_embedding/NV-Embed-v2/dreamclipcc12mhf_raw_caption" 
image_embedding_list="data/tensor_data/image_embedding/dinov2-base/yfcc15m data/tensor_data/image_embedding/dinov2-base/dreamclipcc3m data/tensor_data/image_embedding/dinov2-base/dreamclipcc12mhf"
extra_text_embedding_list="data/tensor_data/text_embedding/NV-Embed-v2/yfcc15m_shortSV_captions data/tensor_data/text_embedding/NV-Embed-v2/dreamclipcc3m_longSV_captions data/tensor_data/text_embedding/NV-Embed-v2/dreamclipcc12mhf_shortSV_captions"
output_name="sail_l_nv2_merged23m"
# ------------------------------------------------------------


python main.py \
    --text-embedding-list $text_embedding_list \
    --image-embedding-list $image_embedding_list \
    --extra-text-embedding-list $extra_text_embedding_list \
    --val-frequency 1 \
    --dataset-type embedding \
    --seed 42 \
    --resume latest \
    --save-frequency 2 \
    --report-to wandb \
    --batch-size $bs \
    --lr $lr \
    --epochs $epoch_num \
    --workers 0 \
    --optimizer lion \
    --siglip \
    --wd 1e-4 \
    --target-dimension $d \
    --linear-type star \
    --diagonal-weight 0 \
    --log-every-n-steps 5 \
    --wandb-project-name sail_train \
    --name $output_name \
    --logit_scale $logit_scale \
    --logit_bias $logit_bias



if [ $? -ne 0 ]; then
    echo "Training failed. Checking for checkpoints..."
    
    if ls ./logs/$output_name/checkpoints/*.pt 1> /dev/null 2>&1; then
        echo "Checkpoint file(s) found. Keeping the log directory."
    else
        echo "No checkpoint files found. Cleaning up ./logs/${output_name}"
        rm -rf ./logs/$output_name
    fi
fi
