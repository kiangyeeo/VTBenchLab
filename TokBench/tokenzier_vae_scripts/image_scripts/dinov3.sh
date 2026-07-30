#!/bin/bash
# TokBench reconstruction with the DINOv3-L K=1 representation tokenizer.
# Usage: CUDA_VISIBLE_DEVICES=0 PADDING_SIZES=256 bash dinov3.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="dinov3"
PYTHON_ENTRY="dinov3_rec.py"
ENCODER_CKPT_REL="encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DECODER_CKPT_REL="stage1/imagenet/dinov3l-k1/decoder.pt"
STATS_CKPT_REL="stage1/imagenet/dinov3l-k1/stats.pt"
NEEDS_DINOV3=1

source "$SCRIPT_DIR/rae_stage1_launch_common.sh"
run_rae_stage1
