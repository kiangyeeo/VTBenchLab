#!/bin/bash
# TokBench reconstruction with RAEv2: DINOv3-L, K=23 multi-layer aggregation.
# Usage: CUDA_VISIBLE_DEVICES=0 PADDING_SIZES=256 bash raev2.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="raev2"
PYTHON_ENTRY="raev2_rec.py"
ENCODER_CKPT_REL="encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DECODER_CKPT_REL="stage1/imagenet/dinov3l-k23/decoder.pt"
STATS_CKPT_REL="stage1/imagenet/dinov3l-k23/stats.pt"
NEEDS_DINOV3=1

source "$SCRIPT_DIR/rae_stage1_launch_common.sh"
run_rae_stage1
