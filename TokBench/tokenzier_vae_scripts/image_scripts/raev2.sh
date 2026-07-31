#!/bin/bash
# TokBench reconstruction with the released RAEv2 DINOv3-L K=1/7/23 variants.
# Usage: RAEV2_K=7 CUDA_VISIBLE_DEVICES=0 PADDING_SIZES=256 bash raev2.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAEV2_K="${RAEV2_K:-23}"

case "$RAEV2_K" in
    1|7|23)
        ;;
    *)
        echo "RAEV2_K must be one of: 1, 7, 23; got: $RAEV2_K" >&2
        exit 2
        ;;
esac

MODEL_NAME="raev2"
OUTPUT_NAME="${OUTPUT_NAME:-raev2_k${RAEV2_K}}"
PYTHON_ENTRY="raev2_rec.py"
ENCODER_CKPT_REL="encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DECODER_CKPT_REL="stage1/imagenet/dinov3l-k${RAEV2_K}/decoder.pt"
STATS_CKPT_REL="stage1/imagenet/dinov3l-k${RAEV2_K}/stats.pt"
NEEDS_DINOV3=1
PYTHON_EXTRA_ARGS=(--k "$RAEV2_K")

source "$SCRIPT_DIR/rae_stage1_launch_common.sh"
run_rae_stage1
