#!/bin/bash
# TokBench reconstruction with the I-JEPA-H K=1 representation tokenizer.
# Usage: CUDA_VISIBLE_DEVICES=0 PADDING_SIZES=256 bash ijepa.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="ijepa"
PYTHON_ENTRY="ijepa_rec.py"
ENCODER_CKPT_REL="encoders/ijepa/ijepa_vith.pth"
DECODER_CKPT_REL="stage1/imagenet/jepa-h-k1/decoder.pt"
STATS_CKPT_REL="stage1/imagenet/jepa-h-k1/stats.pt"
NEEDS_DINOV3=0

source "$SCRIPT_DIR/rae_stage1_launch_common.sh"
run_rae_stage1
