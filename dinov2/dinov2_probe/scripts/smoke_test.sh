#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

python -m dinov2_probe.eval_linear_probe \
  --models dinov2_vits14 \
  --datasets cifar10 \
  --smoke \
  --epochs 1 \
  --feature-batch-size 32 \
  --train-batch-size 128 \
  --allow-download \
  --dataset-download \
  "$@"

python -m dinov2_probe.collect_results "$@"

