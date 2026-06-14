#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <model_alias> <dataset_alias> <linear|knn|paper_knn> [extra args...]"
  exit 2
fi

MODEL="$1"
DATASET="$2"
TASK="$3"
shift 3

case "$TASK" in
  linear)
    python -m dinov2_probe.eval_linear_probe --models "$MODEL" --datasets "$DATASET" "$@"
    ;;
  knn)
    python -m dinov2_probe.eval_knn_extended --models "$MODEL" --datasets "$DATASET" "$@"
    ;;
  paper_knn)
    if [[ "$DATASET" != "imagenet1k" ]]; then
      echo "paper_knn is only defined for imagenet1k."
      exit 2
    fi
    python -m dinov2_probe.eval_imagenet_knn --models "$MODEL" "$@"
    ;;
  *)
    echo "Unknown task: $TASK"
    exit 2
    ;;
esac

