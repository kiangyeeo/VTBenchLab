#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
for split in train validation test; do
  "$PYTHON_BIN" -m vtm_lcg.cvrvtm.cache \
    --config "configs/coco_karpathy_full/phase0_${split}.yaml" \
    --artifact-root "artifacts/cvrvtm/coco_karpathy_full/phase0/${split}" \
    --all \
    "$@"
done
