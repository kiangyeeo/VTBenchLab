#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" -m vtm_lcg.cvrvtm.cache \
  --config configs/phase0_smoke.yaml \
  --artifact-root artifacts/cvrvtm/phase0_smoke \
  --all \
  "$@"

"$PYTHON_BIN" -m vtm_lcg.cvrvtm.train \
  --config configs/cvrvtm/phase1_smoke.yaml \
  --all
