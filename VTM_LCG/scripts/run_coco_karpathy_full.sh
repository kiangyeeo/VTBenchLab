#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/run_phase0_coco_karpathy_full.sh"
"$SCRIPT_DIR/run_phase1_coco_karpathy_full.sh"

