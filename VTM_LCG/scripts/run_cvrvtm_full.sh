#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/run_cvrvtm_phase0_full.sh" "$@"
"$SCRIPT_DIR/run_cvrvtm_phase1_full.sh"
