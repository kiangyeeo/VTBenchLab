#!/bin/bash
# Backward-compatible launcher for both requested Cosmos DV tokenizer benchmarks.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/cosmos_dv4x8x8.sh"
bash "$SCRIPT_DIR/cosmos_dv8x16x16.sh"
