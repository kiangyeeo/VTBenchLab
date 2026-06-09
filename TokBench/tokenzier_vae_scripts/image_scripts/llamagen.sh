#!/bin/bash
# Backward-compatible launcher for the LlamaGen VQ-16 tokenizer benchmark.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/llamagen_vq16.sh" "$@"
