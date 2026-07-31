#!/bin/bash
# Sequential RAEv2 K sweep for TokBench reconstruction and/or evaluation.
#
# Examples:
#   SWEEP_MODE=reconstruct RAEV2_KS="1 7 23" CUDA_VISIBLE_DEVICES=0 bash raev2_k_sweep.sh
#   SWEEP_MODE=evaluate RAEV2_KS="1 7 23" CUDA_VISIBLE_DEVICES=0 bash raev2_k_sweep.sh
#   SWEEP_MODE=all RAEV2_KS="1 7 23" CUDA_VISIBLE_DEVICES=0 bash raev2_k_sweep.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKBENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SWEEP_MODE="${SWEEP_MODE:-reconstruct}"
OUT_ROOT="${OUT_ROOT:-$TOKBENCH_ROOT/image_outputs}"

read -r -a k_values <<< "${RAEV2_KS:-1 7 23}"
if [ "${#k_values[@]}" -eq 0 ]; then
    echo "RAEV2_KS cannot be empty" >&2
    exit 2
fi

case "$SWEEP_MODE" in
    reconstruct|evaluate|all)
        ;;
    *)
        echo "SWEEP_MODE must be reconstruct, evaluate, or all; got: $SWEEP_MODE" >&2
        exit 2
        ;;
esac

for k in "${k_values[@]}"; do
    case "$k" in
        1|7|23)
            ;;
        *)
            echo "Every RAEV2_KS value must be one of: 1, 7, 23; got: $k" >&2
            exit 2
            ;;
    esac
done

reconstruct_k() {
    local k="$1"
    local tokenizer_name="raev2_k${k}"
    echo "Reconstructing $tokenizer_name"
    RAEV2_K="$k" OUTPUT_NAME="$tokenizer_name" bash "$SCRIPT_DIR/raev2.sh"
}

evaluate_k() {
    local k="$1"
    local tokenizer_name="raev2_k${k}"
    echo "Evaluating $tokenizer_name"
    TOKENIZER_NAME="$tokenizer_name" \
        OUT_DIR="$OUT_ROOT/${tokenizer_name}_256" \
        RES=256 \
        bash "$TOKBENCH_ROOT/image_eval.sh"
}

for k in "${k_values[@]}"; do
    case "$SWEEP_MODE" in
        reconstruct)
            reconstruct_k "$k"
            ;;
        evaluate)
            evaluate_k "$k"
            ;;
        all)
            reconstruct_k "$k"
            evaluate_k "$k"
            ;;
    esac
done

echo "RAEv2 K sweep complete: mode=$SWEEP_MODE K=${k_values[*]}"
