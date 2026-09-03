#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$workspace_root"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/cache/ma-user/tmp/matplotlib}"

stage="${1:-all}"

extract_or_tokenize() {
  local domain="$1"
  local stem="$domain"
  if [[ "$domain" == "answer_other" ]]; then
    stem="answer"
    domain="answer"
  fi
  if [[ -f "lar/text/${stem}__coco4618.npy" ]]; then
    if [[ ! -f "lar/text/${stem}__coco4618.token_lengths.npy" ]]; then
      python -m lar.extract_text --domain "$domain" --image-set coco4618 --tokenize-only
    fi
  else
    python -m lar.extract_text --domain "$domain" --image-set coco4618
  fi
}

run_text() {
  extract_or_tokenize caption
  extract_or_tokenize answer_other
  extract_or_tokenize question_other
  extract_or_tokenize qa_concat
  extract_or_tokenize answer_all_types

  if [[ ! -f lar/text/eval_answer__coco4618.npy ]]; then
    python -m lar.extract_text --domain eval_answer --image-set coco4618 --prepare-only
    matched_n="$(python -c 'import json; print(json.load(open("lar/text/eval_answer__coco4618.meta.json"))["dataset"]["matched_N"])')"
    if [[ "$matched_n" -gt 0 ]]; then
      python -m lar.extract_text --domain eval_answer --image-set coco4618
    else
      echo "T6 eval_answer coverage is 0; add the train-split files listed in lar/configs/eval_answer_sources.yaml."
    fi
  fi
}

run_metrics() {
  python -m lar.compute_metrics_v3
}

run_eval() {
  python -m lar.eval_e3
  python -m lar.eval_e4
}

case "$stage" in
  text) run_text ;;
  metrics) run_metrics ;;
  eval) run_eval ;;
  all) run_text; run_metrics; run_eval ;;
  *) echo "usage: $0 [text|metrics|eval|all]" >&2; exit 2 ;;
esac
