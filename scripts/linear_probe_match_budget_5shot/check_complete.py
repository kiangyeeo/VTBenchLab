#!/usr/bin/env python3
"""Return success only for a completed five-shot match-budget result."""

import json
from pathlib import Path
import sys


PROTOCOL_VERSION = "tokenizer_linear_probe_imagenet_5shot_1epoch_v1"
EXPECTED_ITERATION = 5


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULTS_EVAL_LINEAR_JSON")
    path = Path(sys.argv[1])
    if not path.is_file():
        return 1
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 1
    classifiers = payload.get("classifiers", [])
    best_name = payload.get("best_classifier", {}).get("name")
    return 0 if (
        payload.get("protocol_version") == PROTOCOL_VERSION
        and payload.get("iteration") == EXPECTED_ITERATION
        and len(classifiers) == 13
        and sum(item.get("name") == best_name for item in classifiers) == 1
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
