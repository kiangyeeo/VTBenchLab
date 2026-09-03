"""Record each tokenizer's visual sequence shape before --clean-token-cache deletes it.

The sweep worker writes ``tokens/<name>/cache.json`` (which carries the
``[records, seq_len, dim]`` shape) and then removes the whole directory once the
projector and loss results are safe. Sequence length is the main confound for a
caption-NLL proxy, so it has to be captured while the cache still exists. This
poller only reads; it never touches the worker's files.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .utils import atomic_write_json, load_config, resolve_path


def collect(tokens_root: Path, known: dict) -> int:
    added = 0
    if not tokens_root.is_dir():
        return added
    for cache_path in sorted(tokens_root.glob("*/cache.json")):
        name = cache_path.parent.name
        if name in known:
            continue
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue  # mid-write; pick it up on the next pass
        shape = payload.get("shape")
        if not shape or len(shape) != 3:
            continue
        known[name] = {
            "tokenizer": name,
            "record_count": int(shape[0]),
            "seq_len": int(shape[1]),
            "feature_dim": int(shape[2]),
            "family": payload.get("spec", {}).get("family"),
            "cache_complete": bool(payload.get("complete")),
        }
        added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument(
        "--stop-after-idle-seconds",
        type=float,
        default=0.0,
        help="Exit once no new tokenizer appears for this long (0 = run until killed).",
    )
    args = parser.parse_args()

    config, _ = load_config(args.config)
    root = resolve_path(config, config["runtime"]["artifact_root"])
    tokens_root = root / "tokens"
    output_path = root / "analysis" / "token_shapes.json"

    known: dict = {}
    if output_path.is_file():
        known = {
            row["tokenizer"]: row
            for row in json.loads(output_path.read_text(encoding="utf-8"))["rows"]
        }
    expected = set(config["tokenizers"])
    last_change = time.monotonic()

    while True:
        if collect(tokens_root, known):
            atomic_write_json(
                output_path,
                {
                    "schema_version": 1,
                    "purpose": "confound控制: visual sequence length and feature dim per tokenizer",
                    "captured": len(known),
                    "expected": len(expected),
                    "missing": sorted(expected - set(known)),
                    "rows": [known[name] for name in sorted(known)],
                },
            )
            print(f"captured {len(known)}/{len(expected)}: {sorted(expected - set(known))[:3]}...")
            last_change = time.monotonic()
        if len(known) >= len(expected):
            print("all tokenizers captured")
            return
        if (
            args.stop_after_idle_seconds > 0
            and time.monotonic() - last_change > args.stop_after_idle_seconds
        ):
            print("idle timeout; exiting")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
