#!/usr/bin/env python3
"""Print compact cache/budget progress for the 70-tokenizer FLOP sweep."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "scripts/linear_probe_match_budget_5shot_noaug_cached_10epoch/tokenizers.tsv"
OUTPUT = Path(
    os.environ.get(
        "SWEEP_OUTPUT_ROOT",
        ROOT / "outputs/vae_linear_probing_flop_sweep_noaug_cached_1epoch_allbn",
    )
).expanduser().resolve()
CACHE = ROOT / "outputs/vae_linear_probing_flop_sweep_noaug_cached_5epoch_allbn/_feature_cache"


rows = []
with MANIFEST.open("r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip() and not line.startswith("#"):
            rank, requested_id, _model, _head = line.rstrip("\n").split("\t")
            rows.append((int(rank), requested_id))

completed_models = 0
completed_points = 0
total_points = 0
for rank, requested_id in rows:
    schedule_path = OUTPUT / requested_id / "sweep_schedule.json"
    if not schedule_path.is_file():
        progress_files = list(
            (CACHE / requested_id).glob(
                "*/.train_progress.json"
            )
        )
        if progress_files:
            progress_path = max(
                progress_files, key=lambda path: path.stat().st_mtime_ns
            )
            with progress_path.open("r", encoding="utf-8") as handle:
                cache_progress = json.load(handle)
            completed = int(cache_progress["completed"])
            total = int(cache_progress["shape"][0])
            print(
                f"[{rank:02d}/70] {requested_id}: caching train "
                f"{completed}/{total} ({100.0 * completed / total:.1f}%)"
            )
        else:
            print(f"[{rank:02d}/70] {requested_id}: waiting")
        continue
    with schedule_path.open("r", encoding="utf-8") as handle:
        schedule = json.load(handle)
    caps = schedule["cap_shots"]
    done = [
        cap
        for cap in caps
        if (OUTPUT / requested_id / f"cap_{cap:04d}/results_eval_linear.json").is_file()
    ]
    completed_points += len(done)
    total_points += len(caps)
    if len(done) == len(caps):
        completed_models += 1
    latest = str(done[-1]) if done else "cache/training"
    print(f"[{rank:02d}/70] {requested_id}: {len(done)}/{len(caps)} points, latest={latest}")
print(f"TOTAL: models={completed_models}/70, points={completed_points}/{total_points or '?'}")
