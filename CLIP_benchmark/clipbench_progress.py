#!/usr/bin/env python
"""Show progress for CLIP benchmark JSON outputs."""

import argparse
import glob
import json
import os
import time
from collections import defaultdict


DEFAULT_REPO_ROOT = "/cache/ma-user/VTBenchLab"


def read_models(path):
    models = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            model, pretrained = line.split(",", 1)
            models.append((model, pretrained))
    return models


def read_datasets(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def bar(done, total, width=32):
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = round(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def collect_completed(out_dir, models, datasets):
    valid_models = set(models)
    valid_datasets = set(datasets)
    completed = defaultdict(set)
    bad_files = 0

    for path in glob.glob(os.path.join(out_dir, "*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            bad_files += 1
            continue

        model_key = (data.get("model"), data.get("pretrained"))
        dataset = data.get("dataset")
        if model_key in valid_models and dataset in valid_datasets:
            completed[dataset].add(model_key)

    return completed, bad_files


def render(args):
    models = read_models(args.model_list)
    datasets = read_datasets(args.dataset_list)
    completed, bad_files = collect_completed(args.out_dir, models, datasets)

    total_runs = len(models) * len(datasets)
    done_runs = sum(len(completed.get(dataset, set())) for dataset in datasets)
    complete_datasets = sum(1 for dataset in datasets if len(completed.get(dataset, set())) == len(models))
    pct = 100.0 * done_runs / total_runs if total_runs else 0.0

    lines = []
    lines.append(time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append(f"runs     {bar(done_runs, total_runs)} {done_runs}/{total_runs} ({pct:.1f}%)")
    lines.append(f"datasets {bar(complete_datasets, len(datasets))} {complete_datasets}/{len(datasets)} complete")
    if bad_files:
        lines.append(f"ignored unreadable json files: {bad_files}")
    lines.append("")

    for idx, dataset in enumerate(datasets, start=1):
        n_done = len(completed.get(dataset, set()))
        if args.incomplete_only and n_done == len(models):
            continue
        ds_pct = 100.0 * n_done / len(models) if models else 0.0
        lines.append(
            f"{idx:02d}/{len(datasets):02d} {bar(n_done, len(models), width=16)} "
            f"{n_done:2d}/{len(models):2d} {ds_pct:5.1f}% {dataset}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Show CLIP benchmark progress.")
    parser.add_argument(
        "--model-list",
        default=os.path.join(DEFAULT_REPO_ROOT, "CLIP_benchmark", "model_lists", "models.txt"),
    )
    parser.add_argument(
        "--dataset-list",
        default=os.path.join(DEFAULT_REPO_ROOT, "CLIP_benchmark", "clip_benchmark", "datasets", "webdatasets.txt"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(DEFAULT_REPO_ROOT, "outputs", "clip_benchmark"),
    )
    parser.add_argument("--incomplete-only", action="store_true", help="Hide completed datasets.")
    parser.add_argument("--watch", type=float, default=0, help="Refresh every N seconds.")
    args = parser.parse_args()

    if args.watch:
        while True:
            print("\033[2J\033[H", end="")
            print(render(args), flush=True)
            time.sleep(args.watch)
    else:
        print(render(args))


if __name__ == "__main__":
    main()
