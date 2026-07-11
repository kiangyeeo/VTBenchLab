#!/usr/bin/env python
"""Collect tokenizer k-shot results into per-seed and mean/std summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


MODELS = ("unitok", "toklips", "toklipl", "vilau", "metaclip")
SEED_FIELDS = (
    "model",
    "seed",
    "shot",
    "top1",
    "top5",
    "selected_C",
    "selection_top1",
    "converged",
)
AGGREGATE_FIELDS = (
    "model",
    "shot",
    "n_seeds",
    "seeds",
    "top1_mean",
    "top1_std",
    "top5_mean",
    "top5_std",
    "selected_Cs",
)


def _load_rows(output_root: Path, seeds: list[int]) -> list[dict]:
    rows = []
    for model in MODELS:
        for seed in seeds:
            path = output_root / model / f"seed{seed}" / "results.json"
            if not path.exists():
                continue
            with path.open() as handle:
                payload = json.load(handle)
            for result in payload.get("results", []):
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "shot": int(result["shot"]),
                        "top1": float(result["top1"]),
                        "top5": float(result["top5"]),
                        "selected_C": float(result.get("selected_C", result.get("C"))),
                        "selection_top1": result.get("selection_top1"),
                        "converged": bool(result["converged"]),
                    }
                )
    return rows


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _seed_markdown(rows: list[dict], seed: int) -> str:
    seed_rows = [row for row in rows if row["seed"] == seed]
    shots = sorted({row["shot"] for row in seed_rows})
    by_model = {(row["model"], row["shot"]): row for row in seed_rows}
    lines = [
        f"# ImageNet k-shot linear probing — seed {seed}",
        "",
        "Official ImageNet val 50k Top-1 (%). C is selected on the disjoint train selection split.",
        "",
        "| model | " + " | ".join(f"{shot}-shot" for shot in shots) + " |",
        "|---|" + "---:|" * len(shots),
    ]
    for model in MODELS:
        values = []
        for shot in shots:
            row = by_model.get((model, shot))
            values.append("—" if row is None else f"{row['top1']:.2f}")
        lines.append(f"| {model} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["shot"]), []).append(row)
    aggregates = []
    for model in MODELS:
        for shot in sorted({key_shot for key_model, key_shot in grouped if key_model == model}):
            group = sorted(grouped[(model, shot)], key=lambda row: row["seed"])
            top1 = [row["top1"] for row in group]
            top5 = [row["top5"] for row in group]
            aggregates.append(
                {
                    "model": model,
                    "shot": shot,
                    "n_seeds": len(group),
                    "seeds": " ".join(str(row["seed"]) for row in group),
                    "top1_mean": statistics.fmean(top1),
                    "top1_std": statistics.pstdev(top1),
                    "top5_mean": statistics.fmean(top5),
                    "top5_std": statistics.pstdev(top5),
                    "selected_Cs": " ".join(f"{row['selected_C']:.12g}" for row in group),
                }
            )
    return aggregates


def _aggregate_markdown(rows: list[dict], requested_seeds: list[int]) -> str:
    shots = sorted({row["shot"] for row in rows})
    by_model = {(row["model"], row["shot"]): row for row in rows}
    lines = [
        "# ImageNet k-shot linear probing — mean ± population std",
        "",
        f"Requested support seeds: {', '.join(map(str, requested_seeds))}. Values use all completed seeds shown in CSV.",
        "",
        "| model | " + " | ".join(f"{shot}-shot" for shot in shots) + " |",
        "|---|" + "---:|" * len(shots),
    ]
    for model in MODELS:
        values = []
        for shot in shots:
            row = by_model.get((model, shot))
            values.append(
                "—"
                if row is None
                else f"{row['top1_mean']:.2f} ± {row['top1_std']:.2f} (n={row['n_seeds']})"
            )
        lines.append(f"| {model} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args(argv)
    args.seeds = sorted(set(args.seeds))
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.output_root, args.seeds)
    for seed in args.seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        csv_path = args.output_root / f"summary_seed{seed}.csv"
        markdown_path = args.output_root / f"summary_seed{seed}.md"
        _write_csv(csv_path, SEED_FIELDS, seed_rows)
        markdown_path.write_text(_seed_markdown(rows, seed))
        print(f"Seed {seed}: {csv_path} / {markdown_path}")

    aggregates = _aggregate_rows(rows)
    aggregate_csv = args.output_root / "summary_mean_std.csv"
    aggregate_markdown = args.output_root / "summary_mean_std.md"
    _write_csv(aggregate_csv, AGGREGATE_FIELDS, aggregates)
    aggregate_markdown.write_text(_aggregate_markdown(aggregates, args.seeds))
    print(aggregate_markdown.read_text())
    print(f"Aggregate: {aggregate_csv} / {aggregate_markdown}")


if __name__ == "__main__":
    main()
