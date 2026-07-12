#!/usr/bin/env python
"""Summarize completed K-shot-aligned ImageNet full-support probes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from linear_probe_tokenizers import MODEL_NAMES


FIELDS = (
    "model", "train_samples", "mean_samples_per_class", "min_samples_per_class",
    "max_samples_per_class", "selected_C", "selection_top1", "top1", "top5", "converged",
)


def load_rows(output_root: Path) -> list[dict]:
    rows = []
    for model in MODEL_NAMES:
        path = output_root / model / "full_support" / "results.json"
        if not path.exists():
            continue
        with path.open() as handle:
            result = json.load(handle)["result"]
        counts = result["train_samples_per_class"]
        rows.append(
            {
                "model": model,
                "train_samples": result["train_samples"],
                "mean_samples_per_class": counts["mean"],
                "min_samples_per_class": counts["min"],
                "max_samples_per_class": counts["max"],
                "selected_C": result["selected_C"],
                "selection_top1": result["selection_top1"],
                "top1": result["top1"],
                "top5": result["top5"],
                "converged": result["converged"],
            }
        )
    return rows


def markdown(rows: list[dict]) -> str:
    lines = [
        "# ImageNet K-shot-aligned full-support linear probing", "",
        "The final classifier uses the complete disjoint 90% support pool. The 10% C-selection split is excluded.",
        "",
        "| model | train samples | mean/class | selected C | selection Top-1 | val Top-1 | val Top-5 | converged |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        selection = "—" if row["selection_top1"] is None else f"{row['selection_top1']:.2f}"
        lines.append(
            f"| {row['model']} | {row['train_samples']} | {row['mean_samples_per_class']:.2f} "
            f"| {row['selected_C']:.6g} | {selection} | {row['top1']:.2f} "
            f"| {row['top5']:.2f} | {'yes' if row['converged'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = load_rows(args.output_root)
    csv_path = args.output_root / "summary_full_support.csv"
    md_path = args.output_root / "summary_full_support.md"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(markdown(rows))
    print(md_path.read_text())
    print(f"Summary: {csv_path} / {md_path}")


if __name__ == "__main__":
    main()
