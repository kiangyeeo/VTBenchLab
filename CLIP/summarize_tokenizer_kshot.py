#!/usr/bin/env python
"""Collect tokenizer k-shot JSON results into a compact CSV and Markdown table."""

import argparse
import csv
import json
from pathlib import Path


MODELS = ("unitok", "toklips", "toklipl", "vilau", "metaclip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for model in MODELS:
        path = args.output_root / model / f"seed{args.seed}" / "results.json"
        if not path.exists():
            continue
        with path.open() as handle:
            payload = json.load(handle)
        for result in payload["results"]:
            rows.append(
                {
                    "model": model,
                    "seed": args.seed,
                    "shot": result["shot"],
                    "top1": result["top1"],
                    "top5": result["top5"],
                    "C": result["C"],
                    "converged": result["converged"],
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / f"summary_seed{args.seed}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "seed", "shot", "top1", "top5", "C", "converged"))
        writer.writeheader()
        writer.writerows(rows)

    shots = sorted({row["shot"] for row in rows})
    by_model = {(row["model"], row["shot"]): row for row in rows}
    lines = [
        "| model | " + " | ".join(f"{shot}-shot" for shot in shots) + " |",
        "|---|" + "---:|" * len(shots),
    ]
    for model in MODELS:
        values = []
        for shot in shots:
            row = by_model.get((model, shot))
            values.append("—" if row is None else f"{row['top1']:.2f}")
        lines.append(f"| {model} | " + " | ".join(values) + " |")
    markdown = "\n".join(lines) + "\n"
    markdown_path = args.output_root / f"summary_seed{args.seed}.md"
    markdown_path.write_text(markdown)
    print(markdown)
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
