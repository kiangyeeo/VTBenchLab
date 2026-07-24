#!/usr/bin/env python
"""Aggregate completed ImageNet compatibility runs without extra dependencies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, stdev
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=EXPERIMENT_ROOT / "outputs" / "imagenet1k",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "outputs" / "imagenet1k_summary",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    summaries: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted(input_root.glob("*/*/seed*/summary.json")):
        with path.open() as handle:
            summary = json.load(handle)
        key = (summary["model"], summary["readout"], int(summary["seed"]))
        summaries[key] = summary
        for evaluation in summary["evaluations"]:
            run_rows.append(
                {
                    "model": summary["model"],
                    "readout": summary["readout"],
                    "seed": int(summary["seed"]),
                    "step": int(evaluation["step"]),
                    "top1": float(evaluation["top1"]),
                    "top5": float(evaluation["top5"]),
                    "loss": float(evaluation["loss"]),
                    "top1_aulc_log_updates": float(summary["top1_aulc_log_updates"]),
                    "trainable_parameters": int(summary["trainable_parameters"]),
                    "summary_path": str(path),
                }
            )
    if not run_rows:
        raise FileNotFoundError(f"No completed summary.json files found under {input_root}")

    run_fields = [
        "model",
        "readout",
        "seed",
        "step",
        "top1",
        "top5",
        "loss",
        "top1_aulc_log_updates",
        "trainable_parameters",
        "summary_path",
    ]
    write_csv(output_dir / "runs.csv", run_rows, run_fields)

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in run_rows:
        grouped.setdefault((row["model"], row["readout"], row["step"]), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (model, readout, step), rows in sorted(grouped.items()):
        top1_values = [float(row["top1"]) for row in rows]
        top5_values = [float(row["top5"]) for row in rows]
        aggregate_rows.append(
            {
                "model": model,
                "readout": readout,
                "step": step,
                "seeds": len(rows),
                "top1_mean": fmean(top1_values),
                "top1_std": stdev(top1_values) if len(top1_values) > 1 else 0.0,
                "top5_mean": fmean(top5_values),
                "top5_std": stdev(top5_values) if len(top5_values) > 1 else 0.0,
                "top1_aulc_mean": fmean(
                    float(row["top1_aulc_log_updates"]) for row in rows
                ),
                "top1_aulc_std": (
                    stdev(float(row["top1_aulc_log_updates"]) for row in rows)
                    if len(rows) > 1
                    else 0.0
                ),
            }
        )
    aggregate_fields = [
        "model",
        "readout",
        "step",
        "seeds",
        "top1_mean",
        "top1_std",
        "top5_mean",
        "top5_std",
        "top1_aulc_mean",
        "top1_aulc_std",
    ]
    write_csv(output_dir / "aggregate.csv", aggregate_rows, aggregate_fields)

    gains: list[dict[str, Any]] = []
    for (model, readout, seed), transformer_summary in sorted(summaries.items()):
        if readout != "transformer":
            continue
        mlp_summary = summaries.get((model, "gap_mlp", seed))
        if mlp_summary is None:
            continue
        mlp_by_step = {
            int(item["step"]): item for item in mlp_summary["evaluations"]
        }
        for transformer_eval in transformer_summary["evaluations"]:
            step = int(transformer_eval["step"])
            if step not in mlp_by_step:
                continue
            gains.append(
                {
                    "model": model,
                    "seed": seed,
                    "step": step,
                    "transformer_top1": float(transformer_eval["top1"]),
                    "gap_mlp_top1": float(mlp_by_step[step]["top1"]),
                    "transformer_gain_top1": (
                        float(transformer_eval["top1"])
                        - float(mlp_by_step[step]["top1"])
                    ),
                }
            )
    if gains:
        write_csv(
            output_dir / "transformer_gain.csv",
            gains,
            [
                "model",
                "seed",
                "step",
                "transformer_top1",
                "gap_mlp_top1",
                "transformer_gain_top1",
            ],
        )

    final_rows = [
        row for row in aggregate_rows if int(row["step"]) == max(
            int(candidate["step"])
            for candidate in aggregate_rows
            if candidate["model"] == row["model"]
            and candidate["readout"] == row["readout"]
        )
    ]
    markdown = [
        "# ImageNet-1k Transformer compatibility summary",
        "",
        "| Model | Readout | Seeds | Final Top-1 | Top-1 AULC |",
        "|---|---|---:|---:|---:|",
    ]
    for row in final_rows:
        markdown.append(
            f"| {row['model']} | {row['readout']} | {row['seeds']} | "
            f"{row['top1_mean']:.3f} ± {row['top1_std']:.3f} | "
            f"{row['top1_aulc_mean']:.3f} ± {row['top1_aulc_std']:.3f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Wrote summaries to {output_dir}")


if __name__ == "__main__":
    main()

