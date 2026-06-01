#!/usr/bin/env python3
"""Collect GVT benchmark metric json files into json and markdown summaries."""

import argparse
import json
from pathlib import Path
from statistics import mean


SUMMARY_KEYS = [
    "VQA Acc",
    "COCO Caption CIDEr",
    "COCO Caption SPICE",
    "COCO-OC Acc",
    "COCO-MCI Acc",
    "VCR-OC Acc",
    "VCR-MCI Acc",
]


def read_json(path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def normalize_metric(name, value, normalize=True):
    if value is None or not normalize:
        return value
    value = float(value)
    if name == "COCO Caption CIDEr" and value <= 10:
        return value * 100.0
    if name == "COCO Caption SPICE" and value <= 1:
        return value * 100.0
    if name.endswith("Acc") and value <= 1:
        return value * 100.0
    return value


def update_summary(summary, path, metrics, normalize=True):
    name = path.name

    if name.startswith("vqa_result_"):
        value = metrics.get("agg_metrics", metrics.get("overall", metrics.get("accuracy")))
        summary["VQA Acc"] = normalize_metric("VQA Acc", value, normalize)
    elif name.startswith("caption_result_"):
        summary["COCO Caption CIDEr"] = normalize_metric(
            "COCO Caption CIDEr", metrics.get("CIDEr"), normalize
        )
        summary["COCO Caption SPICE"] = normalize_metric(
            "COCO Caption SPICE", metrics.get("SPICE"), normalize
        )
    elif name.startswith("count_result_") and "eval_coco_count" in name:
        summary["COCO-OC Acc"] = normalize_metric("COCO-OC Acc", metrics.get("accuracy"), normalize)
    elif name.startswith("count_result_") and "eval_vcr_count" in name:
        summary["VCR-OC Acc"] = normalize_metric("VCR-OC Acc", metrics.get("accuracy"), normalize)
    elif name.startswith("multiclass_result_") and "eval_coco_multiclass" in name:
        summary["COCO-MCI Acc"] = normalize_metric("COCO-MCI Acc", metrics.get("accuracy"), normalize)
    elif name.startswith("multiclass_result_") and "eval_vcr_multiclass" in name:
        summary["VCR-MCI Acc"] = normalize_metric("VCR-MCI Acc", metrics.get("accuracy"), normalize)


def format_value(value):
    if value is None:
        return "N/A"
    return f"{float(value):.2f}"


def markdown_table(summary):
    headers = SUMMARY_KEYS + ["avg_without_spice", "avg_with_spice"]
    values = [summary.get(key) for key in SUMMARY_KEYS]
    without_spice_keys = [
        "VQA Acc",
        "COCO Caption CIDEr",
        "COCO-OC Acc",
        "COCO-MCI Acc",
        "VCR-OC Acc",
        "VCR-MCI Acc",
    ]
    with_spice_keys = without_spice_keys[:2] + ["COCO Caption SPICE"] + without_spice_keys[2:]

    summary["avg_without_spice"] = average_present(summary.get(key) for key in without_spice_keys)
    summary["avg_with_spice"] = average_present(summary.get(key) for key in with_spice_keys)
    values.extend([summary["avg_without_spice"], summary["avg_with_spice"]])

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(format_value(value) for value in values) + " |",
    ]
    return "\n".join(lines) + "\n"


def average_present(values):
    present = [float(value) for value in values if value is not None]
    return mean(present) if present else None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("pred_results"))
    parser.add_argument("--output-json", type=Path, default=Path("summary_results.json"))
    parser.add_argument("--output-md", type=Path, default=Path("summary_results.md"))
    parser.add_argument("--no-normalize", action="store_true", help="Keep raw metric scales from evaluator outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = {key: None for key in SUMMARY_KEYS}
    metric_files = sorted(args.result_dir.rglob("*_metrics.json"))

    for path in metric_files:
        update_summary(summary, path, read_json(path), normalize=not args.no_normalize)

    md = markdown_table(summary)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    args.output_md.write_text(md, encoding="utf-8")

    print(md)
    print(f"saved json to {args.output_json}")
    print(f"saved markdown to {args.output_md}")


if __name__ == "__main__":
    main()
