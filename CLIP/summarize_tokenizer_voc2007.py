#!/usr/bin/env python
"""Summarize tokenizer VOC2007 11-point mAP results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


PROTOCOL = "voc2007-kornblith-lbfgs-v1"
MODEL_ORDER = ("unitok", "vilau", "metaclip", "toklips", "toklipl")
VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
SUMMARY_FIELDS = (
    "model",
    "status",
    "feature_dim",
    "selected_lambda",
    "selected_C",
    "validation_mAP_11point",
    "test_mAP_11point",
    "selection_converged",
    "final_converged",
)
PER_CLASS_FIELDS = (
    "model",
    "class",
    "validation_AP_11point",
    "test_AP_11point",
)


def _load_model_result(output_root: Path, model: str) -> dict | None:
    path = output_root / model / "results.json"
    if not path.is_file():
        return None
    with path.open() as handle:
        payload = json.load(handle)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError(
            f"Unexpected protocol in {path}: {payload.get('protocol')!r}; expected {PROTOCOL!r}"
        )
    if payload.get("model") != model:
        raise ValueError(f"Result model mismatch in {path}: {payload.get('model')!r}")
    return payload


def collect_rows(output_root: Path, models: list[str]) -> tuple[list[dict], list[dict]]:
    summary_rows = []
    per_class_rows = []
    for model in models:
        payload = _load_model_result(output_root, model)
        if payload is None:
            summary_rows.append(
                {
                    "model": model,
                    "status": "missing",
                    "feature_dim": None,
                    "selected_lambda": None,
                    "selected_C": None,
                    "validation_mAP_11point": None,
                    "test_mAP_11point": None,
                    "selection_converged": None,
                    "final_converged": None,
                }
            )
            continue
        selection = payload["selection"]
        final = payload["final_evaluation"]
        selection_converged = len(selection.get("nonconverged_classes", [])) == 0
        final_converged = bool(final["converged"])
        status = "complete" if selection_converged and final_converged else "nonconverged"
        summary_rows.append(
            {
                "model": model,
                "status": status,
                "feature_dim": int(payload["feature_dim"]),
                "selected_lambda": float(selection["selected_lambda"]),
                "selected_C": float(selection["selected_C"]),
                "validation_mAP_11point": float(selection["validation_mAP_11point"]),
                "test_mAP_11point": float(final["mAP_11point"]),
                "selection_converged": selection_converged,
                "final_converged": final_converged,
            }
        )
        validation_ap = selection["validation_AP_11point"]
        test_ap = final["AP_11point"]
        if set(validation_ap) != set(VOC_CLASSES) or set(test_ap) != set(VOC_CLASSES):
            raise ValueError(f"Result for {model} does not contain exactly the 20 VOC classes")
        for class_name in VOC_CLASSES:
            per_class_rows.append(
                {
                    "model": model,
                    "class": class_name,
                    "validation_AP_11point": float(validation_ap[class_name]),
                    "test_AP_11point": float(test_ap[class_name]),
                }
            )
    return summary_rows, per_class_rows


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _format_number(value, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _summary_markdown(rows: list[dict]) -> str:
    lines = [
        "# PASCAL VOC 2007 multi-label linear probing",
        "",
        "Official VOC2007 11-point mAP (%). L2 lambda is selected on val; the final head is refit on train+val and evaluated on test.",
        "",
        "| model | feature dim | selected lambda | val mAP | test mAP | convergence |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["status"] == "missing":
            lines.append(f"| {row['model']} | — | — | — | — | missing |")
            continue
        convergence = "ok" if row["status"] == "complete" else "warning"
        lines.append(
            f"| {row['model']} | {row['feature_dim']} | {row['selected_lambda']:.12g} | "
            f"{_format_number(row['validation_mAP_11point'])} | "
            f"{_format_number(row['test_mAP_11point'])} | {convergence} |"
        )
    return "\n".join(lines) + "\n"


def _per_class_markdown(rows: list[dict], models: list[str]) -> str:
    by_key = {(row["model"], row["class"]): row for row in rows}
    lines = [
        "# PASCAL VOC 2007 per-class test AP",
        "",
        "All values are official VOC2007 11-point AP (%).",
        "",
        "| class | " + " | ".join(models) + " |",
        "|---|" + "---:|" * len(models),
    ]
    for class_name in VOC_CLASSES:
        values = []
        for model in models:
            row = by_key.get((model, class_name))
            values.append("—" if row is None else f"{row['test_AP_11point']:.2f}")
        lines.append(f"| {class_name} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _write_text_atomic(path: Path, content: str):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    args = parser.parse_args(argv)
    args.models = list(dict.fromkeys(args.models))
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary_rows, per_class_rows = collect_rows(args.output_root, args.models)
    _write_csv(args.output_root / "summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(args.output_root / "per_class_ap.csv", PER_CLASS_FIELDS, per_class_rows)
    summary_markdown = _summary_markdown(summary_rows)
    _write_text_atomic(args.output_root / "summary.md", summary_markdown)
    _write_text_atomic(
        args.output_root / "per_class_ap.md",
        _per_class_markdown(per_class_rows, args.models),
    )
    print(summary_markdown)


if __name__ == "__main__":
    main()
