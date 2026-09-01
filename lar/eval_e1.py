#!/usr/bin/env python
"""E1: correlate ImageNet Waste with BatchNorm linear-probe gain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .data import WORKSPACE
    from .eval_common import finite_float, read_csv, rho, write_json
except ImportError:  # Direct execution
    from data import WORKSPACE
    from eval_common import finite_float, read_csv, rho, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=WORKSPACE / "lar" / "results" / "lar_metrics.csv")
    parser.add_argument("--probes", type=Path, required=True, help="CSV: name,probe_with_BN,probe_without_BN")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "lar" / "results" / "e1.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    waste = {
        row["name"]: finite_float(row.get("Waste"))
        for row in read_csv(args.metrics)
        if row.get("text_domain") == "imagenet" and row.get("image_set") == "in1k10k"
    }
    records = []
    for row in read_csv(args.probes):
        with_bn = finite_float(row.get("probe_with_BN"))
        without_bn = finite_float(row.get("probe_without_BN"))
        model_waste = waste.get(row.get("name", ""))
        if with_bn is None or without_bn is None or model_waste is None:
            continue
        records.append({
            "name": row["name"], "Waste": model_waste,
            "probe_with_BN": with_bn, "probe_without_BN": without_bn,
            "BN_gain": with_bn - without_bn,
            "representation_matches_patch_mean": (
                row.get("representation_matches_patch_mean", "").lower() == "true"
            ),
        })
    coefficient, pvalue = rho(
        [row["Waste"] for row in records], [row["BN_gain"] for row in records]
    )
    payload = {
        "n": len(records), "spearman_rho": coefficient, "pvalue": pvalue,
        "passes_rho_gt_0.7": coefficient > 0.7, "records": records,
        "strict_representation_match_n": sum(
            bool(row["representation_matches_patch_mean"]) for row in records
        ),
        "warning": (
            "Rows with representation_matches_patch_mean=false are exploratory: "
            "their BN gain was measured on a CLS/pooled probing representation."
        ),
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    write_json(args.output, rendered)
    print(rendered)


if __name__ == "__main__":
    main()
