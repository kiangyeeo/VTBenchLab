#!/usr/bin/env python
"""E2: quantify between-family separation and within-family utility."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .data import WORKSPACE
    from .eval_common import finite_float, read_csv, rho, write_json
except ImportError:  # Direct execution
    from data import WORKSPACE
    from eval_common import finite_float, read_csv, rho, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=WORKSPACE / "lar" / "results" / "lar_metrics.csv")
    parser.add_argument("--targets", type=Path, required=True, help="CSV containing name,family and target")
    parser.add_argument("--metric", default="LAR_64")
    parser.add_argument("--target", default="MLLM_Avg")
    parser.add_argument("--text-domain", default="answer")
    parser.add_argument("--image-set", default="coco4618")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "lar" / "results" / "e2.json")
    return parser.parse_args()


def anova_f(values_by_family: dict[str, list[float]]) -> float:
    groups = [np.asarray(values, dtype=np.float64) for values in values_by_family.values() if values]
    n = sum(len(group) for group in groups)
    g = len(groups)
    if g < 2 or n <= g:
        return math.nan
    grand = np.concatenate(groups).mean()
    ss_between = sum(len(group) * float((group.mean() - grand) ** 2) for group in groups)
    ss_within = sum(float(np.sum((group - group.mean()) ** 2)) for group in groups)
    ms_between = ss_between / (g - 1)
    ms_within = ss_within / (n - g)
    return ms_between / ms_within if ms_within > 0 else math.inf


def main() -> None:
    args = parse_args()
    metric_rows = {
        row["name"]: row for row in read_csv(args.metrics)
        if row.get("text_domain") == args.text_domain and row.get("image_set") == args.image_set
    }
    merged = []
    for target_row in read_csv(args.targets):
        metric_row = metric_rows.get(target_row.get("name", ""))
        metric = None if metric_row is None else finite_float(metric_row.get(args.metric))
        target = finite_float(target_row.get(args.target))
        family = target_row.get("family", "").strip()
        if metric is None or target is None or not family:
            continue
        merged.append((target_row["name"], family, metric, target))

    by_family: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for name, family, metric, target in merged:
        by_family[family].append((name, metric, target))
    within = {}
    for family, rows in sorted(by_family.items()):
        coefficient, pvalue = rho([row[1] for row in rows], [row[2] for row in rows])
        within[family] = {"n": len(rows), "spearman_rho": coefficient, "pvalue": pvalue}
    payload = {
        "metric": args.metric, "target": args.target, "n": len(merged),
        "n_families": len(by_family),
        "F_between_within": anova_f({key: [row[1] for row in rows] for key, rows in by_family.items()}),
        "F_definition": "one-way ANOVA MS_between / MS_within on the selected LAR metric",
        "within_family": within,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    write_json(args.output, rendered)
    print(rendered)


if __name__ == "__main__":
    main()
