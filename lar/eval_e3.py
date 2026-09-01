#!/usr/bin/env python
"""E3: full-table, family-balanced, and shortlist-regret protocols."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

try:
    from .data import WORKSPACE
    from .eval_common import finite_float, read_csv, rho, write_json
except ImportError:  # Direct execution
    from data import WORKSPACE
    from eval_common import finite_float, read_csv, rho, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=WORKSPACE / "lar" / "results" / "lar_metrics.csv")
    parser.add_argument("--targets", type=Path, required=True, help="CSV with name,family,target and baselines")
    parser.add_argument("--target", default="MLLM_Avg")
    parser.add_argument("--pc1-columns", nargs="+", default=None, help="Use standardized PC1 of these target columns")
    parser.add_argument("--lar-metric", default="LAR_64")
    parser.add_argument("--baselines", nargs="*", default=("probe_epoch1", "retrieval-IN", "CKA", "pretrain loss", "A score", "RankMe", "eff_rank"))
    parser.add_argument("--lower-is-better", nargs="*", default=("pretrain loss",))
    parser.add_argument("--probe-column", default="probe_epoch1")
    parser.add_argument("--text-domain", default="answer")
    parser.add_argument("--image-set", default="coco4618")
    parser.add_argument("--family-repeats", type=int, default=5000)
    parser.add_argument("--regret-repeats", type=int, default=20000)
    parser.add_argument("--shortlist-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "lar" / "results" / "e3.json")
    return parser.parse_args()


def family_sample_rho(values: np.ndarray, target: np.ndarray, families: list[str], repeats: int, rng) -> float:
    indices_by_family: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(families):
        indices_by_family[family].append(index)
    coefficients = []
    for _ in range(repeats):
        selected = [rng.choice(indices) for indices in indices_by_family.values()]
        coefficient, _ = rho(values[selected].tolist(), target[selected].tolist())
        if math.isfinite(coefficient):
            coefficients.append(coefficient)
    return float(np.mean(coefficients)) if coefficients else math.nan


def mean_regret(values: np.ndarray, target: np.ndarray, size: int, repeats: int, rng) -> float:
    if len(values) < size:
        return math.nan
    regrets = np.empty(repeats, dtype=np.float64)
    for iteration in range(repeats):
        selected = rng.choice(len(values), size=size, replace=False)
        picked = selected[int(np.argmax(values[selected]))]
        regrets[iteration] = float(target[selected].max() - target[picked])
    return float(regrets.mean())


def combined_regret(probe: np.ndarray, lar: np.ndarray, target: np.ndarray, size: int, repeats: int, rng) -> float:
    if len(target) <= size + 1:
        return math.nan
    features = np.column_stack((probe, lar))
    regrets = np.empty(repeats, dtype=np.float64)
    all_indices = np.arange(len(target))
    for iteration in range(repeats):
        holdout = rng.choice(len(target), size=size, replace=False)
        train_mask = np.ones(len(target), dtype=bool)
        train_mask[holdout] = False
        model = LinearRegression().fit(features[train_mask], target[train_mask])
        predictions = model.predict(features[holdout])
        picked = holdout[int(np.argmax(predictions))]
        regrets[iteration] = float(target[holdout].max() - target[picked])
    return float(regrets.mean())


def main() -> None:
    args = parse_args()
    lar_rows = {
        row["name"]: row for row in read_csv(args.metrics)
        if row.get("text_domain") == args.text_domain and row.get("image_set") == args.image_set
    }
    rows = []
    for row in read_csv(args.targets):
        lar_row = lar_rows.get(row.get("name", ""))
        lar = None if lar_row is None else finite_float(lar_row.get(args.lar_metric))
        family = row.get("family", "").strip()
        if lar is None or not family:
            continue
        enriched = dict(row)
        enriched[args.lar_metric] = str(lar)
        # Spectral baselines naturally live in lar_metrics.csv.
        for column in ("RankMe", "eff_rank"):
            if lar_row.get(column):
                enriched[column] = lar_row[column]
        rows.append(enriched)

    if args.pc1_columns:
        valid_rows = [row for row in rows if all(finite_float(row.get(col)) is not None for col in args.pc1_columns)]
        matrix = np.asarray([[float(row[col]) for col in args.pc1_columns] for row in valid_rows])
        matrix = (matrix - matrix.mean(0)) / (matrix.std(0) + 1e-8)
        target = PCA(n_components=1).fit_transform(matrix).ravel()
        # Fix PCA's arbitrary sign so larger average MLLM score means larger PC1.
        if np.corrcoef(target, matrix.mean(1))[0, 1] < 0:
            target *= -1
        rows = valid_rows
        target_name = "PC1(" + ",".join(args.pc1_columns) + ")"
    else:
        valid = [(row, finite_float(row.get(args.target))) for row in rows]
        rows = [row for row, value in valid if value is not None]
        target = np.asarray([value for _row, value in valid if value is not None], dtype=np.float64)
        target_name = args.target

    families = [row["family"] for row in rows]
    candidate_columns = [args.lar_metric, *args.baselines]
    results = {}
    for column_index, column in enumerate(candidate_columns):
        usable = [(index, finite_float(row.get(column))) for index, row in enumerate(rows)]
        usable = [(index, value) for index, value in usable if value is not None]
        if len(usable) < 2:
            continue
        indices = np.asarray([item[0] for item in usable], dtype=int)
        values = np.asarray([item[1] for item in usable], dtype=np.float64)
        local_target = target[indices]
        local_families = [families[index] for index in indices]
        coefficient, pvalue = rho(values.tolist(), local_target.tolist())
        selection_values = -values if column in args.lower_is_better else values
        rng = np.random.default_rng(args.seed + 1009 * (column_index + 1))
        results[column] = {
            "n": len(indices), "full_spearman_rho": coefficient, "full_pvalue": pvalue,
            "one_per_family_spearman_mean": family_sample_rho(
                values, local_target, local_families, args.family_repeats, rng
            ),
            "top1_regret_k5_mean": mean_regret(
                selection_values, local_target, args.shortlist_size, args.regret_repeats, rng
            ),
            "selection_direction": "min" if column in args.lower_is_better else "max",
        }

    combo_rows = []
    for index, row in enumerate(rows):
        probe = finite_float(row.get(args.probe_column))
        lar = finite_float(row.get(args.lar_metric))
        if probe is not None and lar is not None:
            combo_rows.append((index, probe, lar))
    if combo_rows:
        indices = np.asarray([item[0] for item in combo_rows], dtype=int)
        rng = np.random.default_rng(args.seed + 99991)
        results[f"{args.probe_column}+{args.lar_metric}"] = {
            "n": len(indices),
            "top1_regret_k5_mean": combined_regret(
                np.asarray([item[1] for item in combo_rows]),
                np.asarray([item[2] for item in combo_rows]),
                target[indices], args.shortlist_size, args.regret_repeats, rng,
            ),
            "fit_protocol": "for every sampled shortlist, OLS is fit on all non-shortlisted models",
        }
    payload = {
        "target": target_name, "seed": args.seed,
        "family_repeats": args.family_repeats, "regret_repeats": args.regret_repeats,
        "shortlist_size": args.shortlist_size, "results": results,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    write_json(args.output, rendered)
    print(rendered)


if __name__ == "__main__":
    main()
