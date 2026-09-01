#!/usr/bin/env python
"""Build the E1 BN-gain table from existing paired probing outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

try:
    from .data import WORKSPACE
except ImportError:
    from data import WORKSPACE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-config", type=Path, default=WORKSPACE / "lar" / "configs" / "models.yaml")
    parser.add_argument("--with-bn-root", type=Path, default=WORKSPACE / "outputs" / "vae_linear_probing_dinov2_single_paperlr_bn")
    parser.add_argument("--without-bn-root", type=Path, default=WORKSPACE / "outputs" / "vae_linear_probing_dinov2_single_paperlr")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "lar" / "configs" / "e1_probes.csv")
    return parser.parse_args()


def accuracy(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return 100.0 * float(payload["best_classifier"]["accuracy"])


def main() -> None:
    args = parse_args()
    with args.models_config.open("r", encoding="utf-8") as handle:
        specs = yaml.safe_load(handle)["models"]
    rows = []
    for spec in specs:
        if not spec.get("enabled", False):
            continue
        output_name = spec.get("probing_output_name", spec["name"])
        with_bn = args.with_bn_root / output_name / "results_eval_linear.json"
        without_bn = args.without_bn_root / output_name / "results_eval_linear.json"
        if not with_bn.is_file() or not without_bn.is_file():
            raise FileNotFoundError(f"Missing paired E1 probe for {spec['name']}: {with_bn}, {without_bn}")
        rows.append({
            "name": spec["name"],
            "probe_with_BN": accuracy(with_bn),
            "probe_without_BN": accuracy(without_bn),
            "representation_matches_patch_mean": bool(
                spec.get("probe_representation_matches_patch_mean", False)
            ),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "name", "probe_with_BN", "probe_without_BN",
                "representation_matches_patch_mean",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
