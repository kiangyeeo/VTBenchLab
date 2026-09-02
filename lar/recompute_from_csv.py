#!/usr/bin/env python
"""Recompute LAR v2 metrics without rerunning visual or text encoders."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

try:
    from .compute_lar import CSV_FIELDS, LAR_MS, compute_spectral_metrics
    from .data import WORKSPACE
except ImportError:  # Direct execution
    from compute_lar import CSV_FIELDS, LAR_MS, compute_spectral_metrics
    from data import WORKSPACE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=WORKSPACE / "lar" / "results" / "lar_metrics.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=WORKSPACE / "lar" / "results" / "lar_metrics_v2.csv",
    )
    parser.add_argument(
        "--allow-truncated-k", action="store_true",
        help="Use only serialized top-128 values when the complete spectrum sidecar is unavailable.",
    )
    return parser.parse_args()


def parse_spectrum(value: str) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split(";") if item.strip()], dtype=np.float64)
    return values[np.isfinite(values)]


def load_spectra(row: dict[str, str], input_path: Path, allow_truncated: bool) -> tuple[np.ndarray, np.ndarray, int]:
    requested_k = min(int(float(row["d"])), 512)
    candidates = []
    if row.get("spectrum_path"):
        path = Path(row["spectrum_path"])
        candidates.extend((path, input_path.parent / path, WORKSPACE / path))
    candidates.append(
        input_path.parent / "spectra" /
        f"{row['name']}__{row['text_domain']}__{row['image_set']}.npz"
    )
    for path in candidates:
        if path.is_file():
            with np.load(path) as payload:
                lam = np.asarray(payload["lam"], dtype=np.float64)
                r = np.asarray(payload["r"], dtype=np.float64)
            if min(len(lam), len(r)) >= requested_k:
                return lam, r, requested_k

    lam = parse_spectrum(row["lam_top128"])
    r = parse_spectrum(row["r_top128"])
    available = min(len(lam), len(r), requested_k)
    if available < requested_k and not allow_truncated:
        raise RuntimeError(
            f"{row['name']}/{row['text_domain']}: K={requested_k}, but CSV contains only "
            f"{available} finite directions and no complete spectrum sidecar was found"
        )
    return lam[:available], r[:available], available


def recompute_row(
    row: dict[str, str], input_path: Path, allow_truncated: bool
) -> tuple[dict[str, str], float]:
    lam, r, k = load_spectra(row, input_path, allow_truncated)
    metrics = compute_spectral_metrics(lam, r, k)
    differences = []
    for m in LAR_MS:
        old = row.get(f"LAR_{m}", "")
        if old:
            try:
                old_value = float(old)
            except ValueError:
                continue
            new_value = metrics[f"LAR_{m}"]
            if math.isfinite(old_value) and math.isfinite(new_value):
                differences.append(abs(old_value - new_value))
    output = {field: row.get(field, "") for field in CSV_FIELDS}
    output["K"] = str(k)
    for key, value in metrics.items():
        if key != "K":
            output[key] = format(float(value), ".12g")
    return output, max(differences, default=0.0)


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {args.input}")
    recomputed = []
    max_lar_difference = 0.0
    for row in rows:
        result, difference = recompute_row(row, args.input, args.allow_truncated_k)
        recomputed.append(result)
        max_lar_difference = max(max_lar_difference, difference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(recomputed)
    temporary.replace(args.output)
    print(
        f"wrote {len(recomputed)} rows to {args.output}; "
        f"max |recomputed LAR - stored LAR|={max_lar_difference:.3g}"
    )


if __name__ == "__main__":
    main()
