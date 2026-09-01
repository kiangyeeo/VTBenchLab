#!/usr/bin/env python
"""Compute LAR, Waste, and spectral baselines from aligned feature arrays."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from .data import WORKSPACE
except ImportError:  # Direct execution: python lar/compute_lar.py
    from data import WORKSPACE


LAR_MS = (8, 16, 32, 64, 128)
CSV_FIELDS = (
    "name", "text_domain", "image_set", "d", "n_tokens", "N", "K",
    "lam_top128", "r_top128", "LAR_8", "LAR_16", "LAR_32", "LAR_64",
    "LAR_128", "LAR_frac05", "Waste", "eff_rank", "RankMe",
    "answer_count_mean", "spectrum_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--text-domain", choices=("caption", "answer", "imagenet"), required=True)
    parser.add_argument("--image-set", choices=("coco4618", "coco5000", "in1k10k"), required=True)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "lar" / "results" / "lar_metrics.csv")
    parser.add_argument("--spectrum-root", type=Path, default=WORKSPACE / "lar" / "results" / "spectra")
    return parser.parse_args()


def _standardize(array: np.ndarray) -> np.ndarray:
    centered = array - array.mean(axis=0, keepdims=True)
    return centered / (centered.std(axis=0, keepdims=True) + 1e-8)


def compute_metrics(visual: np.ndarray, text: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    if visual.ndim != 2 or text.ndim != 2 or visual.shape[0] != text.shape[0]:
        raise ValueError(f"Expected aligned 2D arrays, got Z={visual.shape}, E={text.shape}")
    if visual.shape[0] < 3 or not np.isfinite(visual).all() or not np.isfinite(text).all():
        raise ValueError("Features must contain at least three finite rows")

    n, d = visual.shape
    z = visual.astype(np.float64, copy=False)
    z = z - z.mean(axis=0, keepdims=True)
    e = _standardize(text.astype(np.float64, copy=False))
    u, singular, _vt = np.linalg.svd(z, full_matrices=False)
    lam = singular**2 / (n - 1)
    scores = u * singular
    zs = _standardize(scores)
    correlations = (zs.T @ e) / n
    r = np.maximum(np.mean(correlations**2, axis=1) - 1.0 / n, 0.0)
    w = lam * r
    k = min(len(lam), 512)
    lam_k, r_k, w_k = lam[:k], r[:k], w[:k]
    w_sum = float(w_k.sum())

    metrics: dict[str, float] = {}
    for m in LAR_MS:
        metrics[f"LAR_{m}"] = float(w_k[: min(m, k)].sum() / w_sum) if w_sum > 0 else math.nan
    frac_m = max(1, min(k, int(0.05 * d)))
    metrics["LAR_frac05"] = float(w_k[:frac_m].sum() / w_sum) if w_sum > 0 else math.nan

    r_max = float(r_k.max(initial=0.0))
    if r_max > 0 and lam_k.sum() > 0:
        metrics["Waste"] = float((lam_k * (1.0 - r_k / r_max)).sum() / lam_k.sum())
    else:
        metrics["Waste"] = 1.0

    eigen_p = lam_k / lam_k.sum()
    metrics["eff_rank"] = float(1.0 / np.sum(eigen_p**2))
    singular_k = np.sqrt(np.maximum(lam_k, 0.0))
    singular_p = singular_k / singular_k.sum()
    positive = singular_p > 0
    metrics["RankMe"] = float(np.exp(-np.sum(singular_p[positive] * np.log(singular_p[positive]))))
    metrics["K"] = float(k)
    return metrics, {"lam": lam, "r": r, "w": w, "singular": singular}


def _sidecar(path: Path, suffix: str) -> Path:
    return path.with_suffix(suffix)


def _load_ids(array_path: Path) -> list[str]:
    path = _sidecar(array_path, ".ids.txt")
    if not path.is_file():
        raise FileNotFoundError(f"Missing alignment sidecar: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def _load_meta(array_path: Path) -> dict:
    path = _sidecar(array_path, ".meta.json")
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _serial(values: np.ndarray, length: int = 128) -> str:
    padded = np.full(length, np.nan, dtype=np.float64)
    count = min(length, len(values))
    padded[:count] = values[:count]
    return ";".join(format(float(value), ".12g") for value in padded)


def _upsert_csv(path: Path, row: dict[str, object]) -> None:
    rows: list[dict[str, str]] = []
    key = (str(row["name"]), str(row["text_domain"]), str(row["image_set"]))
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows = [
            old for old in rows
            if (old.get("name"), old.get("text_domain"), old.get("image_set")) != key
        ]
    rows.append({field: str(row.get(field, "")) for field in CSV_FIELDS})
    rows.sort(key=lambda item: (item["name"], item["text_domain"], item["image_set"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    visual_ids = _load_ids(args.feature)
    text_ids = _load_ids(args.text)
    if visual_ids != text_ids:
        mismatch = next(
            (index for index, pair in enumerate(zip(visual_ids, text_ids)) if pair[0] != pair[1]),
            min(len(visual_ids), len(text_ids)),
        )
        raise RuntimeError(f"Visual/text ID ordering differs at row {mismatch}")
    visual = np.load(args.feature, mmap_mode="r")
    text = np.load(args.text, mmap_mode="r")
    if len(visual_ids) != visual.shape[0] or len(text_ids) != text.shape[0]:
        raise RuntimeError("ID sidecar length does not match array rows")

    metrics, spectra = compute_metrics(visual, text)
    feature_meta = _load_meta(args.feature)
    text_meta = _load_meta(args.text)
    args.spectrum_root.mkdir(parents=True, exist_ok=True)
    spectrum_path = args.spectrum_root / f"{args.name}__{args.text_domain}__{args.image_set}.npz"
    np.savez_compressed(spectrum_path, **spectra)
    row: dict[str, object] = {
        "name": args.name,
        "text_domain": args.text_domain,
        "image_set": args.image_set,
        "d": visual.shape[1],
        "n_tokens": feature_meta.get("n_tokens", ""),
        "N": visual.shape[0],
        "K": int(metrics.pop("K")),
        "lam_top128": _serial(spectra["lam"]),
        "r_top128": _serial(spectra["r"]),
        "answer_count_mean": text_meta.get("answer_count_mean", ""),
        "spectrum_path": str(spectrum_path),
        **metrics,
    }
    _upsert_csv(args.output, row)
    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
