#!/usr/bin/env python
"""Compute LAR v2 and spectral baselines from aligned feature arrays."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

try:
    from .data import WORKSPACE
except ImportError:  # Direct execution: python lar/compute_lar.py
    from data import WORKSPACE


LAR_MS = (8, 16, 32, 64, 128)
CSV_FIELDS = (
    "name", "text_domain", "image_set", "d", "n_tokens", "N", "K",
    "lam_top128", "r_top128", "LAR_8", "LAR_16", "LAR_32", "LAR_64",
    "LAR_128", "Lift_8", "Lift_16", "Lift_32", "Lift_64", "Lift_128",
    "m50", "m90", "log2_m50", "log2_m90", "VSA", "eff_rank", "RankMe",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--text-domain", choices=("caption", "answer", "imagenet"), required=True)
    parser.add_argument("--image-set", choices=("coco4618", "coco5000", "in1k10k"), required=True)
    parser.add_argument(
        "--output", type=Path,
        default=WORKSPACE / "lar" / "results" / "lar_metrics_v2.csv",
    )
    parser.add_argument("--spectrum-root", type=Path, default=WORKSPACE / "lar" / "results" / "spectra")
    return parser.parse_args()


def _standardize(array: np.ndarray) -> np.ndarray:
    centered = array - array.mean(axis=0, keepdims=True)
    return centered / (centered.std(axis=0, keepdims=True) + 1e-8)


def compute_spectral_metrics(
    lam: np.ndarray, r: np.ndarray, k: int | None = None
) -> dict[str, float]:
    """Compute v2 metrics from variance and language-usefulness spectra.

    ``lam`` must already be sorted in descending order.  Component counts are
    one-based: m50=1 means that the first principal direction reaches 50%.
    """
    lam = np.asarray(lam, dtype=np.float64).reshape(-1)
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    if len(lam) != len(r) or not len(lam):
        raise ValueError(f"Expected equally sized non-empty spectra, got {len(lam)} and {len(r)}")
    if not np.isfinite(lam).all() or not np.isfinite(r).all():
        raise ValueError("Spectra must be finite")
    if np.any(lam < 0) or np.any(r < 0):
        raise ValueError("lam and bias-corrected r must be non-negative")

    requested_k = min(len(lam), 512) if k is None else int(k)
    if requested_k < 1 or requested_k > min(len(lam), len(r), 512):
        raise ValueError(
            f"K={requested_k} is outside the available spectrum length {min(len(lam), len(r))}"
        )
    lam_k = lam[:requested_k]
    r_k = r[:requested_k]
    w_k = lam_k * r_k
    lam_sum = float(lam_k.sum())
    w_sum = float(w_k.sum())

    metrics: dict[str, float] = {"K": float(requested_k)}
    for m in LAR_MS:
        stop = min(m, requested_k)
        lar_m = float(w_k[:stop].sum() / w_sum) if w_sum > 0 else math.nan
        lambda_m = float(lam_k[:stop].sum() / lam_sum) if lam_sum > 0 else math.nan
        metrics[f"LAR_{m}"] = lar_m
        metrics[f"Lift_{m}"] = (
            float(lar_m / lambda_m)
            if math.isfinite(lar_m) and math.isfinite(lambda_m) and lambda_m > 0
            else math.nan
        )

    if w_sum > 0:
        cumulative = np.cumsum(w_k) / w_sum
        m50 = int(np.searchsorted(cumulative, 0.5, side="left") + 1)
        m90 = int(np.searchsorted(cumulative, 0.9, side="left") + 1)
        m50 = min(m50, requested_k)
        m90 = min(m90, requested_k)
    else:
        m50 = requested_k
        m90 = requested_k
    metrics.update(
        m50=float(m50),
        m90=float(m90),
        log2_m50=float(math.log2(m50)),
        log2_m90=float(math.log2(m90)),
    )

    # A flat usefulness spectrum has no monotone alignment; define its VSA as
    # the neutral value zero instead of scipy's undefined correlation.
    if requested_k < 2 or np.ptp(r_k) == 0:
        metrics["VSA"] = 0.0
    else:
        vsa_result = spearmanr(np.arange(requested_k), r_k)
        metrics["VSA"] = -float(vsa_result.statistic)

    if lam_sum > 0:
        eigen_p = lam_k / lam_sum
        metrics["eff_rank"] = float(1.0 / np.sum(eigen_p**2))
        singular_k = np.sqrt(np.maximum(lam_k, 0.0))
        singular_sum = float(singular_k.sum())
        if singular_sum > 0:
            singular_p = singular_k / singular_sum
            positive = singular_p > 0
            metrics["RankMe"] = float(
                np.exp(-np.sum(singular_p[positive] * np.log(singular_p[positive])))
            )
        else:
            metrics["RankMe"] = math.nan
    else:
        metrics["eff_rank"] = math.nan
        metrics["RankMe"] = math.nan
    return metrics


def compute_visual_spectrum(visual: np.ndarray) -> dict[str, np.ndarray]:
    """Center FP32 image rows and compute their spectrum in FP64."""
    if visual.ndim != 2 or visual.shape[0] < 3 or not np.isfinite(visual).all():
        raise ValueError(f"Expected at least three finite image rows, got Z={visual.shape}")
    if visual.dtype != np.float32:
        raise ValueError(f"Stored visual features must be float32, got {visual.dtype}")
    n = visual.shape[0]
    z = visual.astype(np.float64, copy=False)
    z = z - z.mean(axis=0, keepdims=True)
    u, singular, _vt = np.linalg.svd(z, full_matrices=False)
    lam = singular**2 / (n - 1)
    scores = u * singular
    return {"lam": lam, "scores": scores, "singular": singular}


def compute_language_usefulness(
    scores: np.ndarray,
    text: np.ndarray,
    *,
    scores_standardized: bool = False,
    text_standardized: bool = False,
) -> np.ndarray:
    """Compute the bias-corrected language usefulness for aligned image rows."""
    if scores.ndim != 2 or text.ndim != 2 or scores.shape[0] != text.shape[0]:
        raise ValueError(f"Expected aligned 2D arrays, got scores={scores.shape}, E={text.shape}")
    if scores.shape[0] < 3 or not np.isfinite(scores).all() or not np.isfinite(text).all():
        raise ValueError("Features must contain at least three finite rows")
    n = scores.shape[0]
    e_raw = text.astype(np.float64, copy=False)
    scores_raw = scores.astype(np.float64, copy=False)
    e = e_raw if text_standardized else _standardize(e_raw)
    zs = scores_raw if scores_standardized else _standardize(scores_raw)
    correlations = (zs.T @ e) / n
    return np.maximum(np.mean(correlations**2, axis=1) - 1.0 / n, 0.0)


def compute_metrics(visual: np.ndarray, text: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    if visual.ndim != 2 or text.ndim != 2 or visual.shape[0] != text.shape[0]:
        raise ValueError(f"Expected aligned 2D arrays, got Z={visual.shape}, E={text.shape}")
    if visual.shape[0] < 3 or not np.isfinite(visual).all() or not np.isfinite(text).all():
        raise ValueError("Features must contain at least three finite rows")

    spectrum = compute_visual_spectrum(visual)
    lam = spectrum["lam"]
    singular = spectrum["singular"]
    r = compute_language_usefulness(spectrum["scores"], text)
    w = lam * r
    k = min(len(lam), 512)
    metrics = compute_spectral_metrics(lam, r, k)
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
        **metrics,
    }
    _upsert_csv(args.output, row)
    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
