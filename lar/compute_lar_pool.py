#!/usr/bin/env python
"""Compute both COCO text domains for a configured encoder pool.

Each visual feature array is decomposed once.  Caption and answer text arrays
are loaded once for the whole run and reuse the same visual SVD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .compute_lar import (
        _load_ids,
        _load_meta,
        _serial,
        _standardize,
        _upsert_csv,
        compute_language_usefulness,
        compute_spectral_metrics,
        compute_visual_spectrum,
    )
    from .data import WORKSPACE
    from .extract_visual import load_specs
except ImportError:  # Direct execution
    from compute_lar import (
        _load_ids,
        _load_meta,
        _serial,
        _standardize,
        _upsert_csv,
        compute_language_usefulness,
        compute_spectral_metrics,
        compute_visual_spectrum,
    )
    from data import WORKSPACE
    from extract_visual import load_specs


DOMAINS = ("caption", "answer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-config", type=Path,
        default=WORKSPACE / "lar" / "configs" / "models_e3.yaml",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--image-set", default="coco4618", choices=("coco4618",))
    parser.add_argument("--feature-root", type=Path, default=WORKSPACE / "lar" / "features")
    parser.add_argument("--text-root", type=Path, default=WORKSPACE / "lar" / "text")
    parser.add_argument(
        "--output", type=Path,
        default=WORKSPACE / "lar" / "results" / "lar_metrics_v2.csv",
    )
    parser.add_argument(
        "--spectrum-root", type=Path,
        default=WORKSPACE / "lar" / "results" / "spectra",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip a model only when both CSV rows and both full spectrum sidecars exist.",
    )
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Report missing visual feature arrays and continue instead of failing preflight.",
    )
    return parser.parse_args()


def _existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.is_file():
        return set()
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            (row.get("name", ""), row.get("text_domain", ""), row.get("image_set", ""))
            for row in csv.DictReader(handle)
        }


def main() -> None:
    args = parse_args()
    specs = load_specs(args.models_config, args.models)
    if not specs:
        raise RuntimeError(f"No enabled models in {args.models_config}")

    text_arrays: dict[str, np.ndarray] = {}
    text_ids: dict[str, list[str]] = {}
    for domain in DOMAINS:
        path = args.text_root / f"{domain}__{args.image_set}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing shared {domain} text embedding: {path}")
        ids = _load_ids(path)
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or len(ids) != array.shape[0]:
            raise RuntimeError(f"Invalid shared text array {path}: shape={array.shape}, ids={len(ids)}")
        text_arrays[domain] = _standardize(array.astype(np.float64, copy=False))
        text_ids[domain] = ids
    if text_ids["caption"] != text_ids["answer"]:
        raise RuntimeError("Caption and answer text arrays do not use the same image rows")

    feature_paths = {
        spec["name"]: args.feature_root / f"{spec['name']}__{args.image_set}.npy"
        for spec in specs
    }
    missing = [str(path) for path in feature_paths.values() if not path.is_file()]
    if missing and not args.allow_missing:
        preview = "\n".join(missing[:20])
        extra = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(f"Missing {len(missing)} visual feature arrays:\n{preview}{extra}")
    if missing:
        print(f"WARNING: skipping {len(missing)} missing visual arrays", flush=True)

    args.spectrum_root.mkdir(parents=True, exist_ok=True)
    existing = _existing_keys(args.output)
    completed = 0
    for model_index, spec in enumerate(specs, start=1):
        name = spec["name"]
        feature_path = feature_paths[name]
        if not feature_path.is_file():
            print(f"SKIP missing {feature_path}", flush=True)
            continue
        pending_domains = []
        for domain in DOMAINS:
            key = (name, domain, args.image_set)
            sidecar = args.spectrum_root / f"{name}__{domain}__{args.image_set}.npz"
            if not args.resume or key not in existing or not sidecar.is_file():
                pending_domains.append(domain)
        if not pending_domains:
            print(f"[{model_index}/{len(specs)}] SKIP complete {name}", flush=True)
            continue

        visual_ids = _load_ids(feature_path)
        if visual_ids != text_ids["caption"]:
            mismatch = next(
                (
                    index for index, pair in enumerate(zip(visual_ids, text_ids["caption"]))
                    if pair[0] != pair[1]
                ),
                min(len(visual_ids), len(text_ids["caption"])),
            )
            raise RuntimeError(f"{name}: visual/text ID ordering differs at row {mismatch}")
        visual = np.load(feature_path, mmap_mode="r")
        if len(visual_ids) != visual.shape[0]:
            raise RuntimeError(f"{name}: ID sidecar length does not match visual rows")
        print(f"[{model_index}/{len(specs)}] SVD {name} shape={visual.shape}", flush=True)
        spectrum = compute_visual_spectrum(visual)
        lam = spectrum["lam"]
        standardized_scores = _standardize(spectrum["scores"])
        k = min(visual.shape[1], 512)
        metadata = _load_meta(feature_path)

        for domain in pending_domains:
            print(f"[{model_index}/{len(specs)}] r({domain}) {name}", flush=True)
            r = compute_language_usefulness(
                standardized_scores,
                text_arrays[domain],
                scores_standardized=True,
                text_standardized=True,
            )
            metrics = compute_spectral_metrics(lam, r, k)
            sidecar = args.spectrum_root / f"{name}__{domain}__{args.image_set}.npz"
            np.savez_compressed(
                sidecar,
                lam=lam,
                r=r,
                w=lam * r,
                singular=spectrum["singular"],
            )
            row: dict[str, object] = {
                "name": name,
                "text_domain": domain,
                "image_set": args.image_set,
                "d": visual.shape[1],
                "n_tokens": metadata.get("n_tokens", ""),
                "N": visual.shape[0],
                "K": int(metrics.pop("K")),
                "lam_top128": _serial(lam),
                "r_top128": _serial(r),
                **metrics,
            }
            _upsert_csv(args.output, row)
            existing.add((name, domain, args.image_set))
        completed += 1
        print(json.dumps({"completed_model": name, "domains": pending_domains}), flush=True)

    print(f"completed {completed} model SVDs; metrics={args.output}")


if __name__ == "__main__":
    main()
