#!/usr/bin/env python
"""Compute E4 cross-modal metrics for every available visual encoder/text domain.

The output is long-form and restart-safe.  Missing text domains (notably T6 when
the external train splits are absent) are recorded in the metadata audit and are
never fabricated or silently replaced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

try:
    from .compute_lar import compute_language_usefulness, compute_spectral_metrics, compute_visual_spectrum
    from .data import WORKSPACE
except ImportError:
    from compute_lar import compute_language_usefulness, compute_spectral_metrics, compute_visual_spectrum
    from data import WORKSPACE


DOMAINS = (
    "caption", "answer_other", "question_other", "qa_concat",
    "answer_all_types", "eval_answer",
)
BASE_METRICS = ("VSA", "mutual_kNN_k5", "mutual_kNN_k10", "mutual_kNN_k20", "cm_cka", "cm_r2")
ABTT_METRICS = ("VSA_abtt1", "VSA_abtt2", "VSA_abtt3")
CSV_FIELDS = ("name", "text_domain", "metric_name", "value", "d", "n_tokens", "N", "K")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-config", type=Path, default=WORKSPACE / "lar/configs/models_e3.yaml")
    parser.add_argument("--feature-root", type=Path, default=WORKSPACE / "lar/features")
    parser.add_argument("--text-root", type=Path, default=WORKSPACE / "lar/text")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "lar/results/metrics_v3.csv")
    parser.add_argument("--metadata", type=Path, default=WORKSPACE / "lar/results/metrics_v3_meta.json")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--model", action="append", help="Only compute these model names (repeatable).")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--knn-block-size", type=int, default=512)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-repeats", type=int, default=5)
    parser.add_argument("--cca-pca-rank", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_seed(base: int, *parts: object) -> int:
    digest = hashlib.blake2b("\x1f".join(map(str, parts)).encode(), digest_size=8).digest()
    return (base + int.from_bytes(digest, "little")) % (2**63 - 1)


def read_ids(path: Path) -> list[str]:
    sidecar = path.with_suffix(".ids.txt")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing alignment sidecar: {sidecar}")
    return sidecar.read_text(encoding="utf-8").splitlines()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def model_names(path: Path, selected: list[str] | None) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = [str(row["name"]) for row in payload.get("models", []) if row.get("enabled", True)]
    if selected:
        unknown = sorted(set(selected) - set(names))
        if unknown:
            raise ValueError(f"Unknown --model names: {unknown}")
        names = [name for name in names if name in set(selected)]
    return names


def resolve_text_path(root: Path, domain: str) -> Path:
    direct = root / f"{domain}__coco4618.npy"
    if direct.is_file():
        return direct
    if domain == "answer_other":
        legacy = root / "answer__coco4618.npy"
        if legacy.is_file():
            return legacy
    return direct


def effective_rank_and_rankme(singular: np.ndarray) -> tuple[float, float]:
    eigen = singular.astype(np.float64) ** 2
    if eigen.sum() <= 0:
        return math.nan, math.nan
    eigen_p = eigen / eigen.sum()
    eff_rank = 1.0 / np.sum(eigen_p**2)
    singular_p = singular / singular.sum()
    positive = singular_p > 0
    rankme = np.exp(-np.sum(singular_p[positive] * np.log(singular_p[positive])))
    return float(eff_rank), float(rankme)


def mean_pair_cosine(array: np.ndarray) -> float:
    norms = np.linalg.norm(array.astype(np.float64), axis=1, keepdims=True)
    normalized = array.astype(np.float64) / np.maximum(norms, 1e-12)
    n = len(normalized)
    if n < 2:
        return math.nan
    return float((np.square(normalized.sum(axis=0)).sum() - n) / (n * (n - 1)))


def text_bundle(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r")
    if array.dtype != np.float32 or array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"Text embeddings must be finite FP32 matrices: {path} {array.shape} {array.dtype}")
    ids = read_ids(path)
    if len(ids) != len(array):
        raise RuntimeError(f"Text ID mismatch: {path}")
    centered = np.array(array, dtype=np.float64, copy=True)
    centered -= centered.mean(axis=0, keepdims=True)
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    eff_rank, rankme = effective_rank_and_rankme(singular)
    meta = read_json(path.with_suffix(".meta.json"))
    texts_path = path.with_suffix(".texts.json")
    texts = json.loads(texts_path.read_text(encoding="utf-8")) if texts_path.is_file() else []
    lengths_path = path.with_suffix(".token_lengths.npy")
    lengths = np.load(lengths_path) if lengths_path.is_file() else None
    diag = {
        "N": len(array), "d": int(array.shape[1]), "eff_rank": eff_rank, "RankMe": rankme,
        "mean_pair_cosine": mean_pair_cosine(np.asarray(array)),
        "top1_eig_frac": float(singular[0] ** 2 / np.square(singular).sum()),
        "mean_token_length": None if lengths is None else float(np.mean(lengths)),
        "unique_texts": int(meta.get("unique_texts", len(set(texts)) if texts else 0)),
        "coverage": meta.get("dataset", {}),
    }
    rank = min(args.cca_pca_rank, len(singular), array.shape[1])
    pca_scores = u[:, :rank] * singular[:rank]
    abtt = {}
    if path.name.startswith("caption__"):
        for count in (1, 2, 3):
            abtt[count] = (centered - (u[:, :count] * singular[:count]) @ vt[:count]).astype(np.float32)
    return {
        "array": np.asarray(array), "ids": ids, "centered": centered,
        "singular": singular, "pca_scores": pca_scores, "diag": diag, "abtt": abtt,
    }


def centered_l2_knn(array: np.ndarray, max_k: int, device: str, block_size: int) -> np.ndarray:
    if len(array) <= max_k:
        raise ValueError(f"Need N>{max_k} for kNN, got {len(array)}")
    target_device = torch.device(device)
    x = torch.tensor(array, dtype=torch.float32, device=target_device)
    x = x - x.mean(dim=0, keepdim=True)
    x = torch.nn.functional.normalize(x, dim=1)
    output = torch.empty((len(x), max_k), dtype=torch.int64, device="cpu")
    for start in range(0, len(x), block_size):
        stop = min(start + block_size, len(x))
        similarities = x[start:stop] @ x.T
        local = torch.arange(stop - start, device=target_device)
        similarities[local, torch.arange(start, stop, device=target_device)] = -torch.inf
        output[start:stop] = torch.topk(similarities, max_k, dim=1, sorted=False).indices.cpu()
    del x
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
    return output.numpy().astype(np.int32, copy=False)


def mutual_knn(visual_neighbors: np.ndarray, text_neighbors: np.ndarray, k: int) -> float:
    if visual_neighbors.shape[0] != text_neighbors.shape[0]:
        raise ValueError("kNN row counts differ")
    overlaps = np.empty(len(visual_neighbors), dtype=np.float64)
    for index in range(len(overlaps)):
        overlaps[index] = np.intersect1d(
            visual_neighbors[index, :k], text_neighbors[index, :k], assume_unique=True,
        ).size / k
    return float(overlaps.mean())


def linear_cka(
    visual: np.ndarray,
    text_centered: np.ndarray,
    visual_cov_frob_sq: float | None = None,
    text_cov_frob_sq: float | None = None,
) -> float:
    z = np.array(visual, dtype=np.float64, copy=True)
    z -= z.mean(axis=0, keepdims=True)
    cross = z.T @ text_centered
    numerator = np.square(cross).sum()
    z_norm = np.square(z.T @ z).sum() if visual_cov_frob_sq is None else visual_cov_frob_sq
    e_norm = (
        np.square(text_centered.T @ text_centered).sum()
        if text_cov_frob_sq is None else text_cov_frob_sq
    )
    denominator = math.sqrt(z_norm * e_norm)
    return float(numerator / denominator) if denominator > 0 else math.nan


def ridge_r2(
    visual: np.ndarray, text: np.ndarray, alpha: float, repeats: int, seed: int, device: str,
) -> tuple[float, list[float], list[int]]:
    if alpha <= 0:
        raise ValueError("--ridge-alpha must be positive")
    n = len(visual)
    rng = np.random.default_rng(seed)
    scores, split_seeds = [], []
    target_device = torch.device(device)
    for repeat in range(repeats):
        split_seed = int(rng.integers(0, 2**31 - 1))
        split_seeds.append(split_seed)
        permutation = np.random.default_rng(split_seed).permutation(n)
        train_idx, test_idx = permutation[: n // 2], permutation[n // 2 :]
        x_train_np = visual[train_idx].astype(np.float64)
        x_test_np = visual[test_idx].astype(np.float64)
        y_train_np = text[train_idx].astype(np.float64)
        y_test_np = text[test_idx].astype(np.float64)
        x_mean, y_mean = x_train_np.mean(0, keepdims=True), y_train_np.mean(0, keepdims=True)
        x_train_np -= x_mean
        x_test_np -= x_mean
        y_train_np -= y_mean
        y_test_centered = y_test_np - y_mean
        x_train = torch.as_tensor(x_train_np, dtype=torch.float64, device=target_device)
        x_test = torch.as_tensor(x_test_np, dtype=torch.float64, device=target_device)
        y_train = torch.as_tensor(y_train_np, dtype=torch.float64, device=target_device)
        eye_size = min(x_train.shape)
        eye = torch.eye(eye_size, dtype=torch.float64, device=target_device)
        if x_train.shape[1] <= x_train.shape[0]:
            weights = torch.linalg.solve(x_train.T @ x_train + alpha * eye, x_train.T @ y_train)
            prediction = x_test @ weights
        else:
            dual = torch.linalg.solve(x_train @ x_train.T + alpha * eye, y_train)
            prediction = (x_test @ x_train.T) @ dual
        residual = prediction.cpu().numpy() - y_test_centered
        # Standard variance-weighted multi-output R² uses the held-out mean in
        # the SST denominator; the train mean is used only for the fitted intercept.
        denominator = np.square(y_test_np - y_test_np.mean(axis=0, keepdims=True)).sum()
        scores.append(float(1.0 - np.square(residual).sum() / denominator) if denominator > 0 else math.nan)
        del x_train, x_test, y_train, eye, prediction
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    return float(np.mean(scores)), scores, split_seeds


def pca_cca(visual_scores: np.ndarray, text_scores: np.ndarray, count: int = 20) -> list[float]:
    rank = min(visual_scores.shape[1], text_scores.shape[1])
    x = visual_scores[:, :rank].astype(np.float64)
    y = text_scores[:, :rank].astype(np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=0, keepdims=True), 1e-12)
    y /= np.maximum(np.linalg.norm(y, axis=0, keepdims=True), 1e-12)
    correlations = np.linalg.svd(x.T @ y, compute_uv=False)
    return [float(np.clip(value, 0.0, 1.0)) for value in correlations[:count]]


def vsa_metric(spectrum: dict[str, np.ndarray], text: np.ndarray) -> float:
    k = min(text.shape[0] - 1, spectrum["scores"].shape[1], 512)
    r = compute_language_usefulness(spectrum["scores"][:, :k], text)
    return float(compute_spectral_metrics(spectrum["lam"][:k], r, k=k)["VSA"])


def upsert_rows(path: Path, new_rows: list[dict[str, object]], overwrite: bool) -> None:
    existing: list[dict[str, str]] = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    keys = {(str(row["name"]), str(row["text_domain"]), str(row["metric_name"])) for row in new_rows}
    existing = [r for r in existing if (r["name"], r["text_domain"], r["metric_name"]) not in keys]
    existing.extend({field: str(row.get(field, "")) for field in CSV_FIELDS} for row in new_rows)
    existing.sort(key=lambda row: (row["name"], row["text_domain"], row["metric_name"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader(); writer.writerows(existing)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
    names = model_names(args.models_config, args.model)
    names = [name for index, name in enumerate(names) if index % args.num_shards == args.shard_index]

    bundles: dict[str, dict[str, Any]] = {}
    audit: dict[str, Any] = {}
    for domain in args.domains:
        path = resolve_text_path(args.text_root, domain)
        if not path.is_file():
            meta = read_json(path.with_suffix(".meta.json"))
            audit[domain] = {"available": False, "path": str(path), "metadata": meta}
            print(f"SKIP domain={domain}: missing {path}", flush=True)
            continue
        bundle = text_bundle(path, args)
        bundle["neighbors"] = centered_l2_knn(bundle["array"], 20, args.device, args.knn_block_size)
        bundles[domain] = bundle
        audit[domain] = {"available": True, "path": str(path), **bundle["diag"]}

    metadata = read_json(args.metadata)
    metadata.update({
        "schema_version": 3, "seed": args.seed, "ridge_alpha": args.ridge_alpha,
        "ridge_repeats": args.ridge_repeats, "cca_definition": (
            f"CCA in the leading min({args.cca_pca_rank}, ranks) PCA score spaces; first 20"
        ),
        "preprocessing": {
            "VSA": "visual centered only; FP32 storage, FP64 SVD; no row L2",
            "mutual_kNN": "each space centered then row-L2-normalized",
            "cm_cka": "linear CKA on centered matrices",
            "cm_r2": "train-mean centered ridge, random half train/test",
        },
        "domain_audit": audit,
    })
    metadata.setdefault("per_model", {})
    existing_keys: set[tuple[str, str, str]] = set()
    if args.output.is_file() and not args.overwrite:
        with args.output.open(encoding="utf-8", newline="") as handle:
            existing_keys = {
                (row["name"], row["text_domain"], row["metric_name"])
                for row in csv.DictReader(handle)
            }

    for model_index, name in enumerate(names, 1):
        pending_domains = []
        for domain in bundles:
            expected = set(BASE_METRICS) | (set(ABTT_METRICS) if domain == "caption" else set())
            if args.overwrite or not all((name, domain, metric) in existing_keys for metric in expected):
                pending_domains.append(domain)
        if not pending_domains:
            print(f"[{model_index}/{len(names)}] {name}: already complete", flush=True)
            continue
        feature_path = args.feature_root / f"{name}__coco4618.npy"
        if not feature_path.is_file():
            metadata["per_model"][name] = {"available": False, "reason": f"missing {feature_path}"}
            print(f"SKIP model={name}: missing features", flush=True)
            continue
        visual = np.load(feature_path, mmap_mode="r")
        if visual.dtype != np.float32 or visual.ndim != 2 or not np.isfinite(visual).all():
            raise ValueError(f"Visual features must be finite FP32: {feature_path}")
        visual_ids = read_ids(feature_path)
        visual_meta = read_json(feature_path.with_suffix(".meta.json"))
        n_tokens = visual_meta.get("n_tokens", visual_meta.get("tokens", ""))
        full_spectrum = compute_visual_spectrum(np.asarray(visual))
        full_neighbors = centered_l2_knn(np.asarray(visual), 20, args.device, args.knn_block_size)
        per_model = {"available": True, "d": int(visual.shape[1]), "domains": {}}
        rows: list[dict[str, object]] = []

        for domain in pending_domains:
            bundle = bundles[domain]
            index_by_id = {identifier: index for index, identifier in enumerate(visual_ids)}
            missing_ids = [identifier for identifier in bundle["ids"] if identifier not in index_by_id]
            if missing_ids:
                per_model["domains"][domain] = {"available": False, "missing_visual_ids": len(missing_ids)}
                continue
            selected = np.asarray([index_by_id[value] for value in bundle["ids"]], dtype=int)
            z = np.asarray(visual)[selected]
            is_full = len(selected) == len(visual) and np.array_equal(selected, np.arange(len(visual)))
            spectrum = full_spectrum if is_full else compute_visual_spectrum(z)
            visual_neighbors = full_neighbors if is_full else centered_l2_knn(z, 20, args.device, args.knn_block_size)
            k_value = min(len(z) - 1, z.shape[1], 512)
            values = {
                "VSA": vsa_metric(spectrum, bundle["array"]),
                "mutual_kNN_k5": mutual_knn(visual_neighbors, bundle["neighbors"], 5),
                "mutual_kNN_k10": mutual_knn(visual_neighbors, bundle["neighbors"], 10),
                "mutual_kNN_k20": mutual_knn(visual_neighbors, bundle["neighbors"], 20),
                "cm_cka": linear_cka(
                    z, bundle["centered"],
                    float(np.sum(np.asarray(spectrum["singular"], dtype=np.float64) ** 4)),
                    float(np.sum(np.asarray(bundle["singular"], dtype=np.float64) ** 4)),
                ),
            }
            ridge_seed = stable_seed(args.seed, name, domain, "ridge")
            ridge_mean, ridge_values, split_seeds = ridge_r2(
                z, bundle["array"], args.ridge_alpha, args.ridge_repeats, ridge_seed, args.device,
            )
            values["cm_r2"] = ridge_mean
            if domain == "caption":
                for count, transformed in bundle["abtt"].items():
                    values[f"VSA_abtt{count}"] = vsa_metric(spectrum, transformed)
            for metric_name, value in values.items():
                rows.append({
                    "name": name, "text_domain": domain, "metric_name": metric_name,
                    "value": format(float(value), ".12g"), "d": int(z.shape[1]),
                    "n_tokens": n_tokens, "N": len(z), "K": k_value,
                })
            visual_cca_scores = spectrum["scores"][:, : args.cca_pca_rank]
            per_model["domains"][domain] = {
                "available": True, "N": len(z), "ridge_seed": ridge_seed,
                "ridge_split_seeds": split_seeds, "ridge_r2_repeats": ridge_values,
                "cca_first20": pca_cca(visual_cca_scores, bundle["pca_scores"]),
            }
        upsert_rows(args.output, rows, args.overwrite)
        metadata["per_model"][name] = per_model
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[{model_index}/{len(names)}] {name}: wrote {len(rows)} rows", flush=True)

    print(json.dumps({"metrics": str(args.output), "metadata": str(args.metadata)}))


if __name__ == "__main__":
    main()
