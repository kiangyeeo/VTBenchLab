#!/usr/bin/env python
"""ImageNet full-support linear probe aligned with the local k-shot protocol.

This is the directly comparable endpoint for ``linear_probe_tokenizers.py``:
the feature surface, deterministic preprocessing, 10% regularization-selection
split, L-BFGS classifier, C search, and official ImageNet validation evaluation
are unchanged. The only substantive change is that the classifier is trained
on every example in the disjoint 90% support pool instead of exactly k examples
per class.

The selection 10% is deliberately not added to the final fit because the local
k-shot protocol also reports support-only classifiers. Consequently this is
called ``full-support`` (about 1,153 examples/class on average), rather than a
100%-of-ImageNet-train full-shot probe.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch
from torchvision import datasets

import linear_probe_tokenizers as kshot


PROTOCOL_NAME = "clip-paper-v1-full-support"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def class_count_summary(labels: np.ndarray) -> dict:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=kshot.NUM_CLASSES)
    if len(counts) != kshot.NUM_CLASSES or np.any(counts == 0):
        raise ValueError("Full-support pool must contain every ImageNet class")
    return {
        "num_classes": int(len(counts)),
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
    }


def resumable_parametric_c_search(
    evaluate_candidate: Callable[[float], tuple[float, dict]],
    state_path: Path,
    protocol_config_sha256: str,
) -> tuple[float, list[dict]]:
    """Run the canonical C search while persisting every expensive full fit."""
    state = {"protocol_config_sha256": protocol_config_sha256, "candidates": {}}
    if state_path.exists():
        with state_path.open() as handle:
            state = json.load(handle)
        if state.get("protocol_config_sha256") != protocol_config_sha256:
            raise RuntimeError(
                f"C-search state does not match this protocol: {state_path}. "
                "Use --overwrite-probe to start a new search."
            )
    candidates = state.setdefault("candidates", {})

    def cached_evaluation(exponent: float) -> float:
        exponent = round(float(exponent), 12)
        key = format(exponent, ".12g")
        if key in candidates:
            print(f"Using completed C-search candidate: log10(C)={exponent:g}")
            return float(candidates[key]["selection_top1"])

        selection_top1, details = evaluate_candidate(exponent)
        candidates[key] = {
            "log10_C": exponent,
            "C": 10.0**exponent,
            "selection_top1": float(selection_top1),
            **details,
        }
        _write_json_atomic(state_path, state)
        return float(selection_top1)

    selected_c, canonical_history = kshot.parametric_c_search(cached_evaluation)
    history = [candidates[format(item["log10_C"], ".12g")] for item in canonical_history]
    return selected_c, history


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=kshot.MODEL_NAMES)
    parser.add_argument("--data-root", type=Path, default=kshot.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=kshot.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=0, help="Logistic-regression random state.")
    parser.add_argument("--selection-seed", type=int, default=kshot.DEFAULT_SELECTION_SEED)
    parser.add_argument("--selection-fraction", type=float, default=kshot.DEFAULT_SELECTION_FRACTION)
    parser.add_argument("--batch-size", type=int, default=100, help="GPU feature-extraction batch size.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--logreg-verbose", type=int, default=1)
    parser.add_argument(
        "--fixed-c",
        type=float,
        default=None,
        help="Skip C search and use this C. Faster, but not the exact k-shot protocol.",
    )
    parser.add_argument("--prepare-split-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--overwrite-probe", action="store_true")
    parser.add_argument("--overwrite-features", action="store_true")
    args = parser.parse_args(argv)

    if args.batch_size < 1 or args.num_workers < 0 or args.max_iter < 1 or args.tol <= 0:
        parser.error("batch size/max iterations must be positive, workers nonnegative, and tol positive")
    if not 0.0 < args.selection_fraction < 1.0:
        parser.error("--selection-fraction must be between zero and one")
    if args.fixed_c is not None and args.fixed_c <= 0:
        parser.error("--fixed-c must be positive")
    if args.features_only and args.probe_only:
        parser.error("--features-only and --probe-only are mutually exclusive")

    # Required by shared cache-fingerprint and classifier helpers.
    args.protocol = "clip-paper-v1"
    args.c = args.fixed_c
    fraction_string = format(args.selection_fraction, ".12g")
    args.selection_split_path = (
        args.output_root / "splits" / f"selection_seed{args.selection_seed}_fraction{fraction_string}.npz"
    )
    return args


def _validate_imagenet(args):
    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Expected ImageNet train/ and val/ under {args.data_root}")
    train = datasets.ImageFolder(train_dir)
    val = datasets.ImageFolder(val_dir)
    if len(train) != kshot.IMAGENET_TRAIN_SIZE or len(train.classes) != kshot.NUM_CLASSES:
        raise ValueError(f"Unexpected ImageNet train set: samples={len(train)}, classes={len(train.classes)}")
    if len(val) != kshot.IMAGENET_VAL_SIZE or len(val.classes) != kshot.NUM_CLASSES:
        raise ValueError(f"Unexpected ImageNet val set: samples={len(val)}, classes={len(val.classes)}")
    if train.class_to_idx != val.class_to_idx:
        raise ValueError("ImageNet train and validation class mappings differ")
    return train_dir, val_dir, train, val


def main(argv=None) -> None:
    args = parse_args(argv)
    train_dir, val_dir, target_dataset, val_target_dataset = _validate_imagenet(args)
    dataset_sha256 = kshot._dataset_order_sha256(target_dataset)
    val_dataset_sha256 = kshot._dataset_order_sha256(val_target_dataset)
    selection_indices, support_pool, selection_metadata = kshot.load_or_create_train_selection(
        target_dataset,
        args.selection_split_path,
        args.selection_fraction,
        args.selection_seed,
        dataset_sha256,
    )
    support_pool_sha256 = selection_metadata["support_pool_indices_sha256"]
    support_labels_expected = np.asarray(target_dataset.targets, dtype=np.int64)[support_pool]
    support_counts = class_count_summary(support_labels_expected)
    print(
        f"Full-support pool: samples={len(support_pool)} per_class_mean={support_counts['mean']:.3f} "
        f"min={support_counts['min']} max={support_counts['max']}"
    )
    print(f"Selection split: samples={len(selection_indices)} path={args.selection_split_path}")
    if args.prepare_split_only:
        return

    checkpoint_metadata = kshot.checkpoint_manifest(args.model)
    model_root = args.output_root / args.model
    result_root = model_root / "full_support"
    support_cache = model_root / "features_full_support"
    selection_cache = model_root / "features_selection"
    val_cache = model_root / "features_val"
    output_path = result_root / "results.json"
    search_state_path = result_root / "c_search_state.json"

    support_fingerprint = kshot._make_cache_fingerprint(
        args, checkpoint_metadata, "full_support_pool", support_pool_sha256
    )
    selection_fingerprint = kshot._make_cache_fingerprint(
        args, checkpoint_metadata, "selection", selection_metadata["selection_indices_sha256"]
    )
    val_fingerprint = kshot._make_cache_fingerprint(
        args, checkpoint_metadata, "official_val", val_dataset_sha256
    )
    cache_requirements = [
        (support_cache, len(support_pool), support_fingerprint),
        (selection_cache, len(selection_indices), selection_fingerprint),
        (val_cache, kshot.IMAGENET_VAL_SIZE, val_fingerprint),
    ]

    if args.probe_only:
        for cache_dir, count, fingerprint in cache_requirements:
            if not kshot._feature_cache_complete(cache_dir, count, fingerprint):
                raise FileNotFoundError(f"Required feature cache is missing: {cache_dir}")
    else:
        caches_complete = all(
            kshot._feature_cache_complete(cache_dir, count, fingerprint, overwrite=args.overwrite_features)
            for cache_dir, count, fingerprint in cache_requirements
        )
        if args.smoke_test or not caches_complete:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for tokenizer feature extraction")
            device = torch.device("cuda")
            encoder, transform, autocast_factory = kshot.load_feature_encoder(args.model, device)
            print(f"Loaded {args.model}: parameters={sum(p.numel() for p in encoder.parameters()):,}")
            if args.smoke_test:
                dataset = datasets.ImageFolder(val_dir, transform=transform)
                images = torch.stack([dataset[index][0] for index in range(2)]).to(device)
                with torch.inference_mode(), autocast_factory():
                    features = encoder(images)
                print(
                    f"Smoke test: images={tuple(images.shape)} features={tuple(features.shape)} "
                    f"finite={bool(torch.isfinite(features).all())}"
                )
                return

            train_dataset = datasets.ImageFolder(train_dir, transform=transform)
            val_dataset = datasets.ImageFolder(val_dir, transform=transform)
            kshot.extract_features(
                encoder, train_dataset, support_pool, support_cache, args.batch_size,
                args.num_workers, device, autocast_factory, support_fingerprint, args.overwrite_features,
            )
            kshot.extract_features(
                encoder, train_dataset, selection_indices, selection_cache, args.batch_size,
                args.num_workers, device, autocast_factory, selection_fingerprint, args.overwrite_features,
            )
            kshot.extract_features(
                encoder, val_dataset, None, val_cache, args.batch_size,
                args.num_workers, device, autocast_factory, val_fingerprint, args.overwrite_features,
            )
            del encoder
            gc.collect()
            torch.cuda.empty_cache()

    if args.features_only:
        print("Feature extraction complete; skipping the CPU probe (--features-only).")
        return

    protocol_config = {
        "protocol": PROTOCOL_NAME,
        "base_kshot_protocol": "clip-paper-v1",
        "model": args.model,
        "random_state": args.seed,
        "selection_seed": args.selection_seed,
        "selection_fraction": args.selection_fraction,
        "selection_indices_sha256": selection_metadata["selection_indices_sha256"],
        "support_pool_indices_sha256": support_pool_sha256,
        "support_definition": "all examples in the disjoint 90% support pool",
        "train_dataset_order_sha256": dataset_sha256,
        "val_dataset_order_sha256": val_dataset_sha256,
        "checkpoint_manifest_sha256": checkpoint_metadata["sha256"],
        "feature_surface": kshot.FEATURE_SURFACES[args.model],
        "feature_normalized": False,
        "preprocessing": kshot.TRANSFORM_DESCRIPTIONS[args.model],
        "feature_extraction_batch_size": args.batch_size,
        "classifier": "sklearn.linear_model.LogisticRegression(solver='lbfgs')",
        "max_iter": args.max_iter,
        "tol": args.tol,
        "fixed_C": args.fixed_c,
        "c_search_initial_log10": list(kshot.DEFAULT_C_EXPONENTS) if args.fixed_c is None else None,
        "c_search_resolution_decades": kshot.DEFAULT_C_RESOLUTION if args.fixed_c is None else None,
        "environment": kshot._environment_metadata(),
    }
    protocol_config_sha256 = kshot._sha256_json(protocol_config)
    if output_path.exists() and not args.overwrite_probe:
        with output_path.open() as handle:
            completed = json.load(handle)
        if completed.get("protocol_config_sha256") != protocol_config_sha256:
            raise RuntimeError(
                f"Existing result does not match this protocol: {output_path}. "
                "Use --overwrite-probe to replace it."
            )
        print(f"Using completed full-support result: {output_path}")
        print(json.dumps(completed["result"], indent=2))
        return
    if args.overwrite_probe and search_state_path.exists():
        search_state_path.unlink()

    support_features = np.load(support_cache / "features.npy", mmap_mode="r")
    support_labels = np.load(support_cache / "labels.npy", mmap_mode="r")
    selection_features = np.load(selection_cache / "features.npy", mmap_mode="r")
    selection_labels = np.load(selection_cache / "labels.npy", mmap_mode="r")
    val_features = np.load(val_cache / "features.npy", mmap_mode="r")
    val_labels = np.load(val_cache / "labels.npy", mmap_mode="r")
    if not np.array_equal(support_labels, support_labels_expected):
        raise ValueError("Full-support feature-cache labels do not match the support pool")
    if len(np.unique(selection_labels)) != kshot.NUM_CLASSES:
        raise ValueError("Selection cache does not contain every ImageNet class")
    if len(val_labels) != kshot.IMAGENET_VAL_SIZE:
        raise ValueError(f"Expected {kshot.IMAGENET_VAL_SIZE} validation examples")

    search_started = time.time()
    if args.fixed_c is None:
        def evaluate_candidate(exponent: float) -> tuple[float, dict]:
            c_value = 10.0**exponent
            started = time.time()
            print(
                f"Selecting C: model={args.model} samples={len(support_labels)} "
                f"dim={support_features.shape[1]} C={c_value:.12g}"
            )
            classifier = kshot._fit_logistic_regression(args, support_features, support_labels, c_value)
            selection_top1, _ = kshot.evaluate_classifier(
                classifier, selection_features, selection_labels, compute_top5=False
            )
            details = {
                "n_iter_max": int(np.max(classifier.n_iter_)),
                "converged": bool(np.max(classifier.n_iter_) < args.max_iter),
                "elapsed_seconds": time.time() - started,
            }
            del classifier
            gc.collect()
            return selection_top1, details

        selected_c, search_history = resumable_parametric_c_search(
            evaluate_candidate, search_state_path, protocol_config_sha256
        )
        selected_item = next(
            item for item in search_history if np.isclose(item["C"], selected_c, rtol=1e-12)
        )
        selection_top1 = selected_item["selection_top1"]
    else:
        selected_c = args.fixed_c
        selection_top1 = None
        search_history = []
    search_elapsed_seconds = time.time() - search_started

    final_started = time.time()
    print(
        f"Final full-support fit: model={args.model} samples={len(support_labels)} "
        f"dim={support_features.shape[1]} C={selected_c:.12g}"
    )
    classifier = kshot._fit_logistic_regression(args, support_features, support_labels, selected_c)
    top1, top5 = kshot.evaluate_classifier(classifier, val_features, val_labels)
    result = {
        "shot": "full_support",
        "train_samples": int(len(support_labels)),
        "train_samples_per_class": support_counts,
        "selection_samples": int(len(selection_labels)),
        "selected_C": float(selected_c),
        "selection_top1": selection_top1,
        "C_search": search_history,
        "top1": top1,
        "top5": top5,
        "max_iter": args.max_iter,
        "n_iter_max": int(np.max(classifier.n_iter_)),
        "converged": bool(np.max(classifier.n_iter_) < args.max_iter),
        "search_elapsed_seconds": search_elapsed_seconds,
        "final_fit_and_eval_elapsed_seconds": time.time() - final_started,
    }
    payload = {
        "protocol": PROTOCOL_NAME,
        "protocol_description": (
            "K-shot-aligned full-support probe: all examples in the disjoint 90% support pool; "
            "the 10% selection split is never added to the final fit"
        ),
        "protocol_config": protocol_config,
        "protocol_config_sha256": protocol_config_sha256,
        "model": args.model,
        "dataset": "ImageNet-1K",
        "selection_split": selection_metadata,
        "support_pool": {
            "count": int(len(support_pool)),
            "indices_sha256": support_pool_sha256,
            "samples_per_class": support_counts,
        },
        "final_evaluation_split": {
            "name": "official ImageNet validation",
            "count": kshot.IMAGENET_VAL_SIZE,
            "dataset_order_sha256": val_dataset_sha256,
        },
        "feature_dim": int(support_features.shape[1]),
        "feature_surface": kshot.FEATURE_SURFACES[args.model],
        "feature_normalized": False,
        "preprocessing": kshot.TRANSFORM_DESCRIPTIONS[args.model],
        "checkpoint_manifest": checkpoint_metadata,
        "result": result,
    }
    _write_json_atomic(output_path, payload)
    print(json.dumps(result, indent=2))
    print(f"Result: {output_path}")


if __name__ == "__main__":
    main()
