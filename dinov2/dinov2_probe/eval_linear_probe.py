from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .common import add_common_paths, configured_root, load_dataset_config, load_model_config, split_csv, write_json
from .datasets import class_count_from_targets, dataset_spec, linear_dataset_names, resolve_split
from .features import ensure_features, feature_cache_path, load_feature_cache
from .metrics import tensor_targets, topk_accuracy


def _ensure_pair(
    *,
    model: str,
    dataset: str,
    data_root: Path,
    feature_root: Path,
    model_root: Path,
    args,
):
    train_dataset, train_split = resolve_split(dataset, "train", args.dataset_config)
    eval_dataset, eval_split = resolve_split(dataset, "eval", args.dataset_config)
    train_spec = dataset_spec(train_dataset, args.dataset_config)
    eval_spec = dataset_spec(eval_dataset, args.dataset_config)

    ensure_features(
        model_alias=model,
        dataset_name=train_dataset,
        split=train_split,
        data_root=data_root,
        feature_root=feature_root,
        model_root=model_root,
        model_config=args.model_config,
        dataset_config=args.dataset_config,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
        allow_download=args.allow_download,
        dataset_download=args.dataset_download,
        overwrite=args.overwrite_features,
        smoke=args.smoke,
        device=args.device,
        amp_dtype=args.amp_dtype,
    )
    ensure_features(
        model_alias=model,
        dataset_name=eval_dataset,
        split=eval_split,
        data_root=data_root,
        feature_root=feature_root,
        model_root=model_root,
        model_config=args.model_config,
        dataset_config=args.dataset_config,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
        allow_download=args.allow_download,
        dataset_download=args.dataset_download,
        overwrite=args.overwrite_features,
        smoke=args.smoke,
        device=args.device,
        amp_dtype=args.amp_dtype,
    )
    train_path = feature_cache_path(
        feature_root, model, train_dataset, train_split, aggregation=train_spec.get("aggregation", "image")
    )
    eval_path = feature_cache_path(
        feature_root, model, eval_dataset, eval_split, aggregation=eval_spec.get("aggregation", "image")
    )
    return train_path, eval_path, train_dataset, train_split, eval_dataset, eval_split


def run_linear_probe(
    *,
    model: str,
    dataset: str,
    data_root: Path,
    feature_root: Path,
    result_root: Path,
    model_root: Path,
    args,
):
    train_path, eval_path, train_dataset, train_split, eval_dataset, eval_split = _ensure_pair(
        model=model,
        dataset=dataset,
        data_root=data_root,
        feature_root=feature_root,
        model_root=model_root,
        args=args,
    )
    result_path = result_root / "linear_probe" / model / f"{dataset}.json"
    if result_path.exists() and not args.overwrite_results:
        print(f"[skip] result exists: {result_path}")
        return

    train_cache = load_feature_cache(train_path)
    eval_cache = load_feature_cache(eval_path)
    x_train = train_cache["features"].float()
    y_train = tensor_targets(train_cache["targets"])
    x_eval = eval_cache["features"].float()
    y_eval = eval_cache["targets"]
    num_classes = dataset_spec(dataset, args.dataset_config).get("num_classes") or class_count_from_targets(
        y_train.tolist()
    )

    classifier = torch.nn.Linear(x_train.shape[1], int(num_classes)).to(args.device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(args.epochs, 1) * max(1, len(x_train) // args.train_batch_size))
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    start = time.time()

    classifier.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"linear {model}/{dataset} epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        total_loss = 0.0
        for feats, labels in pbar:
            feats = feats.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            logits = classifier(feats)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    classifier.eval()
    logits = []
    with torch.no_grad():
        for i in tqdm(range(0, len(x_eval), args.eval_batch_size), desc=f"eval linear {model}/{dataset}", dynamic_ncols=True):
            feats = x_eval[i : i + args.eval_batch_size].to(args.device, non_blocking=True)
            logits.append(classifier(feats).cpu())
    logits = torch.cat(logits, dim=0)
    metrics = topk_accuracy(logits, y_eval, topk=(1, 5))

    payload = {
        "protocol": "linear_probe",
        "model": model,
        "dataset": dataset,
        "train_dataset": train_dataset,
        "train_split": train_split,
        "eval_dataset": eval_dataset,
        "eval_split": eval_split,
        "metrics": metrics,
        "feature_dim": int(x_train.shape[1]),
        "num_train": int(x_train.shape[0]),
        "num_eval": int(x_eval.shape[0]),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "runtime_sec": time.time() - start,
        "smoke": args.smoke,
    }
    write_json(result_path, payload)
    print(f"[done] {result_path} top1={metrics.get('top1', 0):.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen-feature linear probing.")
    add_common_paths(parser)
    parser.add_argument("--models", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dataset-download", action="store_true")
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    args = parser.parse_args()

    model_cfg = load_model_config(args.model_config)
    models = split_csv(args.models, model_cfg["models"].keys())
    datasets = split_csv(args.datasets, linear_dataset_names(args.dataset_config))
    data_root = configured_root(args, "data_root", "data/dinov2_baseline")
    feature_root = configured_root(args, "feature_root", "outputs/dinov2_baseline/features")
    result_root = configured_root(args, "result_root", "outputs/dinov2_baseline/results")
    model_root = Path(args.model_root)

    for model in models:
        for dataset in datasets:
            run_linear_probe(
                model=model,
                dataset=dataset,
                data_root=data_root,
                feature_root=feature_root,
                result_root=result_root,
                model_root=model_root,
                args=args,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

