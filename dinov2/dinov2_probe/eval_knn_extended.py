from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .common import add_common_paths, configured_root, load_model_config, split_csv, write_json
from .datasets import class_count_from_targets, dataset_spec, linear_dataset_names, resolve_split
from .eval_linear_probe import _ensure_pair
from .features import load_feature_cache
from .metrics import tensor_targets, topk_accuracy


def knn_predict(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    eval_features: torch.Tensor,
    *,
    k: int,
    temperature: float,
    num_classes: int,
    chunk_size: int,
    device: str,
) -> torch.Tensor:
    train_features = F.normalize(train_features.float(), dim=1).to(device)
    eval_features = F.normalize(eval_features.float(), dim=1)
    train_targets = train_targets.long().to(device)
    logits = []
    for start in tqdm(range(0, len(eval_features), chunk_size), desc=f"kNN k={k}", dynamic_ncols=True):
        feats = eval_features[start : start + chunk_size].to(device)
        sims = feats @ train_features.T
        top_sims, top_idx = sims.topk(k, dim=1)
        labels = train_targets[top_idx]
        weights = torch.softmax(top_sims / temperature, dim=1)
        probs = torch.zeros((len(feats), num_classes), device=device)
        probs.scatter_add_(1, labels, weights)
        logits.append(probs.cpu())
    return torch.cat(logits, dim=0)


def run_knn(*, model: str, dataset: str, protocol: str, data_root: Path, feature_root: Path, result_root: Path, model_root: Path, args):
    train_path, eval_path, train_dataset, train_split, eval_dataset, eval_split = _ensure_pair(
        model=model,
        dataset=dataset,
        data_root=data_root,
        feature_root=feature_root,
        model_root=model_root,
        args=args,
    )
    result_path = result_root / protocol / model / f"{dataset}.json"
    if result_path.exists() and not args.overwrite_results:
        print(f"[skip] result exists: {result_path}")
        return

    train_cache = load_feature_cache(train_path)
    eval_cache = load_feature_cache(eval_path)
    train_targets = tensor_targets(train_cache["targets"])
    num_classes = dataset_spec(dataset, args.dataset_config).get("num_classes") or class_count_from_targets(
        train_targets.tolist()
    )
    start = time.time()
    metrics = {}
    for k in args.k:
        logits = knn_predict(
            train_cache["features"],
            train_targets,
            eval_cache["features"],
            k=k,
            temperature=args.temperature,
            num_classes=int(num_classes),
            chunk_size=args.knn_chunk_size,
            device=args.device,
        )
        for metric, value in topk_accuracy(logits, eval_cache["targets"], topk=(1, 5)).items():
            metrics[f"k{k}_{metric}"] = value

    payload = {
        "protocol": protocol,
        "model": model,
        "dataset": dataset,
        "train_dataset": train_dataset,
        "train_split": train_split,
        "eval_dataset": eval_dataset,
        "eval_split": eval_split,
        "metrics": metrics,
        "feature_dim": int(train_cache["features"].shape[1]),
        "num_train": int(train_cache["features"].shape[0]),
        "num_eval": int(eval_cache["features"].shape[0]),
        "k": args.k,
        "temperature": args.temperature,
        "runtime_sec": time.time() - start,
        "smoke": args.smoke,
    }
    write_json(result_path, payload)
    first_key = f"k{args.k[0]}_top1"
    print(f"[done] {result_path} {first_key}={metrics.get(first_key, 0):.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run optional frozen-feature k-NN on evaluation sets.")
    add_common_paths(parser)
    parser.add_argument("--models", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--protocol", default="extended_knn")
    parser.add_argument("--k", nargs="+", type=int, default=[20])
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--knn-chunk-size", type=int, default=1024)
    parser.add_argument("--feature-batch-size", type=int, default=256)
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
    model_root = configured_root(args, "model_root", "checkpoints/dinov2_baseline")

    for model in models:
        for dataset in datasets:
            run_knn(
                model=model,
                dataset=dataset,
                protocol=args.protocol,
                data_root=data_root,
                feature_root=feature_root,
                result_root=result_root,
                model_root=model_root,
                args=args,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
