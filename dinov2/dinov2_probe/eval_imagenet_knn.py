from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .common import REPO_ROOT, add_common_paths, configured_root, load_model_config, split_csv, workspace_path, write_json


CONFIG_BY_ALIAS = {
    "dinov2_vits14": "vits14_pretrain.yaml",
    "dinov2_vitb14": "vitb14_pretrain.yaml",
    "dinov2_vitl14": "vitl14_pretrain.yaml",
    "dinov2_vitg14": "vitg14_pretrain.yaml",
}


def _weight_source(spec, allow_download: bool) -> str:
    checkpoint = workspace_path(spec["checkpoint"])
    if checkpoint.exists():
        return str(checkpoint)
    if allow_download and spec.get("official_url"):
        return spec["official_url"]
    raise FileNotFoundError(
        f"Missing checkpoint {checkpoint}. Run download_public_models_* or pass --allow-download for official URL loading."
    )


def _parse_official_knn(raw_metrics: Path):
    metrics = {}
    if not raw_metrics.exists():
        return metrics
    with open(raw_metrics, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            for key, value in item.items():
                normalized = (
                    key.replace("('full', ", "k")
                    .replace(") Top 1", "_top1")
                    .replace(") Top 5", "_top5")
                    .replace(" ", "")
                )
                metrics[normalized] = float(value)
    return metrics


def run_official_knn(model: str, args) -> None:
    if model not in CONFIG_BY_ALIAS:
        raise ValueError(f"Paper k-NN only supports DINOv2 aliases: {', '.join(CONFIG_BY_ALIAS)}")

    model_cfg = load_model_config(args.model_config)
    spec = model_cfg["models"][model]
    data_root = configured_root(args, "data_root", "data/dinov2_baseline")
    result_root = configured_root(args, "result_root", "outputs/dinov2_baseline/results")
    imagenet_root = data_root / "imagenet1k"
    output_dir = result_root / "paper_reproduction_raw" / model / "imagenet1k_knn"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_file = REPO_ROOT / "dinov2" / "configs" / "eval" / CONFIG_BY_ALIAS[model]
    weights = _weight_source(spec, args.allow_download)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        "-m",
        "dinov2.eval.knn",
        "--config-file",
        str(config_file),
        "--pretrained-weights",
        weights,
        "--output-dir",
        str(output_dir),
        "--train-dataset",
        f"ImageNet:split=TRAIN:root={imagenet_root}:extra={imagenet_root}",
        "--val-dataset",
        f"ImageNet:split=VAL:root={imagenet_root}:extra={imagenet_root}",
        "--batch-size",
        str(args.feature_batch_size),
        "--temperature",
        str(args.temperature),
        "--nb_knn",
        *[str(k) for k in args.k],
    ]
    if args.gather_on_cpu:
        cmd.append("--gather-on-cpu")

    print("[official dinov2]", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

    metrics = _parse_official_knn(output_dir / "results_eval_knn.json")
    payload = {
        "protocol": "paper_reproduction",
        "model": model,
        "dataset": "imagenet1k",
        "train_dataset": "imagenet1k",
        "train_split": "train",
        "eval_dataset": "imagenet1k",
        "eval_split": "val",
        "metrics": metrics,
        "official_dinov2_eval": True,
        "raw_output_dir": str(output_dir),
        "checkpoint_source": weights,
        "k": args.k,
        "temperature": args.temperature,
    }
    out_path = result_root / "paper_reproduction" / model / "imagenet1k.json"
    write_json(out_path, payload)
    print(f"[done] wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official DINOv2 ImageNet-1k k-NN paper reproduction.")
    add_common_paths(parser)
    parser.add_argument("--models", default="dinov2_vits14,dinov2_vitb14,dinov2_vitl14,dinov2_vitg14")
    parser.add_argument("--k", nargs="+", type=int, default=[20])
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--gather-on-cpu", action="store_true")
    args = parser.parse_args()

    models = split_csv(args.models, CONFIG_BY_ALIAS)
    for model in models:
        run_official_knn(model, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
