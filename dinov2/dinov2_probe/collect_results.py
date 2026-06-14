import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .common import add_common_paths, configured_root


def _read_result(path: Path) -> Dict:
    with open(path, "r") as f:
        payload = json.load(f)
    payload["_path"] = str(path)
    return payload


def _primary_metric(payload: Dict) -> float | None:
    metrics = payload.get("metrics", {})
    for key in ("top1", "k20_top1", "k10_top1", "k100_top1", "k200_top1"):
        if key in metrics:
            return float(metrics[key])
    top1_keys = sorted(k for k in metrics if k.endswith("_top1"))
    if top1_keys:
        return float(metrics[top1_keys[0]])
    return None


def collect(result_root: Path) -> List[Dict]:
    rows = []
    for path in sorted(result_root.glob("*/*/*.json")):
        payload = _read_result(path)
        base = {
            "protocol": payload.get("protocol", path.parts[-3]),
            "model": payload.get("model", path.parts[-2]),
            "dataset": payload.get("dataset", path.stem),
            "train_dataset": payload.get("train_dataset", ""),
            "eval_dataset": payload.get("eval_dataset", ""),
            "feature_dim": payload.get("feature_dim", ""),
            "num_train": payload.get("num_train", ""),
            "num_eval": payload.get("num_eval", ""),
            "primary_top1": _primary_metric(payload),
            "path": payload.get("_path", str(path)),
        }
        for metric, value in payload.get("metrics", {}).items():
            row = dict(base)
            row["metric"] = metric
            row["value"] = value
            rows.append(row)
    return rows


def write_long(rows: List[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "protocol",
        "model",
        "dataset",
        "train_dataset",
        "eval_dataset",
        "metric",
        "value",
        "primary_top1",
        "feature_dim",
        "num_train",
        "num_eval",
        "path",
    ]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_wide(rows: List[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(dict)
    datasets = sorted({row["dataset"] for row in rows})
    for row in rows:
        if row["metric"] not in {"top1", "k20_top1", "k10_top1", "k100_top1", "k200_top1"}:
            continue
        key = (row["protocol"], row["model"])
        grouped[key][row["dataset"]] = row["value"]
    fields = ["protocol", "model", "Average top1 on available datasets"] + datasets
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (protocol, model), values in sorted(grouped.items()):
            available = [float(v) for v in values.values()]
            out = {
                "protocol": protocol,
                "model": model,
                "Average top1 on available datasets": "" if not available else f"{sum(available) / len(available):.4f}",
            }
            for dataset in datasets:
                out[dataset] = "" if dataset not in values else f"{float(values[dataset]):.4f}"
            writer.writerow(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect DINOv2 probe results into CSV files.")
    add_common_paths(parser)
    parser.add_argument("--long-output", default=None)
    parser.add_argument("--wide-output", default=None)
    args = parser.parse_args()
    result_root = configured_root(args, "result_root", "outputs/dinov2_baseline/results")
    long_output = Path(args.long_output) if args.long_output else result_root / "dinov2_probe_results_long.csv"
    wide_output = Path(args.wide_output) if args.wide_output else result_root / "dinov2_probe_results_wide.csv"
    rows = collect(result_root)
    write_long(rows, long_output)
    write_wide(rows, wide_output)
    print(f"Wrote {len(rows)} metric rows to {long_output}")
    print(f"Wrote wide table to {wide_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

