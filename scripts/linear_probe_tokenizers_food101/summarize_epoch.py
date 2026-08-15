#!/usr/bin/env python
"""Validate and report all Food-101 tokenizer scores at one epoch barrier."""

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import statistics


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE / "outputs" / "food101_linear_probing_dinov2_single_surface"
)
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "tokenizers_from_setup.tsv"
PROTOCOL_VERSION = "tokenizer_linear_probe_food101_balanced_epoch_barrier_v2"
EXPECTED_CONFIGS = 45
INDEPENDENT_DUPLICATE_MODEL = "clip_meta__l14"
METRIC_KEYS = ("top-1", "top-5", "macro_top-1", "macro_top-5")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate one completed Food-101 epoch across all tokenizers",
        allow_abbrev=False,
    )
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--total-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing required result: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected one JSON object in {path}")
    return payload


def _load_manifest(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) != 2:
            raise RuntimeError(f"Expected two TSV columns at {path}:{line_number}")
        rows.append((columns[0], columns[1]))
    if len(rows) != EXPECTED_CONFIGS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONFIGS} tokenizer configurations, found {len(rows)}"
        )
    if len({setup_id for setup_id, _model in rows}) != len(rows):
        raise RuntimeError("Duplicate setup id in tokenizer manifest")
    if len({model for _setup_id, model in rows}) != len(rows):
        raise RuntimeError("Duplicate probe model in tokenizer manifest")
    return rows


def _history_record(path: Path, iteration: int) -> dict:
    matches = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing validation history: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {error}") from error
        if payload.get("iteration") == iteration:
            matches.append(payload)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one validation record at iteration {iteration} in "
            f"{path}, found {len(matches)}"
        )
    return matches[0]


def _selected_classifier(payload: dict, *, iteration: int) -> dict:
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(
            f"Validation protocol mismatch: {payload.get('protocol_version')!r}"
        )
    if payload.get("iteration") != iteration:
        raise RuntimeError(
            f"Validation iteration mismatch: {payload.get('iteration')} != {iteration}"
        )
    selected_name = payload.get("best_classifier", {}).get("name")
    matches = [
        classifier
        for classifier in payload.get("classifiers", [])
        if classifier.get("name") == selected_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Could not resolve selected classifier {selected_name!r}")
    selected = matches[0]
    metrics = selected.get("metrics", {})
    missing = set(METRIC_KEYS) - set(metrics)
    if missing:
        raise RuntimeError(f"Selected classifier is missing metrics: {sorted(missing)}")
    return selected


def _discover_protocols(output_root: Path, seed: int) -> dict[str, tuple[Path, dict]]:
    discovered = {}
    for path in sorted(output_root.glob(f"*/seed{seed}/protocol.json")):
        protocol = _load_json(path)
        model = protocol.get("model")
        if not isinstance(model, str):
            continue
        if model in discovered:
            raise RuntimeError(f"Multiple seed-{seed} output directories for model {model}")
        discovered[model] = (path.parent, protocol)
    return discovered


def _metric_percent(metrics: dict, key: str) -> float:
    value = 100.0 * float(metrics[key])
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite metric {key}: {value}")
    return value


def _load_model_epoch(
    *,
    setup_id: str,
    model: str,
    output_dir: Path,
    protocol: dict,
    epoch: int,
    total_epochs: int,
) -> dict:
    if protocol.get("version") != PROTOCOL_VERSION:
        raise RuntimeError(
            f"{model} protocol mismatch: {protocol.get('version')!r}"
        )
    if protocol.get("tokenizer_setup_id") != setup_id:
        raise RuntimeError(
            f"{model} setup mismatch: {protocol.get('tokenizer_setup_id')!r} != "
            f"{setup_id!r}"
        )
    if protocol.get("epochs") != total_epochs:
        raise RuntimeError(
            f"{model} total epochs mismatch: {protocol.get('epochs')} != {total_epochs}"
        )
    epoch_length = int(protocol.get("epoch_length_updates", 0))
    if epoch_length <= 0:
        raise RuntimeError(f"{model} has invalid epoch length: {epoch_length}")
    iteration = epoch * epoch_length
    payload = _history_record(output_dir / "metrics_history.jsonl", iteration)
    selected = _selected_classifier(payload, iteration=iteration)
    metrics = selected["metrics"]
    return {
        "setup_id": setup_id,
        "model": model,
        "output_dir": str(output_dir),
        "epoch": epoch,
        "iteration": iteration,
        "validation_top1_pct": _metric_percent(metrics, "top-1"),
        "validation_top5_pct": _metric_percent(metrics, "top-5"),
        "validation_macro_top1_pct": _metric_percent(metrics, "macro_top-1"),
        "validation_macro_top5_pct": _metric_percent(metrics, "macro_top-5"),
        "selected_classifier": selected["name"],
        "base_lr": float(selected["base_lr"]),
        "effective_lr": float(selected["effective_lr"]),
        "protocol_fingerprint": protocol["fingerprint"],
    }


def _add_final_test(record: dict, output_dir: Path) -> None:
    payload = _load_json(output_dir / "results_test_linear.json")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"{record['model']} final-test protocol mismatch")
    if payload.get("iteration") != record["iteration"]:
        raise RuntimeError(f"{record['model']} final-test iteration mismatch")
    if payload.get("selected_classifier", {}).get("name") != record["selected_classifier"]:
        raise RuntimeError(f"{record['model']} final-test classifier mismatch")
    metrics = payload.get("test_metrics", {})
    missing = set(METRIC_KEYS) - set(metrics)
    if missing:
        raise RuntimeError(
            f"{record['model']} final test is missing metrics: {sorted(missing)}"
        )
    record.update(
        {
            "test_top1_pct": _metric_percent(metrics, "top-1"),
            "test_top5_pct": _metric_percent(metrics, "top-5"),
            "test_macro_top1_pct": _metric_percent(metrics, "macro_top-1"),
            "test_macro_top5_pct": _metric_percent(metrics, "macro_top-5"),
        }
    )


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = 0.5 * (cursor + 1 + end)
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return float("nan")
    return numerator / (left_scale * right_scale)


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = _parse_args()
    if args.total_epochs <= 0:
        raise ValueError("--total-epochs must be positive")
    if not 1 <= args.epoch <= args.total_epochs:
        raise ValueError("--epoch must be in [1, --total-epochs]")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")

    output_root = args.output_root.expanduser().resolve()
    manifest = _load_manifest(args.manifest.expanduser().resolve())
    protocols = _discover_protocols(output_root, args.seed)
    manifest_models = {model for _setup_id, model in manifest}
    missing_models = sorted(manifest_models - set(protocols))
    if missing_models:
        raise RuntimeError(f"Missing protocol outputs for models: {missing_models}")

    records = []
    previous_top1 = {}
    for setup_id, model in manifest:
        output_dir, protocol = protocols[model]
        record = _load_model_epoch(
            setup_id=setup_id,
            model=model,
            output_dir=output_dir,
            protocol=protocol,
            epoch=args.epoch,
            total_epochs=args.total_epochs,
        )
        if args.epoch > 1:
            previous = _load_model_epoch(
                setup_id=setup_id,
                model=model,
                output_dir=output_dir,
                protocol=protocol,
                epoch=args.epoch - 1,
                total_epochs=args.total_epochs,
            )
            previous_top1[model] = previous["validation_top1_pct"]
            record["delta_top1_vs_previous_pct"] = (
                record["validation_top1_pct"] - previous["validation_top1_pct"]
            )
        else:
            record["delta_top1_vs_previous_pct"] = None
        if args.epoch == args.total_epochs:
            _add_final_test(record, output_dir)
        records.append(record)

    records.sort(key=lambda item: (-item["validation_top1_pct"], item["setup_id"]))
    for rank, record in enumerate(records, start=1):
        record["validation_rank"] = rank

    stability_all = None
    stability_independent = None
    if args.epoch > 1:
        current = [record["validation_top1_pct"] for record in records]
        previous = [previous_top1[record["model"]] for record in records]
        stability_all = _spearman(current, previous)
        independent = [
            record for record in records if record["model"] != INDEPENDENT_DUPLICATE_MODEL
        ]
        stability_independent = _spearman(
            [record["validation_top1_pct"] for record in independent],
            [previous_top1[record["model"]] for record in independent],
        )

    top1_values = [record["validation_top1_pct"] for record in records]
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "epoch": args.epoch,
        "total_epochs": args.total_epochs,
        "seed": args.seed,
        "configuration_count": len(records),
        "independent_tokenizer_count": len(records) - 1,
        "validation_top1_mean_pct": statistics.fmean(top1_values),
        "validation_top1_median_pct": statistics.median(top1_values),
        "rank_stability_vs_previous_all45": stability_all,
        "rank_stability_vs_previous_independent44": stability_independent,
        "records": records,
    }

    stem = f"epoch_{args.epoch:02d}_seed{args.seed}"
    report_dir = output_root / "epoch_reports"
    json_path = report_dir / f"{stem}.json"
    csv_path = report_dir / f"{stem}.csv"
    markdown_path = report_dir / f"{stem}.md"
    _atomic_write(json_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")

    fieldnames = [
        "validation_rank",
        "setup_id",
        "model",
        "epoch",
        "iteration",
        "validation_top1_pct",
        "delta_top1_vs_previous_pct",
        "validation_top5_pct",
        "validation_macro_top1_pct",
        "validation_macro_top5_pct",
        "selected_classifier",
        "base_lr",
        "effective_lr",
        "test_top1_pct",
        "test_top5_pct",
        "test_macro_top1_pct",
        "test_macro_top5_pct",
        "output_dir",
        "protocol_fingerprint",
    ]
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    _atomic_write(csv_path, csv_buffer.getvalue())

    markdown = [
        f"# Food-101 epoch {args.epoch} score report",
        "",
        f"- Completed configurations: {len(records)}/{EXPECTED_CONFIGS}",
        f"- Validation top-1 mean: {statistics.fmean(top1_values):.4f}%",
        f"- Validation top-1 median: {statistics.median(top1_values):.4f}%",
    ]
    if stability_all is not None:
        markdown.extend(
            [
                f"- Rank stability vs epoch {args.epoch - 1}, all 45: {stability_all:.6f}",
                (
                    f"- Rank stability vs epoch {args.epoch - 1}, independent 44: "
                    f"{stability_independent:.6f}"
                ),
            ]
        )
    markdown.extend(
        [
            "",
            "| Rank | Setup | Model | Val top-1 | Delta | Macro top-1 | Base LR | Test top-1 |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        test_top1 = record.get("test_top1_pct")
        markdown.append(
            "| {rank} | {setup} | {model} | {top1:.4f}% | {delta} | "
            "{macro:.4f}% | {lr:g} | {test} |".format(
                rank=record["validation_rank"],
                setup=record["setup_id"],
                model=record["model"],
                top1=record["validation_top1_pct"],
                delta=(
                    ""
                    if record["delta_top1_vs_previous_pct"] is None
                    else f"{record['delta_top1_vs_previous_pct']:+.4f}"
                ),
                macro=record["validation_macro_top1_pct"],
                lr=record["base_lr"],
                test="" if test_top1 is None else f"{test_top1:.4f}%",
            )
        )
    markdown_text = "\n".join(markdown) + "\n"
    _atomic_write(markdown_path, markdown_text)

    print(markdown_text, end="")
    print(f"Reports: {markdown_path}, {csv_path}, {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
