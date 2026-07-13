#!/usr/bin/env python
"""Recover DINOv2 linear-probe LR curves and boundary diagnostics.

Older ``results_eval_linear.json`` files contain only the selected classifier,
while the complete per-classifier metrics are present in ``logs/log.txt``.
This utility extracts the final validation block from every run below an output
root and writes:

* ``best_lr_summary.csv``: one selected LR/head per run;
* ``lr_scores.csv``: the best feature configuration at every configured LR;
* ``classifier_scores.csv``: all 2 x 2 x 13 intended classifier heads.

It also identifies the historical 5-decimal classifier-name collision that
made the smallest configured LR unevaluable in batch-128, single-GPU runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re


DEFAULT_CONFIGURED_LRS = (
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    1e-1,
)
N_LAST_BLOCKS = (1, 4)
AVGPOOL_VALUES = (False, True)

CLASSIFIER_LINE_RE = re.compile(r"-- Classifier: (?P<name>\S+) \* \{(?P<metrics>.*)\}")
CLASSIFIER_NAME_RE = re.compile(
    r"^classifier_(?P<blocks>\d+)_blocks_avgpool_(?P<avgpool>True|False)_lr_(?P<lr>.+)$"
)
METRIC_RE = re.compile(
    r"['\"](?P<name>[^'\"]+)['\"]:\s*tensor\((?P<value>[-+0-9.eE]+)"
)

SUMMARY_FIELDS = (
    "run",
    "task",
    "model",
    "selected_configured_lr",
    "selected_effective_lr",
    "selected_n_last_blocks",
    "selected_avgpool",
    "validation_top1_percent",
    "lr_boundary",
    "intended_grid_boundary",
    "intended_heads",
    "evaluated_heads",
    "missing_heads",
    "lr_scale",
)
LR_FIELDS = (
    "run",
    "task",
    "model",
    "configured_lr",
    "effective_lr",
    "status",
    "best_validation_top1_percent",
    "best_validation_top5_percent",
    "best_n_last_blocks",
    "best_avgpool",
    "is_selected_lr",
    "evaluated_grid_boundary",
    "intended_grid_boundary",
)
CLASSIFIER_FIELDS = (
    "run",
    "task",
    "model",
    "classifier",
    "configured_lr",
    "effective_lr",
    "n_last_blocks",
    "avgpool",
    "status",
    "validation_top1_percent",
    "validation_top5_percent",
    "is_selected",
    "name_collision_size",
)


def _legacy_classifier_name(blocks: int, avgpool: bool, effective_lr: float) -> str:
    return f"classifier_{blocks}_blocks_avgpool_{avgpool}_lr_{effective_lr:.5f}".replace(".", "_")


def _lossless_classifier_name(blocks: int, avgpool: bool, effective_lr: float) -> str:
    lr_token = format(float(effective_lr), ".12g")
    lr_token = lr_token.replace(".", "_").replace("+", "p").replace("-", "m")
    return f"classifier_{blocks}_blocks_avgpool_{avgpool}_lr_{lr_token}"


def _parse_metrics(text: str) -> dict[str, float]:
    return {match.group("name"): float(match.group("value")) for match in METRIC_RE.finditer(text)}


def _parse_classifier_name(name: str) -> tuple[int, bool, float]:
    match = CLASSIFIER_NAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"Unrecognized classifier name: {name}")
    lr_token = match.group("lr")
    if "e" in lr_token:
        lr_token = lr_token.replace("p", "+").replace("m", "-").replace("_", ".", 1)
    else:
        lr_token = lr_token.replace("_", ".", 1)
    return int(match.group("blocks")), match.group("avgpool") == "True", float(lr_token)


def _load_validation_block(log_path: Path) -> list[dict]:
    blocks: list[dict] = []
    current: dict | None = None
    next_block_is_test = False
    with log_path.open(errors="replace") as handle:
        for line in handle:
            if "Testing on " in line:
                next_block_is_test = True
            if "running validation !" in line:
                if current is not None and current["classifiers"]:
                    blocks.append(current)
                current = {"is_test": next_block_is_test, "classifiers": []}
                next_block_is_test = False
                continue
            match = CLASSIFIER_LINE_RE.search(line)
            if match is None:
                continue
            if current is None:
                current = {"is_test": False, "classifiers": []}
            metrics = _parse_metrics(match.group("metrics"))
            if "top-1" not in metrics:
                raise ValueError(f"Classifier line has no top-1 metric in {log_path}: {line.rstrip()}")
            blocks_count, avgpool, encoded_effective_lr = _parse_classifier_name(match.group("name"))
            current["classifiers"].append(
                {
                    "name": match.group("name"),
                    "n_last_blocks": blocks_count,
                    "avgpool": avgpool,
                    "encoded_effective_lr": encoded_effective_lr,
                    "top1": metrics["top-1"],
                    "top5": metrics.get("top-5"),
                }
            )
    if current is not None and current["classifiers"]:
        blocks.append(current)

    validation_blocks = [block for block in blocks if not block["is_test"]]
    if not validation_blocks:
        raise ValueError(f"No validation classifier metrics found in {log_path}")
    # Periodic evaluation may produce several complete blocks.  The last
    # non-test block is the final model-selection evaluation.
    selected = validation_blocks[-1]["classifiers"]
    names = [item["name"] for item in selected]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate classifier metrics in final validation block: {log_path}")
    return selected


def _load_recorded_best(result_path: Path) -> list[dict]:
    entries = []
    with result_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "best_classifier" in payload:
                entries.append(payload["best_classifier"])
    return entries


def _infer_lr_scale(
    classifiers: list[dict],
    configured_lrs: tuple[float, ...],
) -> tuple[float, str]:
    encoded = [item["encoded_effective_lr"] for item in classifiers]
    scale = max(encoded) / max(configured_lrs)
    actual_names = {item["name"] for item in classifiers}
    attempted = {}
    for scheme, name_builder in (
        ("lossless", _lossless_classifier_name),
        ("legacy_5_decimal", _legacy_classifier_name),
    ):
        expected_names = {
            name_builder(blocks, avgpool, configured_lr * scale)
            for blocks in N_LAST_BLOCKS
            for avgpool in AVGPOOL_VALUES
            for configured_lr in configured_lrs
        }
        if expected_names == actual_names:
            return scale, scheme
        attempted[scheme] = expected_names
    expected_names = attempted["lossless"]
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    raise ValueError(
        "Could not map logged classifier names to the configured LR grid; "
        f"inferred scale={scale:.12g}, missing={missing[:3]}, unexpected={unexpected[:3]}"
    )


def _boundary(value: float, values: list[float] | tuple[float, ...]) -> str:
    if math.isclose(value, min(values), rel_tol=1e-12, abs_tol=0):
        return "low"
    if math.isclose(value, max(values), rel_tol=1e-12, abs_tol=0):
        return "high"
    return "interior"


def _run_identity(output_root: Path, result_path: Path) -> tuple[str, str, str]:
    relative = result_path.parent.relative_to(output_root)
    parts = relative.parts
    if len(parts) == 1:
        return str(relative), "imagenet1k", parts[0]
    return str(relative), parts[-2], parts[-1]


def _collect_run(
    output_root: Path,
    result_path: Path,
    configured_lrs: tuple[float, ...],
) -> tuple[dict, list[dict], list[dict]]:
    log_path = result_path.parent / "logs/log.txt"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing log for {result_path}: {log_path}")
    run, task, model = _run_identity(output_root, result_path)
    classifiers = _load_validation_block(log_path)
    lr_scale, name_scheme = _infer_lr_scale(classifiers, configured_lrs)
    name_builder = (
        _lossless_classifier_name if name_scheme == "lossless" else _legacy_classifier_name
    )
    by_name = {item["name"]: item for item in classifiers}

    intended = []
    for blocks in N_LAST_BLOCKS:
        for avgpool in AVGPOOL_VALUES:
            names_for_config: dict[str, list[float]] = {}
            for configured_lr in configured_lrs:
                name = name_builder(blocks, avgpool, configured_lr * lr_scale)
                names_for_config.setdefault(name, []).append(configured_lr)
            for configured_lr in configured_lrs:
                effective_lr = configured_lr * lr_scale
                name = name_builder(blocks, avgpool, effective_lr)
                collisions = names_for_config[name]
                # ModuleDict assignment kept only the last LR with this name.
                evaluated = configured_lr == collisions[-1]
                score = by_name.get(name) if evaluated else None
                if evaluated and score is None:
                    raise ValueError(f"Missing evaluated classifier {name} in {log_path}")
                intended.append(
                    {
                        "run": run,
                        "task": task,
                        "model": model,
                        "classifier": name,
                        "configured_lr": configured_lr,
                        "effective_lr": effective_lr,
                        "n_last_blocks": blocks,
                        "avgpool": avgpool,
                        "status": "evaluated" if evaluated else "missing_name_collision",
                        "validation_top1_percent": (
                            None if score is None else round(100.0 * score["top1"], 6)
                        ),
                        "validation_top5_percent": (
                            None
                            if score is None or score["top5"] is None
                            else round(100.0 * score["top5"], 6)
                        ),
                        "is_selected": False,
                        "name_collision_size": len(collisions),
                    }
                )

    evaluated = [row for row in intended if row["status"] == "evaluated"]
    recorded_best = _load_recorded_best(result_path)
    rows_by_name = {row["classifier"]: row for row in evaluated}
    matching_recorded = []
    for index, item in enumerate(recorded_best):
        row = rows_by_name.get(item.get("name"))
        if row is None:
            continue
        difference = abs(100.0 * float(item["accuracy"]) - row["validation_top1_percent"])
        if difference <= 0.011:
            matching_recorded.append((difference, -index, row))
    if matching_recorded:
        # The JSON retains the unrounded winner, so it resolves ties introduced
        # by the four-decimal tensor formatting in log.txt.
        selected = min(matching_recorded, key=lambda item: (item[0], item[1]))[2]
    else:
        selected = max(evaluated, key=lambda row: row["validation_top1_percent"])
        if recorded_best:
            raise ValueError(
                f"Best classifier reconstructed from {log_path} does not match {result_path}: "
                f"{selected['classifier']}"
            )
    selected["is_selected"] = True

    evaluated_lrs = sorted({row["configured_lr"] for row in evaluated})
    summary = {
        "run": run,
        "task": task,
        "model": model,
        "selected_configured_lr": selected["configured_lr"],
        "selected_effective_lr": selected["effective_lr"],
        "selected_n_last_blocks": selected["n_last_blocks"],
        "selected_avgpool": selected["avgpool"],
        "validation_top1_percent": selected["validation_top1_percent"],
        "lr_boundary": _boundary(selected["configured_lr"], evaluated_lrs),
        "intended_grid_boundary": _boundary(selected["configured_lr"], configured_lrs),
        "intended_heads": len(intended),
        "evaluated_heads": len(evaluated),
        "missing_heads": len(intended) - len(evaluated),
        "lr_scale": lr_scale,
    }

    lr_rows = []
    for configured_lr in configured_lrs:
        candidates = [
            row
            for row in intended
            if row["configured_lr"] == configured_lr and row["status"] == "evaluated"
        ]
        best = max(candidates, key=lambda row: row["validation_top1_percent"]) if candidates else None
        lr_rows.append(
            {
                "run": run,
                "task": task,
                "model": model,
                "configured_lr": configured_lr,
                "effective_lr": configured_lr * lr_scale,
                "status": "evaluated" if best is not None else "missing_name_collision",
                "best_validation_top1_percent": (
                    None if best is None else best["validation_top1_percent"]
                ),
                "best_validation_top5_percent": (
                    None if best is None else best["validation_top5_percent"]
                ),
                "best_n_last_blocks": None if best is None else best["n_last_blocks"],
                "best_avgpool": None if best is None else best["avgpool"],
                "is_selected_lr": configured_lr == selected["configured_lr"],
                "evaluated_grid_boundary": (
                    None if best is None else _boundary(configured_lr, evaluated_lrs)
                ),
                "intended_grid_boundary": _boundary(configured_lr, configured_lrs),
            }
        )
    return summary, lr_rows, intended


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Linear probing LR boundary summary",
        "",
        "`configured LR` is the CLI grid value; `effective LR` includes DINOv2's "
        "`batch_size * world_size / 256` scaling.",
        "",
        "| task | model | configured LR | effective LR | blocks | avgpool | val top-1 | boundary |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        warning = f" **{row['lr_boundary']}**" if row["lr_boundary"] != "interior" else "interior"
        lines.append(
            f"| {row['task']} | {row['model']} | {row['selected_configured_lr']:.12g} | "
            f"{row['selected_effective_lr']:.12g} | {row['selected_n_last_blocks']} | "
            f"{row['selected_avgpool']} | {row['validation_top1_percent']:.2f} | {warning} |"
        )
    lines.extend(
        [
            "",
            "Historical runs contain 48 evaluated heads rather than the intended 52 because the "
            "smallest two effective LRs shared the same 5-decimal classifier name. Missing rows are "
            "retained in the CSV files with `status=missing_name_collision`.",
            "",
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines))
    os.replace(temporary, path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=None)
    parser.add_argument(
        "--configured-lrs",
        type=float,
        nargs="+",
        default=list(DEFAULT_CONFIGURED_LRS),
        help="Unscaled LR grid in the same order used during training.",
    )
    args = parser.parse_args(argv)
    configured_lrs = tuple(float(lr) for lr in args.configured_lrs)
    if not configured_lrs or any(not math.isfinite(lr) or lr <= 0 for lr in configured_lrs):
        parser.error("--configured-lrs must contain finite positive values")
    if len(configured_lrs) != len(set(configured_lrs)):
        parser.error("--configured-lrs must not contain duplicates")
    if not args.output_root.is_dir():
        parser.error(f"--output-root is not a directory: {args.output_root}")
    destination = args.destination or args.output_root / "lr_analysis"
    destination.mkdir(parents=True, exist_ok=True)

    result_paths = sorted(args.output_root.rglob("results_eval_linear.json"))
    if not result_paths:
        parser.error(f"No results_eval_linear.json files found below {args.output_root}")
    summaries = []
    lr_rows = []
    classifier_rows = []
    for result_path in result_paths:
        summary, run_lr_rows, run_classifier_rows = _collect_run(
            args.output_root,
            result_path,
            configured_lrs,
        )
        summaries.append(summary)
        lr_rows.extend(run_lr_rows)
        classifier_rows.extend(run_classifier_rows)

    _write_csv(destination / "best_lr_summary.csv", SUMMARY_FIELDS, summaries)
    _write_csv(destination / "lr_scores.csv", LR_FIELDS, lr_rows)
    _write_csv(destination / "classifier_scores.csv", CLASSIFIER_FIELDS, classifier_rows)
    _write_markdown(destination / "best_lr_summary.md", summaries)
    boundary_count = sum(row["lr_boundary"] != "interior" for row in summaries)
    print(
        f"Analyzed {len(summaries)} runs: {boundary_count} selected an evaluated LR boundary. "
        f"Wrote {destination}"
    )


if __name__ == "__main__":
    main()
