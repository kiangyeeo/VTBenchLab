from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .summarize import _spearman
from .summarize_loss_proxy import _average_ranks_lower_is_better
from .utils import atomic_write_json, load_config, resolve_path


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(config: dict, ground_truth_csv: Path | None) -> tuple[Path, Path | None]:
    root = resolve_path(config, config["runtime"]["artifact_root"])
    names = list(config["tokenizers"])
    domains = list(config["protocol"]["reliable_domains"])
    rows_by_name: dict[str, list[dict]] = {}
    missing = []
    for name in names:
        path = root / "analysis" / "by_tokenizer" / f"{name}.json"
        if not path.is_file():
            missing.append(name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("complete"):
            missing.append(name)
            continue
        rows_by_name[name] = payload["rows"]
    if missing:
        raise RuntimeError(
            f"Cannot finalize predictions; missing {len(missing)} tokenizers: {missing}"
        )

    domain_losses: dict[str, dict[str, float]] = {}
    domain_ranks: dict[str, dict[str, float]] = {}
    for domain in domains:
        values = {}
        for name in names:
            matching = [row for row in rows_by_name[name] if row["domain"] == domain]
            if len(matching) != 1:
                raise RuntimeError(f"Expected one {domain} row for {name}, got {len(matching)}")
            values[name] = float(matching[0]["real"])
        domain_losses[domain] = values
        domain_ranks[domain] = _average_ranks_lower_is_better(values)

    aggregate_rank = {
        name: float(np.mean([domain_ranks[domain][name] for domain in domains]))
        for name in names
    }
    predicted_order = sorted(names, key=lambda name: (aggregate_rank[name], name))
    calibration = set(config["protocol"]["calibration_tokenizers"])
    prediction_rows = []
    for final_rank, name in enumerate(predicted_order, start=1):
        controls = {row["domain"]: row for row in rows_by_name[name]}
        prediction_rows.append(
            {
                "predicted_position": final_rank,
                "tokenizer": name,
                "registry_name": config["tokenizers"][name]["registry_name"],
                "family": config["tokenizers"][name]["family"],
                "calibration_model": name in calibration,
                "mean_domain_rank": aggregate_rank[name],
                **{f"{domain}_loss": domain_losses[domain][name] for domain in domains},
                **{f"{domain}_rank": domain_ranks[domain][name] for domain in domains},
                **{
                    f"{domain}_real_minus_shuffled": float(
                        controls[domain]["real_minus_shuffled"]
                    )
                    for domain in domains
                },
                **{
                    f"{domain}_real_minus_zero": float(controls[domain]["real_minus_zero"])
                    for domain in domains
                },
            }
        )

    prediction_path = root / "summary" / "predictions.json"
    prediction_csv = root / "summary" / "predictions.csv"
    prediction_payload = {
        "schema_version": 1,
        "metric": "mean within-domain rank of correct-image post-warmup validation loss",
        "lower_is_better": True,
        "ground_truth_accessed": False,
        "frozen_protocol": config["protocol"],
        "tokenizer_count": len(names),
        "predicted_order": predicted_order,
        "rows": prediction_rows,
    }
    # This blind artifact is committed atomically before ground truth is opened.
    atomic_write_json(prediction_path, prediction_payload)
    _write_csv(prediction_csv, prediction_rows, list(prediction_rows[0]))
    print(f"Wrote blind predictions before label reveal: {prediction_path}")

    if ground_truth_csv is None:
        return prediction_path, None

    with ground_truth_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        truth_rows = {row["name"]: row for row in csv.DictReader(handle)}
    truth = {}
    missing_truth = []
    for name in names:
        lookup = config["tokenizers"][name]["registry_name"]
        value = truth_rows.get(lookup, {}).get("qwen2_5", "")
        if value == "":
            missing_truth.append((name, lookup))
        else:
            truth[name] = float(value)
    if missing_truth:
        raise RuntimeError(f"Missing qwen2_5 ground truth rows: {missing_truth}")

    all_rho = _spearman(
        [-aggregate_rank[name] for name in names], [truth[name] for name in names]
    )
    heldout = [name for name in names if name not in calibration]
    heldout_rho = _spearman(
        [-aggregate_rank[name] for name in heldout], [truth[name] for name in heldout]
    )
    evaluation_rows = []
    truth_order = sorted(names, key=truth.get, reverse=True)
    truth_position = {name: index + 1 for index, name in enumerate(truth_order)}
    for row in prediction_rows:
        name = row["tokenizer"]
        evaluation_rows.append(
            {
                **row,
                "qwen2_5": truth[name],
                "mllm_position": truth_position[name],
                "position_error": int(row["predicted_position"]) - truth_position[name],
            }
        )
    evaluation = {
        "schema_version": 1,
        "prediction_artifact": str(prediction_path.resolve()),
        "prediction_written_before_ground_truth_read": True,
        "ground_truth_csv": str(ground_truth_csv.resolve()),
        "all_tokenizers_spearman": all_rho,
        "heldout_tokenizers_spearman": heldout_rho,
        "calibration_tokenizers": sorted(calibration),
        "heldout_count": len(heldout),
        "rows": evaluation_rows,
    }
    evaluation_path = root / "summary" / "evaluation.json"
    evaluation_csv = root / "summary" / "evaluation.csv"
    atomic_write_json(evaluation_path, evaluation)
    _write_csv(evaluation_csv, evaluation_rows, list(evaluation_rows[0]))
    print(
        f"All-model Spearman={all_rho:.4f}; held-out Spearman={heldout_rho:.4f}; "
        f"wrote {evaluation_path}"
    )
    return prediction_path, evaluation_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize the frozen full-sweep ranking")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ground-truth-csv")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    truth_path = (
        None
        if args.ground_truth_csv is None
        else Path(args.ground_truth_csv).expanduser().resolve()
    )
    summarize(config, truth_path)


if __name__ == "__main__":
    main()
