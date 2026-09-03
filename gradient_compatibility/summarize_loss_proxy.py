from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .summarize import _spearman
from .utils import atomic_write_json, load_config, resolve_path


def _average_ranks_lower_is_better(values: dict[str, float]) -> dict[str, float]:
    names = list(values)
    order = sorted(range(len(names)), key=lambda index: values[names[index]])
    ranks: dict[str, float] = {}
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[names[order[end]]] == values[names[order[start]]]:
            end += 1
        rank = 0.5 * ((start + 1) + end)
        for position in range(start, end):
            ranks[names[order[position]]] = rank
        start = end
    return ranks


def summarize_loss_proxy(
    config: dict,
    loss_path: Path,
    reliability_threshold: float,
) -> Path:
    root = resolve_path(config, config["runtime"]["artifact_root"])
    with (root / "summary" / "domain_scores.csv").open("r", encoding="utf-8") as handle:
        gradient_rows = list(csv.DictReader(handle))
    loss_payload = json.loads(loss_path.read_text(encoding="utf-8"))
    loss_rows = loss_payload["rows"]
    tokenizer_names = list(config["tokenizers"])
    available_domains = sorted({row["domain"] for row in loss_rows})
    diagnostic_domains = sorted(
        {
            row["domain"]
            for row in gradient_rows
            if row["tokenizer"] in tokenizer_names
        }
    )

    stability = {}
    for domain in diagnostic_domains:
        values = [
            float(row["target_delta_split_half_cosine"])
            for row in gradient_rows
            if row["domain"] == domain and row["tokenizer"] in tokenizer_names
        ]
        if not values:
            raise RuntimeError(f"No split-half diagnostics for domain {domain}")
        stability[domain] = float(np.median(values))
    reliable_domains = [
        domain for domain in diagnostic_domains if stability[domain] > reliability_threshold
    ]
    if not reliable_domains:
        raise RuntimeError("Reliability gate rejected every available loss domain")
    missing_losses = sorted(set(reliable_domains) - set(available_domains))
    if missing_losses:
        raise RuntimeError(f"Loss probe is missing reliable domains: {missing_losses}")

    domain_ranks = {}
    domain_losses = {}
    for domain in reliable_domains:
        losses = {
            name: float(next(
                row["real"]
                for row in loss_rows
                if row["domain"] == domain and row["tokenizer"] == name
            ))
            for name in tokenizer_names
        }
        domain_losses[domain] = losses
        domain_ranks[domain] = _average_ranks_lower_is_better(losses)
    aggregate_rank = {
        name: float(np.mean([domain_ranks[domain][name] for domain in reliable_domains]))
        for name in tokenizer_names
    }
    predicted_order = sorted(tokenizer_names, key=aggregate_rank.get)
    has_aggregate_ties = len(set(aggregate_rank.values())) != len(aggregate_rank)
    ground_truth = {
        name: float(config["ground_truth"][name]["qwen2_5_mllm"])
        for name in tokenizer_names
    }
    expected_order = sorted(tokenizer_names, key=ground_truth.get, reverse=True)
    spearman = _spearman(
        [-aggregate_rank[name] for name in tokenizer_names],
        [ground_truth[name] for name in tokenizer_names],
    )
    result = {
        "schema_version": 1,
        "metric": "reliability-gated mean rank of post-warmup real validation loss",
        "selection_uses_mllm_labels": False,
        "reliability_threshold": reliability_threshold,
        "domain_stability_median": stability,
        "reliable_domains": reliable_domains,
        "domain_losses": domain_losses,
        "domain_ranks": domain_ranks,
        "aggregate_rank": aggregate_rank,
        "predicted_order": predicted_order,
        "expected_order": expected_order,
        "has_aggregate_ties": has_aggregate_ties,
        "exact_order_match": not has_aggregate_ties and predicted_order == expected_order,
        "spearman": spearman,
        "loss_probe": str(loss_path.resolve()),
        "projector_seen": loss_payload.get("projector_seen", 4096),
        "max_examples_per_domain": loss_payload.get("max_examples_per_domain"),
    }
    output_json = loss_path.with_name(loss_path.stem + "_summary.json")
    output_md = loss_path.with_name(loss_path.stem + "_summary.md")
    atomic_write_json(output_json, result)
    lines = [
        "# Reliability-gated loss proxy",
        "",
        f"Reliability threshold: `{reliability_threshold:.3f}`. The gate does not use MLLM labels.",
        "",
        "| Domain | Median split-half | Selected |",
        "|---|---:|---|",
    ]
    for domain in diagnostic_domains:
        lines.append(
            f"| {domain} | {stability[domain]:.4f} | "
            f"{'yes' if domain in reliable_domains else 'no'} |"
        )
    predicted_groups = []
    for name in predicted_order:
        if predicted_groups and aggregate_rank[predicted_groups[-1][-1]] == aggregate_rank[name]:
            predicted_groups[-1].append(name)
        else:
            predicted_groups.append([name])
    predicted_text = " > ".join(" = ".join(group) for group in predicted_groups)
    lines += [
        "",
        "| Tokenizer | Mean rank |",
        "|---|---:|",
    ]
    for name in predicted_order:
        lines.append(f"| {name} | {aggregate_rank[name]:.4f} |")
    lines += [
        "",
        f"Predicted: `{predicted_text}`",
        "",
        f"Expected: `{' > '.join(expected_order)}`",
        "",
        f"Exact match: `{'yes' if result['exact_order_match'] else 'no'}`; "
        f"Spearman: `{spearman:.4f}`.",
    ]
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"))
    return output_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize reliability-gated loss proxy")
    parser.add_argument("--config", required=True)
    parser.add_argument("--loss-json", required=True)
    parser.add_argument("--reliability-threshold", type=float, default=0.2)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    summarize_loss_proxy(
        config,
        Path(args.loss_json).expanduser().resolve(),
        args.reliability_threshold,
    )


if __name__ == "__main__":
    main()
