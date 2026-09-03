from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .utils import atomic_write_json, choose_names, load_config, resolve_path, stable_seed


def _alignment(source: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    centroid = target.float().mean(dim=0)
    if float(centroid.norm()) == 0.0:
        raise RuntimeError("Target gradient centroid has zero norm")
    values = F.cosine_similarity(source.float(), centroid.unsqueeze(0), dim=1)
    return values.cpu().numpy().astype(np.float64)


def _bootstrap_ci(values: np.ndarray, seed: int, samples: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.integers(0, len(values), size=len(values))
        estimates[index] = values[chosen].mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _split_half_cosine(values: torch.Tensor, seed: int, repeats: int = 20) -> float:
    if values.shape[0] < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        order = rng.permutation(values.shape[0])
        midpoint = len(order) // 2
        left = values[order[:midpoint]].float().mean(dim=0)
        right = values[order[midpoint:]].float().mean(dim=0)
        scores.append(float(F.cosine_similarity(left, right, dim=0)))
    return float(np.mean(scores))


def _spearman(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) < 2:
        return float("nan")

    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
        result = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            result[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return result

    rank_a = ranks(values_a)
    rank_b = ranks(values_b)
    if float(rank_a.std()) == 0.0 or float(rank_b.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def summarize(config: dict, tokenizer_names: list[str], seeds: list[int]) -> Path:
    root = resolve_path(config, config["runtime"]["artifact_root"])
    output_dir = root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    generic_domain = str(config["probe"]["generic_domain"])
    target_domains = [
        domain for domain in config["probe"]["domains"] if domain != generic_domain
    ]
    bootstrap_samples = int(config["summary"]["bootstrap_samples"])
    rows = []
    for tokenizer_name in tokenizer_names:
        for seed in seeds:
            run_dir = root / "gradients" / tokenizer_name / f"seed_{seed}"
            generic = load_file(run_dir / f"{generic_domain}.safetensors", device="cpu")
            for domain in target_domains:
                target = load_file(run_dir / f"{domain}.safetensors", device="cpu")
                delta_values = _alignment(generic["delta"], target["delta"])
                raw_values = _alignment(generic["real"], target["real"])
                ci_low, ci_high = _bootstrap_ci(
                    delta_values,
                    stable_seed(int(config["summary"]["bootstrap_seed"]), f"{tokenizer_name}:{seed}:{domain}"),
                    bootstrap_samples,
                )
                signal_ratio = target["delta_norm"] / target["real_norm"].clamp_min(1e-12)
                rows.append(
                    {
                        "tokenizer": tokenizer_name,
                        "seed": seed,
                        "domain": domain,
                        "delta_alignment": float(delta_values.mean()),
                        "delta_alignment_ci_low": ci_low,
                        "delta_alignment_ci_high": ci_high,
                        "raw_alignment": float(raw_values.mean()),
                        "target_visual_signal_ratio": float(signal_ratio.mean()),
                        "target_delta_centroid_norm": float(target["delta"].float().mean(0).norm()),
                        "target_delta_split_half_cosine": _split_half_cosine(
                            target["delta"],
                            stable_seed(int(config["summary"]["bootstrap_seed"]), f"split:{domain}"),
                        ),
                        "generic_count": int(generic["delta"].shape[0]),
                        "target_count": int(target["delta"].shape[0]),
                    }
                )

    csv_path = output_dir / "domain_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregates = []
    for tokenizer_name in tokenizer_names:
        tokenizer_rows = [row for row in rows if row["tokenizer"] == tokenizer_name]
        for seed in seeds:
            seed_rows = [row for row in tokenizer_rows if row["seed"] == seed]
            aggregates.append(
                {
                    "tokenizer": tokenizer_name,
                    "seed": seed,
                    "exploratory_mean_delta_alignment": float(
                        np.mean([row["delta_alignment"] for row in seed_rows])
                    ),
                    "mean_visual_signal_ratio": float(
                        np.mean([row["target_visual_signal_ratio"] for row in seed_rows])
                    ),
                }
            )
    comparisons = []
    if len(tokenizer_names) == 2:
        left, right = tokenizer_names
        ground_truth = config.get("ground_truth", {})
        ground_truth_delta = (
            float(ground_truth.get(left, {}).get("qwen2_5_mllm", float("nan")))
            - float(ground_truth.get(right, {}).get("qwen2_5_mllm", float("nan")))
        )
        for seed in seeds:
            for domain in [*target_domains, "exploratory_mean"]:
                if domain == "exploratory_mean":
                    left_score = next(
                        row["exploratory_mean_delta_alignment"]
                        for row in aggregates
                        if row["tokenizer"] == left and row["seed"] == seed
                    )
                    right_score = next(
                        row["exploratory_mean_delta_alignment"]
                        for row in aggregates
                        if row["tokenizer"] == right and row["seed"] == seed
                    )
                else:
                    left_score = next(
                        row["delta_alignment"]
                        for row in rows
                        if row["tokenizer"] == left
                        and row["seed"] == seed
                        and row["domain"] == domain
                    )
                    right_score = next(
                        row["delta_alignment"]
                        for row in rows
                        if row["tokenizer"] == right
                        and row["seed"] == seed
                        and row["domain"] == domain
                    )
                score_delta = float(left_score - right_score)
                comparisons.append(
                    {
                        "left": left,
                        "right": right,
                        "seed": seed,
                        "domain": domain,
                        "score_delta_left_minus_right": score_delta,
                        "qwen2_5_mllm_delta_left_minus_right": ground_truth_delta,
                        "recovers_mllm_direction": bool(
                            np.isfinite(ground_truth_delta)
                            and score_delta * ground_truth_delta > 0
                        ),
                    }
                )
    ranking_checks = []
    ground_truth = config.get("ground_truth", {})
    if len(tokenizer_names) >= 2 and all(
        np.isfinite(float(ground_truth.get(name, {}).get("qwen2_5_mllm", float("nan"))))
        for name in tokenizer_names
    ):
        expected_scores = {
            name: float(ground_truth[name]["qwen2_5_mllm"]) for name in tokenizer_names
        }
        expected_order = sorted(tokenizer_names, key=expected_scores.get, reverse=True)
        for seed in seeds:
            for domain in [*target_domains, "exploratory_mean"]:
                if domain == "exploratory_mean":
                    predicted_scores = {
                        name: next(
                            row["exploratory_mean_delta_alignment"]
                            for row in aggregates
                            if row["tokenizer"] == name and row["seed"] == seed
                        )
                        for name in tokenizer_names
                    }
                else:
                    predicted_scores = {
                        name: next(
                            row["delta_alignment"]
                            for row in rows
                            if row["tokenizer"] == name
                            and row["seed"] == seed
                            and row["domain"] == domain
                        )
                        for name in tokenizer_names
                    }
                predicted_order = sorted(
                    tokenizer_names, key=predicted_scores.get, reverse=True
                )
                ranking_checks.append(
                    {
                        "seed": seed,
                        "domain": domain,
                        "expected_order": expected_order,
                        "predicted_order": predicted_order,
                        "exact_order_match": predicted_order == expected_order,
                        "spearman": _spearman(
                            [predicted_scores[name] for name in tokenizer_names],
                            [expected_scores[name] for name in tokenizer_names],
                        ),
                    }
                )
    atomic_write_json(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "primary_metric": "delta_alignment",
            "delta_definition": "gradient(real image) - gradient(deranged image)",
            "rows": rows,
            "aggregates": aggregates,
            "pairwise_comparisons": comparisons,
            "ranking_checks": ranking_checks,
            "ground_truth": config.get("ground_truth", {}),
        },
    )

    lines = [
        "# Tokenizer-pair gradient compatibility",
        "",
        "Primary scores use real-minus-shuffled LoRA-B gradients. The unweighted mean is",
        "reported as exploratory because two tokenizers are insufficient for fitting weights.",
        "",
        "| Tokenizer | Seed | Domain | Delta alignment | 95% CI | Raw alignment | Visual signal | Split-half |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['tokenizer']} | {row['seed']} | {row['domain']} | "
            f"{row['delta_alignment']:.4f} | [{row['delta_alignment_ci_low']:.4f}, "
            f"{row['delta_alignment_ci_high']:.4f}] | {row['raw_alignment']:.4f} | "
            f"{row['target_visual_signal_ratio']:.4f} | "
            f"{row['target_delta_split_half_cosine']:.4f} |"
        )
    lines += [
        "",
        "## Exploratory aggregate",
        "",
        "| Tokenizer | Seed | Mean delta alignment | Mean visual signal |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['tokenizer']} | {row['seed']} | "
            f"{row['exploratory_mean_delta_alignment']:.4f} | "
            f"{row['mean_visual_signal_ratio']:.4f} |"
        )
    if comparisons:
        lines += [
            "",
            "## Pairwise reversal check",
            "",
            f"The known Qwen2.5 MLLM difference is `{comparisons[0]['left']} - "
            f"{comparisons[0]['right']} = "
            f"{comparisons[0]['qwen2_5_mllm_delta_left_minus_right']:.2f}`.",
            "",
            "| Seed | Domain | Alignment difference | Recovers MLLM direction |",
            "|---:|---|---:|---|",
        ]
        for row in comparisons:
            lines.append(
                f"| {row['seed']} | {row['domain']} | "
                f"{row['score_delta_left_minus_right']:.4f} | "
                f"{'yes' if row['recovers_mllm_direction'] else 'no'} |"
            )
    if ranking_checks:
        lines += [
            "",
            "## MLLM ranking check",
            "",
            "Expected Qwen2.5 order: `"
            + " > ".join(ranking_checks[0]["expected_order"])
            + "`.",
            "",
            "| Seed | Domain | Predicted order | Exact match | Spearman |",
            "|---:|---|---|---|---:|",
        ]
        for row in ranking_checks:
            lines.append(
                f"| {row['seed']} | {row['domain']} | "
                f"{' > '.join(row['predicted_order'])} | "
                f"{'yes' if row['exact_order_match'] else 'no'} | "
                f"{row['spearman']:.4f} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} and {output_dir / 'summary.md'}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PE-pair gradient fingerprints")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizers", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    names = choose_names(args.tokenizers, config["tokenizers"])
    seeds = args.seeds if args.seeds is not None else list(config["projector_training"]["seeds"])
    summarize(config, names, seeds)


if __name__ == "__main__":
    main()
