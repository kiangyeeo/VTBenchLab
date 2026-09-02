#!/usr/bin/env python
"""Complete E3 evaluation over both COCO text domains and both targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LinearRegression

try:
    from .data import WORKSPACE
    from .eval_common import finite_float, read_csv, write_json
    from .eval_e2 import anova_f
except ImportError:  # Direct execution
    from data import WORKSPACE
    from eval_common import finite_float, read_csv, write_json
    from eval_e2 import anova_f


DOMAINS = ("caption", "answer")
LAR_METRICS = (
    "Lift_8", "Lift_16", "Lift_32", "Lift_64", "Lift_128",
    "m50", "m90", "VSA", "LAR_64",
)
BASELINES = (
    "probe_epoch1", "retrieval-ImageNet", "CKA", "pretrain_loss",
    "A_score", "RankMe", "eff_rank",
)
LOWER_IS_BETTER = {"m50", "m90", "pretrain_loss"}
TARGET_AVG = "MLLM_Avg"
TARGET_PC1 = "PC1"
COMBINATIONS = {
    "probe_epoch1": ("probe_epoch1",),
    "probe_epoch1+Lift_64": ("probe_epoch1", "Lift_64"),
    "probe_epoch1+m50": ("probe_epoch1", "m50"),
    "probe_epoch1+VSA": ("probe_epoch1", "VSA"),
    "probe_epoch1+A_score": ("probe_epoch1", "A_score"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics", type=Path,
        default=WORKSPACE / "lar" / "results" / "lar_metrics_v2.csv",
    )
    parser.add_argument(
        "--targets", type=Path,
        default=WORKSPACE / "lar" / "configs" / "e3_targets.csv",
        help=(
            "CSV columns: name,family,MLLM_Avg,qwen3,qwen2_5,smollm2,"
            "probe_epoch1,retrieval-ImageNet,CKA,pretrain_loss,A_score"
        ),
    )
    parser.add_argument("--image-set", default="coco4618")
    parser.add_argument("--family-repeats", type=int, default=5000)
    parser.add_argument("--regret-repeats", type=int, default=20000)
    parser.add_argument("--combo-repeats", type=int, default=4000)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--shortlist-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--strict-pool", action="store_true")
    parser.add_argument(
        "--strict-metric-pool", action="store_true",
        help=(
            "Require every target-table model to have complete caption/answer LAR rows, "
            "while allowing unavailable target and baseline cells."
        ),
    )
    parser.add_argument("--allow-missing-columns", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=WORKSPACE / "lar" / "results" / "e3.json",
    )
    parser.add_argument(
        "--report", type=Path,
        default=WORKSPACE / "lar" / "results" / "e3_report.md",
    )
    parser.add_argument(
        "--figure", type=Path,
        default=WORKSPACE / "lar" / "results" / "e3_lift_curves.png",
    )
    return parser.parse_args()


def stable_seed(base: int, *parts: object) -> int:
    digest = hashlib.blake2b(
        "\x1f".join(str(part) for part in parts).encode("utf-8"), digest_size=8
    ).digest()
    return (int.from_bytes(digest, "little") + int(base)) % (2**63 - 1)


def safe_number(value: float | int | np.floating | None) -> float | int | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def coefficient(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan, math.nan
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def percentile_interval(samples: np.ndarray) -> list[float | None]:
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        return [None, None]
    low, high = np.quantile(finite, (0.025, 0.975))
    return [safe_number(low), safe_number(high)]


def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, repeats: int, rng: np.random.Generator
) -> np.ndarray:
    samples = np.full(repeats, np.nan, dtype=np.float64)
    for iteration in range(repeats):
        selected = rng.integers(0, len(x), size=len(x))
        samples[iteration] = coefficient(x[selected], y[selected])[0]
    return samples


def bootstrap_mean(
    samples: np.ndarray, repeats: int, rng: np.random.Generator
) -> np.ndarray:
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        return np.full(repeats, np.nan)
    output = np.empty(repeats, dtype=np.float64)
    for iteration in range(repeats):
        output[iteration] = rng.choice(finite, size=len(finite), replace=True).mean()
    return output


def family_sample_rhos(
    values: np.ndarray,
    target: np.ndarray,
    families: list[str],
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices_by_family: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(families):
        indices_by_family[family].append(index)
    samples = np.full(repeats, np.nan, dtype=np.float64)
    groups = list(indices_by_family.values())
    for iteration in range(repeats):
        selected = np.asarray([rng.choice(indices) for indices in groups], dtype=int)
        samples[iteration] = coefficient(values[selected], target[selected])[0]
    return samples


def regret_samples(
    values: np.ndarray,
    target: np.ndarray,
    size: int,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(values) < size:
        return np.full(repeats, np.nan)
    regrets = np.empty(repeats, dtype=np.float64)
    for iteration in range(repeats):
        selected = rng.choice(len(values), size=size, replace=False)
        picked = selected[int(np.argmax(values[selected]))]
        regrets[iteration] = float(target[selected].max() - target[picked])
    return regrets


def mean_regret(
    values: np.ndarray,
    target: np.ndarray,
    size: int,
    repeats: int,
    rng: np.random.Generator,
) -> float:
    """Backward-compatible scalar helper used by the original unit tests."""
    samples = regret_samples(values, target, size, repeats, rng)
    return float(np.nanmean(samples)) if np.isfinite(samples).any() else math.nan


def standardized_combo_regret_samples(
    features: np.ndarray,
    target: np.ndarray,
    size: int,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(target) <= size + 1:
        return np.full(repeats, np.nan)
    regrets = np.empty(repeats, dtype=np.float64)
    for iteration in range(repeats):
        holdout = rng.choice(len(target), size=size, replace=False)
        train_mask = np.ones(len(target), dtype=bool)
        train_mask[holdout] = False
        train = features[train_mask]
        mean = train.mean(axis=0)
        scale = train.std(axis=0)
        scale[scale == 0] = 1.0
        model = LinearRegression().fit((train - mean) / scale, target[train_mask])
        predictions = model.predict((features[holdout] - mean) / scale)
        picked = holdout[int(np.argmax(predictions))]
        regrets[iteration] = float(target[holdout].max() - target[picked])
    return regrets


def simulation_summary(
    samples: np.ndarray, bootstrap_repeats: int, rng: np.random.Generator
) -> dict[str, object]:
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        return {"mean": None, "std": None, "bootstrap_95_ci": [None, None]}
    boot = bootstrap_mean(finite, bootstrap_repeats, rng)
    return {
        "mean": safe_number(finite.mean()),
        "std": safe_number(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "bootstrap_95_ci": percentile_interval(boot),
        "sampling_95_interval": percentile_interval(finite),
    }


def compute_pc1(rows: list[dict[str, str]]) -> tuple[dict[str, float], dict[str, object]]:
    columns = ("qwen3", "qwen2_5", "smollm2")
    usable = [
        row for row in rows
        if all(finite_float(row.get(column)) is not None for column in columns)
    ]
    if len(usable) < 3:
        raise RuntimeError(f"PC1 requires all {columns}; only {len(usable)} complete rows")
    matrix = np.asarray([[float(row[column]) for column in columns] for row in usable])
    standardized = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    _u, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    scores = standardized @ vt[0]
    avg = np.asarray([
        finite_float(row.get(TARGET_AVG))
        if finite_float(row.get(TARGET_AVG)) is not None
        else matrix[index, 1]
        for index, row in enumerate(usable)
    ])
    if coefficient(scores, avg)[0] < 0:
        scores *= -1
        vt[0] *= -1
    explained = singular**2 / np.sum(singular**2)
    values = {row["name"]: float(value) for row, value in zip(usable, scores)}
    metadata = {
        "columns": list(columns),
        "n": len(usable),
        "explained_variance_ratio": safe_number(explained[0]),
        "loadings": {column: safe_number(value) for column, value in zip(columns, vt[0])},
        "spearman_vs_MLLM_Avg": safe_number(coefficient(scores, avg)[0]),
        "sign_rule": "positive Spearman correlation with MLLM_Avg",
    }
    return values, metadata


def oriented(column: str, values: np.ndarray) -> np.ndarray:
    return -values if column in LOWER_IS_BETTER else values


def family_diagnostics(
    values: np.ndarray, target: np.ndarray, families: list[str]
) -> dict[str, object]:
    by_family_values: dict[str, list[float]] = defaultdict(list)
    by_family_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for value, outcome, family in zip(values, target, families):
        by_family_values[family].append(float(value))
        by_family_pairs[family].append((float(value), float(outcome)))
    within = {}
    for family, pairs in sorted(by_family_pairs.items()):
        if len(pairs) < 4:
            continue
        x = np.asarray([pair[0] for pair in pairs])
        y = np.asarray([pair[1] for pair in pairs])
        result = coefficient(x, y)
        within[family] = {
            "n": len(pairs), "spearman_rho": safe_number(result[0]),
            "pvalue": safe_number(result[1]),
        }
    return {
        "F_between_within": safe_number(anova_f(by_family_values)),
        "within_family_n_ge_4": within,
    }


def evaluate_candidate(
    column: str,
    values: np.ndarray,
    target: np.ndarray,
    families: list[str],
    args: argparse.Namespace,
    seed_parts: tuple[object, ...],
) -> dict[str, object]:
    selection_values = oriented(column, values)
    seed = stable_seed(args.seed, *seed_parts, column)
    full_rho, pvalue = coefficient(selection_values, target)
    full_boot = bootstrap_spearman(
        selection_values, target, args.bootstrap_repeats,
        np.random.default_rng(stable_seed(seed, "full-bootstrap")),
    )
    family_samples = family_sample_rhos(
        selection_values, target, families, args.family_repeats,
        np.random.default_rng(stable_seed(seed, "family")),
    )
    regrets = regret_samples(
        selection_values, target, args.shortlist_size, args.regret_repeats,
        np.random.default_rng(stable_seed(seed, "regret")),
    )
    return {
        "n": len(values),
        "n_families": len(set(families)),
        "selection_direction": "min" if column in LOWER_IS_BETTER else "max",
        "protocol_1_full_spearman": {
            "rho": safe_number(full_rho), "pvalue": safe_number(pvalue),
            "bootstrap_95_ci": percentile_interval(full_boot),
        },
        "protocol_2_one_per_family_spearman": simulation_summary(
            family_samples, args.bootstrap_repeats,
            np.random.default_rng(stable_seed(seed, "family-bootstrap")),
        ),
        "protocol_3_top1_regret_k5": simulation_summary(
            regrets, args.bootstrap_repeats,
            np.random.default_rng(stable_seed(seed, "regret-bootstrap")),
        ),
        "family_degeneracy": family_diagnostics(selection_values, target, families),
        "seed": seed,
    }


def unavailable_candidate(column: str, n: int, reason: str) -> dict[str, object]:
    simulation = {
        "mean": None, "std": None, "bootstrap_95_ci": [None, None],
        "sampling_95_interval": [None, None],
    }
    return {
        "n": n, "n_families": None,
        "selection_direction": "min" if column in LOWER_IS_BETTER else "max",
        "unavailable_reason": reason,
        "protocol_1_full_spearman": {
            "rho": None, "pvalue": None, "bootstrap_95_ci": [None, None],
        },
        "protocol_2_one_per_family_spearman": dict(simulation),
        "protocol_3_top1_regret_k5": dict(simulation),
        "family_degeneracy": {
            "F_between_within": None, "within_family_n_ge_4": {},
        },
        "seed": None,
    }


def confound_result(
    x: np.ndarray, y: np.ndarray, args: argparse.Namespace, label: str
) -> dict[str, object]:
    observed, pvalue = coefficient(x, y)
    seed = stable_seed(args.seed, "confound", label)
    boot = bootstrap_spearman(
        x, y, args.bootstrap_repeats,
        np.random.default_rng(seed),
    )
    return {
        "n": len(x), "rho": safe_number(observed), "pvalue": safe_number(pvalue),
        "bootstrap_95_ci": percentile_interval(boot),
        "abs_rho_gt_0_6": None if not math.isfinite(observed) else abs(observed) > 0.6,
        "seed": seed,
    }


def create_lift_figure(
    metric_rows: dict[str, dict[str, dict[str, str]]],
    target_rows: dict[str, dict[str, str]],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ms = np.asarray((8, 16, 32, 64, 128))
    families = sorted({row["family"] for row in target_rows.values()})
    cmap = plt.get_cmap("tab20")
    colors = {family: cmap(index % 20) for index, family in enumerate(families)}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for axis, domain in zip(axes, DOMAINS):
        labeled: set[str] = set()
        for name, row in sorted(metric_rows.get(domain, {}).items()):
            target_row = target_rows.get(name)
            if target_row is None:
                continue
            values = [finite_float(row.get(f"Lift_{m}")) for m in ms]
            if any(value is None for value in values):
                continue
            family = target_row["family"]
            label = family if family not in labeled else None
            labeled.add(family)
            axis.plot(np.log2(ms), values, color=colors[family], alpha=0.38, lw=1.0, label=label)
        axis.axhline(1.0, color="black", ls="--", lw=1.0)
        axis.set_title(domain)
        axis.set_xlabel("log2(m)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Lift(m)")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def format_value(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def create_report(payload: dict[str, object], path: Path, figure: Path) -> None:
    lines = ["# E3：全池三协议评估", ""]
    targets = payload["evaluations"]
    for target_name in (TARGET_AVG, TARGET_PC1):
        lines.extend((f"## {target_name}", ""))
        for domain in DOMAINS:
            lines.extend((f"### {domain}", ""))
            lines.append("| 指标 | 全表 Spearman | 一族一个 Spearman | top-1 regret (k=5) |")
            lines.append("|---|---:|---:|---:|")
            result = targets[domain][target_name]["candidates"]
            for metric in (*LAR_METRICS, *BASELINES):
                row = result.get(metric)
                if row is None:
                    lines.append(f"| {metric} | NA | NA | NA |")
                    continue
                p1 = row["protocol_1_full_spearman"]
                p2 = row["protocol_2_one_per_family_spearman"]
                p3 = row["protocol_3_top1_regret_k5"]
                lines.append(
                    f"| {metric} | {format_value(p1['rho'])} | "
                    f"{format_value(p2['mean'])} ± {format_value(p2['std'])} | "
                    f"{format_value(p3['mean'])} |"
                )
            lines.append("")
            lines.append("组合 regret：" + ", ".join(
                f"{name}={format_value(row['mean'])}"
                for name, row in targets[domain][target_name]["combinations"].items()
            ))
            lines.append("")

    relative_figure = Path(figure).name
    lines.extend(("## Lift(m) 曲线", "", f"![Lift curves]({relative_figure})", ""))
    lines.extend(("## 混杂检查", ""))
    for domain in DOMAINS:
        checks = payload["confounds"][domain]
        lines.append(
            f"- {domain}: rho(d, Lift_64)={format_value(checks['dimension_vs_Lift_64']['rho'])}; "
            f"rho(d, m50)={format_value(checks['dimension_vs_m50']['rho'])}; "
            f"rho(n_tokens, Lift_64)={format_value(checks['n_tokens_vs_Lift_64']['rho'])}."
        )
    lines.extend(("", "## 停止判据", ""))
    for target_name in (TARGET_AVG, TARGET_PC1):
        for domain in DOMAINS:
            stop = payload["stop_rule"][domain][target_name]
            lines.append(
                f"- {domain}/{target_name}: (1)={stop['condition_1_all_abs_rho_below_0_5']}, "
                f"(2)={stop['condition_2_one_per_family_rho_below_0_6']}, "
                f"(3)={stop['condition_3_probe_plus_lift_no_improvement']}; "
                f"路线失败={stop['route_failed']}。"
            )
    lines.extend(("", "所有置信区间、家族 F、n>=4 族内相关和覆盖审计见 `e3.json`。", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if min(
        args.family_repeats, args.regret_repeats, args.combo_repeats,
        args.bootstrap_repeats, args.shortlist_size,
    ) < 1:
        raise ValueError("All repeat counts and shortlist size must be positive")

    targets_list = read_csv(args.targets)
    required = {
        "name", "family", TARGET_AVG, "qwen3", "qwen2_5", "smollm2",
        "probe_epoch1", "retrieval-ImageNet", "CKA", "pretrain_loss", "A_score",
    }
    columns = set(targets_list[0]) if targets_list else set()
    missing_columns = sorted(required - columns)
    if missing_columns and not args.allow_missing_columns:
        raise RuntimeError(f"Missing required target/baseline columns: {missing_columns}")
    if not targets_list:
        raise RuntimeError(f"No rows in {args.targets}")
    target_rows = {row["name"]: row for row in targets_list if row.get("name")}
    if len(target_rows) != len([row for row in targets_list if row.get("name")]):
        raise RuntimeError("Duplicate names in target table")
    for name, row in target_rows.items():
        if not row.get("family", "").strip():
            raise RuntimeError(f"Missing family for {name}")
        row["family"] = row["family"].strip().lower()

    pc1_values, pc1_metadata = compute_pc1(targets_list)
    target_values = {
        TARGET_AVG: {
            name: value for name, row in target_rows.items()
            if (value := finite_float(row.get(TARGET_AVG))) is not None
        },
        TARGET_PC1: pc1_values,
    }

    metric_rows: dict[str, dict[str, dict[str, str]]] = {domain: {} for domain in DOMAINS}
    unexpected_metric_rows = []
    for row in read_csv(args.metrics):
        domain = row.get("text_domain", "")
        if row.get("image_set") != args.image_set or domain not in DOMAINS:
            continue
        name = row.get("name", "")
        if name in metric_rows[domain]:
            raise RuntimeError(f"Duplicate metric row: {name}/{domain}/{args.image_set}")
        metric_rows[domain][name] = row
        if name not in target_rows:
            unexpected_metric_rows.append(f"{name}/{domain}")
    missing_metric_rows = {
        domain: sorted(set(target_rows) - set(metric_rows[domain])) for domain in DOMAINS
    }
    target_value_missing = {
        column: sorted(
            name for name, row in target_rows.items()
            if finite_float(row.get(column)) is None
        )
        for column in required - {"name", "family"}
    }
    metric_required = {
        "d", "n_tokens", "N", "K", *LAR_METRICS, "RankMe", "eff_rank",
    }
    metric_value_missing = {
        domain: {
            column: sorted(
                name for name, row in metric_rows[domain].items()
                if finite_float(row.get(column)) is None
            )
            for column in metric_required
        }
        for domain in DOMAINS
    }
    protocol_violations = []
    for domain in DOMAINS:
        for name, row in metric_rows[domain].items():
            n = finite_float(row.get("N"))
            d = finite_float(row.get("d"))
            k = finite_float(row.get("K"))
            if n != 4618:
                protocol_violations.append(f"{name}/{domain}: N={n}, expected 4618")
            if d is not None and k != min(int(d), 512):
                protocol_violations.append(
                    f"{name}/{domain}: K={k}, expected min(d,512)={min(int(d), 512)}"
                )
    incomplete_values = any(target_value_missing.values()) or any(
        names
        for domain_values in metric_value_missing.values()
        for names in domain_values.values()
    )
    metric_pool_invalid = (
        any(missing_metric_rows.values()) or bool(unexpected_metric_rows)
        or any(
            names
            for domain_values in metric_value_missing.values()
            for names in domain_values.values()
        )
        or bool(protocol_violations)
    )
    if (args.strict_metric_pool and metric_pool_invalid) or (
        args.strict_pool and (metric_pool_invalid or incomplete_values)
    ):
        raise RuntimeError(
            "Metric/target pool mismatch: "
            f"missing_rows={missing_metric_rows}, unexpected={unexpected_metric_rows}, "
            f"missing_target_values={target_value_missing}, "
            f"missing_metric_values={metric_value_missing}, "
            f"protocol_violations={protocol_violations}"
        )

    evaluations: dict[str, object] = {}
    for domain in DOMAINS:
        evaluations[domain] = {}
        for target_name in (TARGET_AVG, TARGET_PC1):
            candidates: dict[str, object] = {}
            outcomes = target_values[target_name]
            for column in (*LAR_METRICS, *BASELINES):
                values = []
                y = []
                families = []
                for name, target_value in outcomes.items():
                    metric_row = metric_rows[domain].get(name)
                    target_row = target_rows.get(name)
                    if metric_row is None or target_row is None:
                        continue
                    source = metric_row if column in (*LAR_METRICS, "RankMe", "eff_rank") else target_row
                    value = finite_float(source.get(column))
                    if value is None:
                        continue
                    values.append(value)
                    y.append(target_value)
                    families.append(target_row["family"])
                if len(values) < 3:
                    candidates[column] = unavailable_candidate(
                        column, len(values), "fewer than three complete rows"
                    )
                else:
                    candidates[column] = evaluate_candidate(
                        column, np.asarray(values), np.asarray(y), families, args,
                        (domain, target_name),
                    )

            combinations: dict[str, object] = {}
            for combo_name, feature_names in COMBINATIONS.items():
                feature_rows = []
                y = []
                for name, target_value in outcomes.items():
                    metric_row = metric_rows[domain].get(name)
                    target_row = target_rows.get(name)
                    if metric_row is None or target_row is None:
                        continue
                    row_values = []
                    for feature in feature_names:
                        source = metric_row if feature in LAR_METRICS else target_row
                        value = finite_float(source.get(feature))
                        if value is None:
                            break
                        row_values.append(value)
                    else:
                        feature_rows.append(row_values)
                        y.append(target_value)
                features = np.asarray(feature_rows, dtype=np.float64)
                outcome = np.asarray(y, dtype=np.float64)
                combo_seed = stable_seed(args.seed, domain, target_name, combo_name)
                samples = standardized_combo_regret_samples(
                    features, outcome, args.shortlist_size, args.combo_repeats,
                    np.random.default_rng(combo_seed),
                )
                combinations[combo_name] = {
                    "n": len(outcome),
                    "seed": combo_seed,
                    **simulation_summary(
                        samples, args.bootstrap_repeats,
                        np.random.default_rng(
                            stable_seed(args.seed, domain, target_name, combo_name, "bootstrap")
                        ),
                    ),
                    "fit_protocol": "standardize on non-held-out rows, fit OLS, predict 5 held-out rows",
                }
            evaluations[domain][target_name] = {
                "n_target": len(outcomes), "candidates": candidates,
                "combinations": combinations,
            }

    confounds: dict[str, object] = {}
    for domain in DOMAINS:
        checks = {}
        for label, x_column, y_column in (
            ("dimension_vs_Lift_64", "d", "Lift_64"),
            ("dimension_vs_m50", "d", "m50"),
            ("n_tokens_vs_Lift_64", "n_tokens", "Lift_64"),
        ):
            pairs = []
            for row in metric_rows[domain].values():
                x = finite_float(row.get(x_column))
                y = finite_float(row.get(y_column))
                if x is not None and y is not None:
                    pairs.append((x, y))
            checks[label] = (
                confound_result(
                    np.asarray([pair[0] for pair in pairs]),
                    np.asarray([pair[1] for pair in pairs]), args, f"{domain}:{label}",
                )
                if len(pairs) >= 3
                else {"n": len(pairs), "rho": None, "pvalue": None,
                      "bootstrap_95_ci": [None, None], "abs_rho_gt_0_6": None}
            )
        confounds[domain] = checks

    stop_rule: dict[str, object] = {}
    for domain in DOMAINS:
        stop_rule[domain] = {}
        for target_name in (TARGET_AVG, TARGET_PC1):
            result = evaluations[domain][target_name]
            candidates = result["candidates"]
            rhos = {
                metric: (
                    candidates.get(metric, {})
                    .get("protocol_1_full_spearman", {})
                    .get("rho")
                )
                for metric in ("Lift_64", "m50", "VSA")
            }
            condition_1 = (
                None if any(value is None for value in rhos.values())
                else all(abs(float(value)) < 0.5 for value in rhos.values())
            )
            family_rho = (
                candidates.get("Lift_64", {})
                .get("protocol_2_one_per_family_spearman", {})
                .get("mean")
            )
            condition_2 = None if family_rho is None else float(family_rho) < 0.6
            combos = result["combinations"]
            base = combos.get("probe_epoch1", {}).get("mean")
            enhanced = combos.get("probe_epoch1+Lift_64", {}).get("mean")
            improvement = (
                None if base is None or enhanced is None else float(base) - float(enhanced)
            )
            condition_3 = None if improvement is None else improvement <= 0
            conditions = (condition_1, condition_2, condition_3)
            stop_rule[domain][target_name] = {
                "full_spearman_rhos": rhos,
                "condition_1_all_abs_rho_below_0_5": condition_1,
                "one_per_family_Lift_64_rho": family_rho,
                "condition_2_one_per_family_rho_below_0_6": condition_2,
                "probe_regret": base,
                "probe_plus_Lift_64_regret": enhanced,
                "regret_improvement": improvement,
                "condition_3_probe_plus_lift_no_improvement": condition_3,
                "route_failed": (
                    None if any(value is None for value in conditions) else all(conditions)
                ),
            }

    payload = {
        "schema_version": 2,
        "image_set": args.image_set,
        "seeds": {
            "base": args.seed,
            "derivation": "blake2b(base, domain, target, metric, protocol)",
        },
        "repeats": {
            "family": args.family_repeats, "regret": args.regret_repeats,
            "combination": args.combo_repeats, "bootstrap": args.bootstrap_repeats,
        },
        "shortlist_size": args.shortlist_size,
        "target_metadata": {
            TARGET_AVG: {"column": TARGET_AVG, "definition": "qwen2.5 1.5B MLLM Avg"},
            TARGET_PC1: pc1_metadata,
        },
        "coverage": {
            "n_targets": len(target_rows),
            "metrics_by_domain": {domain: len(rows) for domain, rows in metric_rows.items()},
            "missing_metric_rows": missing_metric_rows,
            "unexpected_metric_rows": sorted(unexpected_metric_rows),
            "missing_requested_columns": missing_columns,
            "missing_target_values": target_value_missing,
            "missing_metric_values": metric_value_missing,
            "protocol_violations": protocol_violations,
        },
        "evaluations": evaluations,
        "confounds": confounds,
        "stop_rule": stop_rule,
        "route_failed_for_both_domains": {},
    }
    for target_name in (TARGET_AVG, TARGET_PC1):
        decisions = [stop_rule[domain][target_name]["route_failed"] for domain in DOMAINS]
        payload["route_failed_for_both_domains"][target_name] = (
            None if any(decision is None for decision in decisions) else all(decisions)
        )
    rendered = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    write_json(args.output, rendered)
    create_lift_figure(metric_rows, target_rows, args.figure)
    create_report(payload, args.report, args.figure)
    print(
        json.dumps(
            {"e3": str(args.output), "report": str(args.report), "figure": str(args.figure)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
