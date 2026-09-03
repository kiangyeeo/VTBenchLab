#!/usr/bin/env python
"""Evaluate the E4 text-domain x metric-form scan and render its report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .compute_metrics_v3 import ABTT_METRICS, BASE_METRICS, DOMAINS
    from .data import WORKSPACE
    from .eval_common import finite_float, read_csv, write_json
    from .eval_e3 import (
        TARGET_AVG, TARGET_AVG_MATCHED, TARGET_PC1, TARGET_NAMES, compute_pc1,
        evaluate_candidate, safe_number, simulation_summary, stable_seed,
        standardized_combo_regret_samples, unavailable_candidate,
    )
    from .metric_config import load_metric_config, oriented
except ImportError:
    from compute_metrics_v3 import ABTT_METRICS, BASE_METRICS, DOMAINS
    from data import WORKSPACE
    from eval_common import finite_float, read_csv, write_json
    from eval_e3 import (
        TARGET_AVG, TARGET_AVG_MATCHED, TARGET_PC1, TARGET_NAMES, compute_pc1,
        evaluate_candidate, safe_number, simulation_summary, stable_seed,
        standardized_combo_regret_samples, unavailable_candidate,
    )
    from metric_config import load_metric_config, oriented


PRIMARY_FORMS = ("VSA", "mutual_kNN_k10", "cm_cka", "cm_r2")
BUG_METRICS = ("m50", "m90", "LAR_64")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=WORKSPACE / "lar/results/metrics_v3.csv")
    parser.add_argument("--metrics-metadata", type=Path, default=WORKSPACE / "lar/results/metrics_v3_meta.json")
    parser.add_argument("--lar-metrics", type=Path, default=WORKSPACE / "lar/results/lar_metrics_v2.csv")
    parser.add_argument("--targets", type=Path, default=WORKSPACE / "lar/configs/e3_targets.csv")
    parser.add_argument("--metrics-config", type=Path, default=WORKSPACE / "lar/configs/metrics.yaml")
    parser.add_argument("--e3", type=Path, default=WORKSPACE / "lar/results/e3.json")
    parser.add_argument("--family-repeats", type=int, default=5000)
    parser.add_argument("--regret-repeats", type=int, default=20000)
    parser.add_argument("--combo-repeats", type=int, default=4000)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--shortlist-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "lar/results/e4.json")
    parser.add_argument("--report", type=Path, default=WORKSPACE / "lar/results/e4_report.md")
    return parser.parse_args()


def target_data(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, float]], dict[str, Any], dict[str, dict[str, str]]]:
    by_name = {row["name"]: row for row in rows if row.get("name")}
    for row in by_name.values():
        row["family"] = row["family"].strip().lower()
    pc1, pc1_meta = compute_pc1(rows)
    values = {
        TARGET_AVG: {
            name: value for name, row in by_name.items()
            if (value := finite_float(row.get(TARGET_AVG))) is not None
        },
        TARGET_AVG_MATCHED: {
            name: float(by_name[name][TARGET_AVG]) for name in pc1
            if name in by_name and finite_float(by_name[name].get(TARGET_AVG)) is not None
        },
        TARGET_PC1: pc1,
    }
    if set(values[TARGET_AVG_MATCHED]) != set(values[TARGET_PC1]):
        raise RuntimeError("Matched-Avg and PC1 pools are not identical")
    metadata = {}
    for target in TARGET_NAMES:
        composition = Counter(by_name[name]["family"] for name in values[target])
        metadata[target] = {
            "n": len(values[target]), "family_composition": dict(sorted(composition.items())),
            "role": "primary" if target == TARGET_AVG else "auxiliary",
        }
    metadata[TARGET_PC1].update(pc1_meta)
    metadata[TARGET_AVG_MATCHED]["pool_identity"] = "exactly the PC1-complete pool"
    return values, metadata, by_name


def long_metrics(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {domain: {} for domain in DOMAINS}
    if not path.is_file():
        return output
    for row in read_csv(path):
        domain, name, metric = row.get("text_domain", ""), row.get("name", ""), row.get("metric_name", "")
        value = finite_float(row.get("value"))
        if domain not in output or not name or not metric or value is None:
            continue
        output[domain].setdefault(name, {})[metric] = value
    return output


def eval_one(
    metric: str, domain_rows: dict[str, dict[str, float]], outcomes: dict[str, float],
    target_rows: dict[str, dict[str, str]], args: argparse.Namespace,
    config: dict[str, dict[str, object]], seed_parts: tuple[object, ...],
) -> dict[str, Any]:
    values, y, families = [], [], []
    for name, outcome in outcomes.items():
        value = domain_rows.get(name, {}).get(metric)
        if value is None or name not in target_rows:
            continue
        values.append(value); y.append(outcome); families.append(target_rows[name]["family"])
    if len(values) < 3:
        return unavailable_candidate(metric, len(values), "fewer than three complete rows", config)
    return evaluate_candidate(
        metric, np.asarray(values), np.asarray(y), families, args, seed_parts, config,
    )


def combo_one(
    metric: str, domain_rows: dict[str, dict[str, float]], outcomes: dict[str, float],
    target_rows: dict[str, dict[str, str]], args: argparse.Namespace,
    config: dict[str, dict[str, object]], domain: str, target: str,
) -> dict[str, Any]:
    feature_rows, y, families = [], [], []
    for name, outcome in outcomes.items():
        metric_value = domain_rows.get(name, {}).get(metric)
        probe = finite_float(target_rows.get(name, {}).get("probe_epoch1"))
        if metric_value is None or probe is None:
            continue
        feature_rows.append([
            float(oriented("probe_epoch1", np.asarray([probe]), config)[0]),
            float(oriented(metric, np.asarray([metric_value]), config)[0]),
        ])
        y.append(outcome); families.append(target_rows[name]["family"])
    seed = stable_seed(args.seed, "combo", domain, target, metric)
    result = {
        "n": len(y), "n_families": len(set(families)),
        "family_composition": dict(sorted(Counter(families).items())), "seed": seed,
    }
    if len(y) <= args.shortlist_size + 1:
        empty = {"mean": None, "std": None, "bootstrap_95_ci": [None, None]}
        result.update({"probe_epoch1": empty, f"probe_epoch1+{metric}": empty})
        return result
    matrix, outcome_array = np.asarray(feature_rows), np.asarray(y)
    paired_samples = {}
    for label, features in (
        ("probe_epoch1", matrix[:, :1]), (f"probe_epoch1+{metric}", matrix),
    ):
        samples = standardized_combo_regret_samples(
            features, outcome_array, args.shortlist_size, args.combo_repeats,
            # Reinitialize the same stream so baseline/enhanced use identical
            # held-out five-model sets (a paired comparison).
            np.random.default_rng(stable_seed(seed, "holdouts")),
        )
        paired_samples[label] = samples
        result[label] = simulation_summary(
            samples, args.bootstrap_repeats,
            np.random.default_rng(stable_seed(seed, label, "bootstrap")),
        )
    base, enhanced = result["probe_epoch1"]["mean"], result[f"probe_epoch1+{metric}"]["mean"]
    result["improvement"] = None if base is None or enhanced is None else float(base) - float(enhanced)
    if base is not None and enhanced is not None:
        delta = paired_samples["probe_epoch1"] - paired_samples[f"probe_epoch1+{metric}"]
        result["paired_improvement"] = simulation_summary(
            delta, args.bootstrap_repeats,
            np.random.default_rng(stable_seed(seed, "improvement-bootstrap")),
        )
    return result


def bug_direction_audit(
    path: Path, targets: dict[str, dict[str, float]], target_rows: dict[str, dict[str, str]],
    args: argparse.Namespace, config: dict[str, dict[str, object]],
) -> dict[str, Any]:
    rows: dict[str, dict[str, dict[str, float]]] = {"caption": {}, "answer": {}}
    for row in read_csv(path):
        domain, name = row.get("text_domain"), row.get("name")
        if domain not in rows:
            continue
        rows[domain][name] = {
            metric: value for metric in BUG_METRICS
            if (value := finite_float(row.get(metric))) is not None
        }
    historical = {name: dict(settings) for name, settings in config.items()}
    # Audit the repository state before this patch, not an inferred state from
    # the large regrets: m50/m90 were already oriented to min; LAR_64 was not.
    historical["m50"]["higher_is_better"] = False
    historical["m90"]["higher_is_better"] = False
    historical["LAR_64"]["higher_is_better"] = True
    output = {
        "historical_direction": {"m50": False, "m90": False, "LAR_64": True},
        "corrected_direction": {metric: False for metric in BUG_METRICS},
        "note": (
            "Code audit found m50/m90 were already min-oriented, so their large regrets are real "
            "and must remain unchanged. LAR_64 was max-oriented and is the only numerical direction fix. "
            "All paths now read metrics.yaml."
        ),
        "results": {},
    }
    for domain in rows:
        output["results"][domain] = {}
        for target in TARGET_NAMES:
            output["results"][domain][target] = {}
            for metric in BUG_METRICS:
                output["results"][domain][target][metric] = {
                    "before": eval_one(metric, rows[domain], targets[target], target_rows, args, historical, ("bug-before", domain, target)),
                    "after": eval_one(metric, rows[domain], targets[target], target_rows, args, config, ("bug-after", domain, target)),
                }
    return output


def variance_analysis(evaluations: dict[str, Any]) -> dict[str, Any]:
    domains = []
    matrix = []
    for domain in DOMAINS:
        values = [
            evaluations[domain][TARGET_AVG]["metrics"][metric]["protocol_1_full_spearman"]["rho"]
            for metric in PRIMARY_FORMS
        ]
        if all(value is not None for value in values):
            domains.append(domain); matrix.append(values)
    if len(matrix) < 2:
        return {"available": False, "domains": domains, "reason": "fewer than two complete domains"}
    y = np.asarray(matrix, dtype=np.float64)
    var_row_by_metric = np.var(y, axis=0, ddof=1)
    var_col_by_domain = np.var(y, axis=1, ddof=1)
    grand, domain_means, metric_means = y.mean(), y.mean(1), y.mean(0)
    ss_domain = y.shape[1] * np.square(domain_means - grand).sum()
    ss_metric = y.shape[0] * np.square(metric_means - grand).sum()
    residual = y - domain_means[:, None] - metric_means[None, :] + grand
    ss_interaction = np.square(residual).sum()
    ss_total = np.square(y - grand).sum()
    shares = {
        "text_domain": safe_number(ss_domain / ss_total),
        "metric_form": safe_number(ss_metric / ss_total),
        "interaction": safe_number(ss_interaction / ss_total),
    }
    var_row, var_col = float(var_row_by_metric.mean()), float(var_col_by_domain.mean())
    return {
        "available": True, "target": TARGET_AVG, "domains": domains,
        "metrics": list(PRIMARY_FORMS), "spearman_matrix": y.tolist(),
        "var_row_definition": "fixed metric, variance across text domains; mean over metrics",
        "var_col_definition": "fixed text domain, variance across metrics; mean over domains",
        "var_row": var_row, "var_col": var_col,
        "var_row_over_var_col": None if var_col == 0 else var_row / var_col,
        "two_factor_anova_no_replication": {
            "variance_shares": shares,
            "note": "One observation per cell: interaction is the non-additive residual; no F-test is claimed.",
        },
        "supports_text_distribution_hypothesis": var_row > var_col,
    }


def mechanism(metadata_path: Path, evaluations: dict[str, Any]) -> dict[str, Any]:
    if not metadata_path.is_file():
        return {"available": False, "reason": f"missing {metadata_path}"}
    source = json.loads(metadata_path.read_text(encoding="utf-8"))
    domain_table = source.get("domain_audit", {})
    cca = {}
    for domain in DOMAINS:
        arrays = []
        per_model = {}
        for model_name, model in source.get("per_model", {}).items():
            row = model.get("domains", {}).get(domain, {})
            values = row.get("cca_first20")
            if values and len(values) >= 20:
                arrays.append(values[:20])
                per_model[model_name] = values[:20]
        cca[domain] = {
            "n_models": len(arrays),
            "per_model_first20": per_model,
            "mean_first20": None if not arrays else np.mean(arrays, axis=0).tolist(),
            "std_first20": None if not arrays else np.std(arrays, axis=0, ddof=1).tolist() if len(arrays) > 1 else [0.0] * 20,
        }
    caption = domain_table.get("caption", {})
    answer = domain_table.get("answer_other", {})
    rho_caption = evaluations["caption"][TARGET_AVG]["metrics"]["VSA"]["protocol_1_full_spearman"]["rho"]
    rho_answer = evaluations["answer_other"][TARGET_AVG]["metrics"]["VSA"]["protocol_1_full_spearman"]["rho"]
    rank_condition = (
        None if caption.get("eff_rank") is None or answer.get("eff_rank") is None
        else answer["eff_rank"] > caption["eff_rank"]
    )
    rho_condition = None if rho_caption is None or rho_answer is None else rho_answer > rho_caption
    return {
        "available": True, "text_domains": domain_table, "cca": cca,
        "hypothesis_test": {
            "answer_eff_rank_gt_caption": rank_condition,
            "answer_VSA_rho_gt_caption": rho_condition,
            "mechanism_supported_by_prespecified_rule": (
                None if rank_condition is None or rho_condition is None else rank_condition and rho_condition
            ),
        },
    }


def appendix_e3(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "reason": f"missing {path}"}
    source = json.loads(path.read_text(encoding="utf-8"))
    output = {"available": True, "source": str(path), "conclusion": "Lift/LAR route failed; appendix only", "results": {}}
    for domain in ("caption", "answer"):
        candidates = source.get("evaluations", {}).get(domain, {}).get(TARGET_AVG, {}).get("candidates", {})
        output["results"][domain] = {
            metric: candidates.get(metric) for metric in ("Lift_8", "Lift_16", "Lift_32", "Lift_64", "Lift_128", "LAR_64")
        }
    return output


def fmt(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def protocol_cell(row: dict[str, Any]) -> str:
    return "/".join((
        fmt(row["protocol_1_full_spearman"]["rho"]),
        fmt(row["protocol_2_one_per_family_spearman"]["mean"]),
        fmt(row["protocol_3_top1_regret_k5"]["mean"]),
    ))


def create_report(payload: dict[str, Any], path: Path) -> None:
    lines = ["# E4：文本域 × 指标形式", "", "## 1. 覆盖审计", ""]
    primary_n = payload["target_metadata"][TARGET_AVG]["n"]
    lines.append(f"主结论仅使用 MLLM_Avg（实际 n={primary_n}）；matched 与 PC1 仅作辅助且池完全相同。")
    if primary_n != 79:
        lines.append(f"**覆盖警告：预期主池 n=79，当前 n={primary_n}。**")
    lines.extend(("", "| 目标 | 文本域 | 指标 | n | family 构成 |", "|---|---|---|---:|---|"))
    for domain in DOMAINS:
        for target in TARGET_NAMES:
            for metric in BASE_METRICS:
                row = payload["evaluations"][domain][target]["metrics"][metric]
                composition = ", ".join(f"{k}:{v}" for k, v in row.get("family_composition", {}).items()) or "NA"
                lines.append(f"| {target} | {domain} | {metric} | {row['n']} | {composition} |")

    lines.extend(("", "## 2. 主表（MLLM_Avg）", "", "格内为：全表 rho / 一族一个 rho / top-1 regret。", ""))
    lines.append("| 文本域 | VSA | mutual-kNN k=10 | linear CKA | ridge R² |")
    lines.append("|---|---:|---:|---:|---:|")
    for domain in DOMAINS:
        metrics = payload["evaluations"][domain][TARGET_AVG]["metrics"]
        lines.append(f"| {domain} | " + " | ".join(protocol_cell(metrics[m]) for m in PRIMARY_FORMS) + " |")
    lines.extend(("", "mutual-kNN 的 k=5/20 完整数值及三个目标均保存在 `e4.json`。", ""))

    variance = payload["variance_analysis"]
    lines.extend(("## 3. 换文本还是换指标", ""))
    if variance.get("available"):
        answer = "支持" if variance["supports_text_distribution_hypothesis"] else "不支持"
        shares = variance["two_factor_anova_no_replication"]["variance_shares"]
        lines.append(
            f"var_row={fmt(variance['var_row'], 5)}，var_col={fmt(variance['var_col'], 5)}，"
            f"比值={fmt(variance['var_row_over_var_col'], 2)}：**{answer}“换文本比换指标重要”**。"
        )
        lines.append(
            f"双因素方差份额：文本域={fmt(shares['text_domain'])}，指标={fmt(shares['metric_form'])}，"
            f"交互={fmt(shares['interaction'])}。这是无重复单元格分解，不报告 F 检验。"
        )
    else:
        lines.append("NA：" + variance.get("reason", "数据不完整"))

    lines.extend(("", "## 4. 机制诊断", "", "| 文本域 | N | eff_rank | RankMe | 平均余弦 | top1 能量 | token 长度 | 唯一文本 | CCA1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"))
    mech = payload["mechanism_diagnostics"]
    for domain in DOMAINS:
        row = mech.get("text_domains", {}).get(domain, {})
        cca_values = mech.get("cca", {}).get(domain, {}).get("mean_first20")
        lines.append(
            f"| {domain} | {row.get('N', 'NA')} | {fmt(row.get('eff_rank'))} | {fmt(row.get('RankMe'))} | "
            f"{fmt(row.get('mean_pair_cosine'))} | {fmt(row.get('top1_eig_frac'))} | "
            f"{fmt(row.get('mean_token_length'))} | {row.get('unique_texts', 'NA')} | "
            f"{fmt(None if not cca_values else cca_values[0])} |"
        )
    lines.append("")
    lines.append("CCA 前 20 维的均值、标准差和模型数见 `e4.json`。")
    hypothesis = mech.get("hypothesis_test", {})
    lines.append(f"预设机制判据成立：{hypothesis.get('mechanism_supported_by_prespecified_rule')}。")
    lines.extend(("", "caption all-but-the-top 消融（全表 rho）：", ""))
    for metric in ABTT_METRICS:
        row = payload["ablations"][metric]
        lines.append(f"- {metric}: {fmt(row['protocol_1_full_spearman']['rho'])}")

    lines.extend(("", "## 5. 组合 regret", "", "| 文本域 | 指标 | probe | probe+指标 | 改善 |", "|---|---|---:|---:|---:|"))
    for domain in DOMAINS:
        for metric in BASE_METRICS:
            row = payload["evaluations"][domain][TARGET_AVG]["combinations"][metric]
            lines.append(
                f"| {domain} | {metric} | {fmt(row['probe_epoch1']['mean'])} | "
                f"{fmt(row[f'probe_epoch1+{metric}']['mean'])} | {fmt(row.get('improvement'))} |"
            )

    lines.extend(("", "## Bug 修复审计", "", "三个指标统一从 `configs/metrics.yaml` 读取方向；修前/修后三协议完整值见 `e4.json`。", "", "| 域 | 指标 | 修前 regret | 修后 regret |", "|---|---|---:|---:|"))
    audit = payload["bug_direction_audit"]["results"]
    for domain in ("caption", "answer"):
        for metric in BUG_METRICS:
            row = audit[domain][TARGET_AVG][metric]
            lines.append(
                f"| {domain} | {metric} | {fmt(row['before']['protocol_3_top1_regret_k5']['mean'])} | "
                f"{fmt(row['after']['protocol_3_top1_regret_k5']['mean'])} |"
            )

    lines.extend(("", "## 6. 附录：Lift/LAR", "", "E3 的 Lift/LAR 失败结论保留，不再作为候选指标。完整三协议值已复制进 `e4.json`。", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_metric_config(args.metrics_config)
    needed = set(BASE_METRICS) | set(ABTT_METRICS) | set(BUG_METRICS) | {"probe_epoch1"}
    missing = sorted(needed - set(config))
    if missing:
        raise RuntimeError(f"Missing direction declarations: {missing}")
    target_values, target_metadata, target_rows = target_data(read_csv(args.targets))
    metrics = long_metrics(args.metrics)
    evaluations: dict[str, Any] = {}
    for domain in DOMAINS:
        evaluations[domain] = {}
        for target in TARGET_NAMES:
            metric_results, combinations = {}, {}
            for metric in BASE_METRICS:
                metric_results[metric] = eval_one(
                    metric, metrics[domain], target_values[target], target_rows, args, config,
                    ("e4", domain, target),
                )
                combinations[metric] = combo_one(
                    metric, metrics[domain], target_values[target], target_rows, args, config,
                    domain, target,
                )
            evaluations[domain][target] = {"metrics": metric_results, "combinations": combinations}

    ablations = {
        metric: eval_one(metric, metrics["caption"], target_values[TARGET_AVG], target_rows, args, config, ("abtt",))
        for metric in ABTT_METRICS
    }
    payload = {
        "schema_version": 4,
        "seeds": {"base": args.seed, "derivation": "blake2b(base, domain, target, metric, protocol)"},
        "repeats": {"family": args.family_repeats, "regret": args.regret_repeats, "combination": args.combo_repeats, "bootstrap": args.bootstrap_repeats},
        "target_metadata": target_metadata,
        "coverage_audit": {
            "expected_primary_n": 79, "actual_primary_n": target_metadata[TARGET_AVG]["n"],
            "matched_pool_equals_pc1": set(target_values[TARGET_AVG_MATCHED]) == set(target_values[TARGET_PC1]),
        },
        "evaluations": evaluations,
        "ablations": ablations,
        "bug_direction_audit": bug_direction_audit(args.lar_metrics, target_values, target_rows, args, config),
        "variance_analysis": variance_analysis(evaluations),
        "mechanism_diagnostics": mechanism(args.metrics_metadata, evaluations),
        "e3_lift_lar_appendix": appendix_e3(args.e3),
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    write_json(args.output, rendered)
    create_report(payload, args.report)
    print(json.dumps({"e4": str(args.output), "report": str(args.report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
