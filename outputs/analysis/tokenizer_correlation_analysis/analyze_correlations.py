#!/usr/bin/env python3
"""Analyze tokenizer-baseline correlations with two downstream MLLM gold settings."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
FULLSHOT_SOURCE = WORKSPACE / "outputs/tokenizer_rankings_11benchmarks_v1/scores.csv"
KSHOT_SOURCE = WORKSPACE / "outputs/imagenet_kshot_linear_clip_paper_v1/summary_seed0.csv"
GOLD_SOURCE = HERE / "mllm_gold_scores.csv"

TOKENIZERS = ["unitok", "vilau", "toklips", "toklipl", "metaclip"]
TOKENIZER_LABELS = {
    "unitok": "UniTok",
    "vilau": "VILA-U",
    "toklips": "TokLIP-S",
    "toklipl": "TokLIP-L",
    "metaclip": "MetaCLIP",
}
GOLD_METRICS = ["MMMU_TEST", "MMBench", "VQAv2_VAL", "VizWiz", "MSCOCO", "FLICKR30K"]
ALL_GOLD_METRICS = GOLD_METRICS + ["OverallMean"]
GOLD_LABELS = {
    "MMMU_TEST": "MMMU",
    "MMBench": "MMBench",
    "VQAv2_VAL": "VQAv2",
    "VizWiz": "VizWiz",
    "MSCOCO": "MSCOCO",
    "FLICKR30K": "Flickr30K",
    "OverallMean": "Overall",
}
SCALE_LABELS = {"150k": "150k", "mix665k": "mix665k"}

CORE_BASELINES = [
    ("cifar100_full", "CIFAR100 full-shot", "full-shot"),
    ("food101_full", "Food101 full-shot", "full-shot"),
    ("oxford_pets_full", "OxfordPets full-shot", "full-shot"),
    ("flowers102_full", "Flowers102 full-shot", "full-shot"),
    ("stanford_cars_full", "StanfordCars full-shot", "full-shot"),
    ("fgvc_aircraft_full", "FGVCAircraft full-shot", "full-shot"),
    ("dtd_full", "DTD full-shot", "full-shot"),
    ("sun397_full", "SUN397 full-shot", "full-shot"),
    ("caltech101_full", "Caltech101 full-shot", "full-shot"),
    ("imagenet_full_legacy", "ImageNet full-shot (DINOv2)", "full-shot"),
    ("imagenet_1shot", "ImageNet 1-shot", "k-shot"),
    ("imagenet_2shot", "ImageNet 2-shot", "k-shot"),
    ("imagenet_4shot", "ImageNet 4-shot", "k-shot"),
    ("imagenet_8shot", "ImageNet 8-shot", "k-shot"),
    ("imagenet_16shot", "ImageNet 16-shot", "k-shot"),
    ("voc2007_multilabel", "VOC2007 multi-label", "multi-label"),
]
CORE_SCOPE = "core16"

FULLSHOT_COLUMNS = {
    "cifar100_full": "cifar100_top1",
    "food101_full": "food101_top1",
    "oxford_pets_full": "oxford_pets_top1",
    "flowers102_full": "flowers102_test_top1",
    "stanford_cars_full": "stanford_cars_top1",
    "fgvc_aircraft_full": "fgvc_aircraft_top1",
    "dtd_full": "dtd_top1",
    "sun397_full": "sun397_test_top1",
    "caltech101_full": "caltech101_top1",
    "voc2007_multilabel": "voc2007_test_map_11point",
}
LEGACY_IMAGENET_PATHS = {
    "unitok": "outputs/vae_linear_probing/unitok/results_eval_linear.json",
    "vilau": "outputs/vae_linear_probing/vilau_7b_256_semantic_penultimate/results_eval_linear.json",
    "toklips": "outputs/vae_linear_probing/toklip_s_semantic_256/results_eval_linear.json",
    "toklipl": "outputs/vae_linear_probing/toklip_l_semantic_384/results_eval_linear.json",
    "metaclip": "outputs/vae_linear_probing/metaclip_b16_2pt5b/results_eval_linear.json",
}
DINO_FULL_10EPOCH_ITERATIONS = (12_499, 12_500)


def _iter_json_records(path: Path) -> list[tuple[int, dict]]:
    records = []
    iteration = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("iter:"):
            iteration = int(line.split(":", 1)[1].strip())
        elif line.startswith("{"):
            if iteration is None:
                raise ValueError(f"JSON record without iteration in {path}")
            records.append((iteration, json.loads(line)))
    if not records:
        raise ValueError(f"No iteration/JSON records in {path}")
    return records


def load_dino_full_10epoch_scores() -> tuple[dict[str, float], pd.DataFrame]:
    scores = {}
    audit_rows = []
    for tokenizer, relative_path in LEGACY_IMAGENET_PATHS.items():
        path = WORKSPACE / relative_path
        records = _iter_json_records(path)
        candidates = [item for item in records if item[0] in DINO_FULL_10EPOCH_ITERATIONS]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one 10-epoch record at iter 12499/12500 in {path}, got {len(candidates)}"
            )
        selected_iteration, selected_record = candidates[0]
        last_iteration, last_record = records[-1]
        selected_score = 100.0 * float(selected_record["best_classifier"]["accuracy"])
        last_score = 100.0 * float(last_record["best_classifier"]["accuracy"])
        scores[tokenizer] = selected_score
        audit_rows.append(
            {
                "tokenizer": tokenizer,
                "tokenizer_label": TOKENIZER_LABELS[tokenizer],
                "selected_iteration": selected_iteration,
                "selected_10epoch_top1": selected_score,
                "last_stored_iteration": last_iteration,
                "last_stored_top1": last_score,
                "source_path": relative_path,
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit["selected_10epoch_rank"] = audit["selected_10epoch_top1"].rank(
        ascending=False, method="average"
    )
    return scores, audit


def load_baseline_scores() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(FULLSHOT_SOURCE).set_index("tokenizer")
    kshot = pd.read_csv(KSHOT_SOURCE)
    scores: dict[str, dict[str, float]] = {tokenizer: {} for tokenizer in TOKENIZERS}

    for tokenizer in TOKENIZERS:
        for baseline_id, source_column in FULLSHOT_COLUMNS.items():
            scores[tokenizer][baseline_id] = float(full.loc[tokenizer, source_column])
    for row in kshot.itertuples(index=False):
        scores[row.model][f"imagenet_{int(row.shot)}shot"] = float(row.top1)
    dino_full_scores, dino_full_audit = load_dino_full_10epoch_scores()
    for tokenizer, score in dino_full_scores.items():
        scores[tokenizer]["imagenet_full_legacy"] = score

    all_metadata = CORE_BASELINES
    rows = []
    for tokenizer in TOKENIZERS:
        row = {"tokenizer": tokenizer, "tokenizer_label": TOKENIZER_LABELS[tokenizer]}
        row.update(scores[tokenizer])
        rows.append(row)
    wide = pd.DataFrame(rows)

    tidy_rows = []
    for order, (baseline_id, name, kind) in enumerate(all_metadata, start=1):
        for tokenizer in TOKENIZERS:
            tidy_rows.append(
                {
                    "scope": CORE_SCOPE,
                    "baseline_order": order,
                    "baseline_id": baseline_id,
                    "baseline_name": name,
                    "baseline_kind": kind,
                    "tokenizer": tokenizer,
                    "score": scores[tokenizer][baseline_id],
                }
            )
    return wide, pd.DataFrame(tidy_rows), dino_full_audit


def compute_correlations(tidy: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metadata = CORE_BASELINES
    for baseline_order, (baseline_id, baseline_name, baseline_kind) in enumerate(metadata, start=1):
        baseline = (
            tidy[tidy["baseline_id"] == baseline_id]
            .set_index("tokenizer")
            .loc[TOKENIZERS, "score"]
            .to_numpy(dtype=float)
        )
        for scale in SCALE_LABELS:
            gold_scale = gold[gold["scale"] == scale].set_index("tokenizer").loc[TOKENIZERS]
            for metric in ALL_GOLD_METRICS:
                rho, pvalue = spearmanr(baseline, gold_scale[metric].to_numpy(dtype=float))
                rows.append(
                    {
                        "scope": CORE_SCOPE,
                        "baseline_order": baseline_order,
                        "baseline_id": baseline_id,
                        "baseline_name": baseline_name,
                        "baseline_kind": baseline_kind,
                        "mllm_scale": scale,
                        "gold_metric": metric,
                        "rho": float(rho),
                        "pvalue_two_sided_asymptotic": float(pvalue),
                        "n_tokenizers": len(TOKENIZERS),
                    }
                )
    return pd.DataFrame(rows)


def validate_inputs(baseline_wide: pd.DataFrame, baseline_tidy: pd.DataFrame, gold: pd.DataFrame) -> None:
    if set(gold["scale"]) != set(SCALE_LABELS):
        raise ValueError(f"Unexpected MLLM settings: {sorted(set(gold['scale']))}")
    for scale in SCALE_LABELS:
        scale_rows = gold[gold["scale"] == scale]
        if set(scale_rows["tokenizer"]) != set(TOKENIZERS) or len(scale_rows) != len(TOKENIZERS):
            raise ValueError(f"MLLM setting {scale} must contain each tokenizer exactly once")
        recomputed = scale_rows[GOLD_METRICS].mean(axis=1)
        if not np.allclose(recomputed, scale_rows["OverallMean"], atol=0.011, rtol=0):
            raise ValueError(f"Screenshot OverallMean values do not match the six-task means for {scale}")
    if len(baseline_wide) != len(TOKENIZERS) or set(baseline_wide["tokenizer"]) != set(TOKENIZERS):
        raise ValueError("Baseline wide table must contain each tokenizer exactly once")
    expected_tidy_rows = len(CORE_BASELINES) * len(TOKENIZERS)
    if len(baseline_tidy) != expected_tidy_rows:
        raise ValueError(f"Expected {expected_tidy_rows} tidy baseline rows, got {len(baseline_tidy)}")


def compute_rank_tables(baseline_wide: pd.DataFrame, gold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_columns = [item[0] for item in CORE_BASELINES]
    baseline_ranks = baseline_wide[["tokenizer", "tokenizer_label"] + baseline_columns].copy()
    baseline_ranks[baseline_columns] = baseline_ranks[baseline_columns].rank(
        axis=0, ascending=False, method="average"
    )
    gold_rank_parts = []
    for scale in SCALE_LABELS:
        part = gold[gold["scale"] == scale].copy()
        part[ALL_GOLD_METRICS] = part[ALL_GOLD_METRICS].rank(
            axis=0, ascending=False, method="average"
        )
        gold_rank_parts.append(part)
    return baseline_ranks, pd.concat(gold_rank_parts, ignore_index=True)


def compute_summary(correlations: pd.DataFrame, scope: str = CORE_SCOPE) -> pd.DataFrame:
    subset = correlations[correlations["scope"] == scope]
    rows = []
    for baseline_id, group in subset.groupby("baseline_id", sort=False):
        first = group.iloc[0]
        item = {
            "baseline_order": int(first["baseline_order"]),
            "baseline_id": baseline_id,
            "baseline_name": first["baseline_name"],
            "baseline_kind": first["baseline_kind"],
        }
        scale_task_macros = []
        scale_overalls = []
        for scale in SCALE_LABELS:
            scale_group = group[group["mllm_scale"] == scale]
            task_macro = float(scale_group[scale_group["gold_metric"].isin(GOLD_METRICS)]["rho"].mean())
            overall = float(scale_group[scale_group["gold_metric"] == "OverallMean"]["rho"].iloc[0])
            item[f"{scale}_task_macro_rho"] = task_macro
            item[f"{scale}_overall_rho"] = overall
            scale_task_macros.append(task_macro)
            scale_overalls.append(overall)
        task_values = group[group["gold_metric"].isin(GOLD_METRICS)]["rho"]
        item["two_scale_task_macro_rho"] = float(np.mean(scale_task_macros))
        item["two_scale_overall_mean_rho"] = float(np.mean(scale_overalls))
        item["min_task_rho"] = float(task_values.min())
        item["max_task_rho"] = float(task_values.max())
        rows.append(item)
    summary = pd.DataFrame(rows).sort_values(
        ["two_scale_task_macro_rho", "two_scale_overall_mean_rho"], ascending=False
    )
    summary.insert(0, "correlation_rank", np.arange(1, len(summary) + 1))
    return summary


def compute_gold_agreement(gold: pd.DataFrame) -> pd.DataFrame:
    indexed = {scale: gold[gold["scale"] == scale].set_index("tokenizer").loc[TOKENIZERS] for scale in SCALE_LABELS}
    rows = []
    for metric in ALL_GOLD_METRICS:
        rho, pvalue = spearmanr(indexed["150k"][metric], indexed["mix665k"][metric])
        rows.append(
            {
                "gold_metric": metric,
                "metric_label": GOLD_LABELS[metric],
                "rho_150k_vs_mix665k": float(rho),
                "pvalue_two_sided_asymptotic": float(pvalue),
            }
        )
    return pd.DataFrame(rows)


def best_baselines_by_metric(correlations: pd.DataFrame) -> pd.DataFrame:
    core = correlations[correlations["scope"] == CORE_SCOPE]
    rows = []
    for metric in ALL_GOLD_METRICS:
        item = {"gold_metric": GOLD_LABELS[metric]}
        for scale in SCALE_LABELS:
            group = core[(core["gold_metric"] == metric) & (core["mllm_scale"] == scale)]
            maximum = group["rho"].max()
            winners = group[np.isclose(group["rho"], maximum)]["baseline_name"].tolist()
            item[f"{scale}_best_baseline"] = ", ".join(winners)
            item[f"{scale}_best_rho"] = float(maximum)
        rows.append(item)
    return pd.DataFrame(rows)


def _matrix(correlations: pd.DataFrame, scale: str, scope: str = CORE_SCOPE) -> pd.DataFrame:
    subset = correlations[(correlations["scope"] == scope) & (correlations["mllm_scale"] == scale)]
    matrix = subset.pivot(index="baseline_name", columns="gold_metric", values="rho")
    ordered_names = [name for _, name, _ in CORE_BASELINES]
    return matrix.loc[ordered_names, ALL_GOLD_METRICS]


def plot_heatmaps(correlations: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 10.5), facecolor="#F8FAFC", sharey=True)
    for ax, scale in zip(axes, SCALE_LABELS):
        matrix = _matrix(correlations, scale)
        image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(SCALE_LABELS[scale], fontsize=16, fontweight="bold", pad=12)
        ax.set_xticks(range(len(ALL_GOLD_METRICS)), [GOLD_LABELS[item] for item in ALL_GOLD_METRICS])
        ax.tick_params(axis="x", rotation=38, labelsize=10)
        ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=10)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iat[row, column]
                color = "white" if abs(value) >= 0.55 else "#172033"
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=9, color=color)
        # Separate transfer full-shot, ImageNet full-shot, K-shot, and VOC rows.
        for boundary in (8.5, 9.5, 14.5):
            ax.axhline(boundary, color="#FFFFFF", linewidth=2.4)
        ax.spines[:].set_visible(False)
        ax.set_facecolor("white")
    fig.suptitle("16 Baselines vs Downstream MLLM Gold Scores", fontsize=20, fontweight="bold", color="#0F172A")
    colorbar_axis = fig.add_axes([0.935, 0.18, 0.015, 0.64])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Spearman rho", fontweight="bold")
    fig.subplots_adjust(left=0.23, right=0.91, top=0.89, bottom=0.12, wspace=0.08)
    fig.savefig(
        HERE / "spearman_heatmaps_corrected_10epoch.png",
        dpi=220,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def plot_summary(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values("two_scale_task_macro_rho", ascending=True)
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.8), facecolor="#F8FAFC", sharey=True)
    panels = [
        ("task_macro_rho", "Mean across six MLLM tasks"),
        ("overall_rho", "Correlation with MLLM overall mean"),
    ]
    colors = {"150k": "#2563EB", "mix665k": "#F97316"}
    for ax, (suffix, title) in zip(axes, panels):
        left = ordered[f"150k_{suffix}"].to_numpy()
        right = ordered[f"mix665k_{suffix}"].to_numpy()
        for index in range(len(y)):
            ax.plot([left[index], right[index]], [y[index], y[index]], color="#CBD5E1", linewidth=1.5, zorder=1)
        ax.scatter(left, y, s=62, color=colors["150k"], label="150k", zorder=3, edgecolor="white", linewidth=0.8)
        ax.scatter(right, y, s=62, color=colors["mix665k"], label="mix665k", zorder=3, edgecolor="white", linewidth=0.8)
        ax.axvline(0, color="#64748B", linewidth=1.0)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xticks(np.linspace(-1, 1, 5))
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Spearman rho", fontweight="bold")
        ax.set_facecolor("white")
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_yticks(y, ordered["baseline_name"], fontsize=10)
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle("Baseline–MLLM Rank-Correlation Summary", fontsize=20, fontweight="bold", color="#0F172A")
    fig.subplots_adjust(left=0.25, right=0.97, top=0.88, bottom=0.10, wspace=0.10)
    fig.savefig(HERE / "correlation_summary.png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, float_columns: set[str] | None = None) -> str:
    float_columns = float_columns or set()
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if column in float_columns and pd.notna(value):
                values.append(f"{float(value):.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    baseline_wide: pd.DataFrame,
    dino_full_audit: pd.DataFrame,
    gold: pd.DataFrame,
    correlations: pd.DataFrame,
    summary: pd.DataFrame,
    gold_agreement: pd.DataFrame,
    best: pd.DataFrame,
) -> None:
    top = summary.iloc[0]
    top_150_value = float(summary["150k_overall_rho"].max())
    top_mix_value = float(summary["mix665k_overall_rho"].max())
    top_150_names = summary[np.isclose(summary["150k_overall_rho"], top_150_value)]["baseline_name"].tolist()
    top_mix_names = summary[np.isclose(summary["mix665k_overall_rho"], top_mix_value)]["baseline_name"].tolist()
    positive_150 = int((summary["150k_task_macro_rho"] > 0).sum())
    positive_mix = int((summary["mix665k_task_macro_rho"] > 0).sum())

    gold_display = gold.copy()
    gold_display["tokenizer"] = gold_display["tokenizer"].map(TOKENIZER_LABELS)
    gold_display = gold_display.rename(columns={"scale": "MLLM setting", "tokenizer": "Tokenizer", **GOLD_LABELS})

    baseline_display = baseline_wide[["tokenizer_label"] + [item[0] for item in CORE_BASELINES]].copy()
    baseline_display = baseline_display.rename(
        columns={"tokenizer_label": "Tokenizer", **{item[0]: item[1] for item in CORE_BASELINES}}
    )

    summary_display = summary[
        [
            "correlation_rank", "baseline_name", "150k_task_macro_rho", "150k_overall_rho",
            "mix665k_task_macro_rho", "mix665k_overall_rho", "two_scale_task_macro_rho",
        ]
    ].rename(
        columns={
            "correlation_rank": "Rank", "baseline_name": "Baseline",
            "150k_task_macro_rho": "150k task macro", "150k_overall_rho": "150k overall",
            "mix665k_task_macro_rho": "mix665k task macro", "mix665k_overall_rho": "mix665k overall",
            "two_scale_task_macro_rho": "Two-setting task macro",
        }
    )
    best_display = best.rename(
        columns={
            "gold_metric": "MLLM metric", "150k_best_baseline": "150k best baseline",
            "150k_best_rho": "150k rho", "mix665k_best_baseline": "mix665k best baseline",
            "mix665k_best_rho": "mix665k rho",
        }
    )
    agreement_display = gold_agreement[["metric_label", "rho_150k_vs_mix665k"]].rename(
        columns={"metric_label": "MLLM metric", "rho_150k_vs_mix665k": "150k vs mix665k rho"}
    )
    audit_display = dino_full_audit[
        [
            "tokenizer_label", "selected_iteration", "selected_10epoch_top1",
            "selected_10epoch_rank", "last_stored_iteration", "last_stored_top1",
        ]
    ].rename(
        columns={
            "tokenizer_label": "Tokenizer", "selected_iteration": "Selected iter",
            "selected_10epoch_top1": "10-epoch Top-1", "selected_10epoch_rank": "10-epoch rank",
            "last_stored_iteration": "Last stored iter", "last_stored_top1": "Last stored Top-1",
        }
    )

    matrix_tables = []
    for scale in SCALE_LABELS:
        matrix = _matrix(correlations, scale).rename(columns=GOLD_LABELS).reset_index().rename(columns={"baseline_name": "Baseline"})
        matrix_tables.append((scale, markdown_table(matrix, set(GOLD_LABELS.values()))))

    lines = [
        "# 16 个 tokenizer baseline 与下游 MLLM 金标准的相关性分析",
        "",
        "## 结论摘要",
        "",
        f"- 按两个 MLLM 设置、12 个任务相关系数的宏平均，最高的是 **{top['baseline_name']}**（ρ={top['two_scale_task_macro_rho']:.2f}）。",
        f"- 对截图中的总体平均值，150k 相关性最高的是 **{'、'.join(top_150_names)}**（并列，ρ={top_150_value:.2f}）；mix665k 最高的是 **{'、'.join(top_mix_names)}**（ρ={top_mix_value:.2f}）。",
        f"- 16 个 baseline 中，任务宏平均为正的数量分别为：150k **{positive_150}/16**，mix665k **{positive_mix}/16**。两个 MLLM 设置给出的 tokenizer 排名并不完全一致，因此同一 baseline 在两个设置上的相关性可能明显变化。",
        "- 每个相关系数都只基于 5 个 tokenizer；结果适合比较排序趋势，不应被当作稳定的统计显著性结论。",
        "",
        "## 主要观察",
        "",
        "1. **Caltech101 full-shot 是相对最稳定的代理指标**：两套设置的六任务宏平均分别为 0.22 和 0.20；它与 mix665k Overall 的相关性达到 0.70。但它在 150k VQAv2 上为 -0.70，说明即使最佳 baseline 也不是跨任务一致的代理。",
        "2. **VOC2007 multi-label 的综合排名为第 2**，并且具有最高的 mix665k 六任务宏平均（ρ=0.28）；它对 150k VQAv2 达到 0.80、对 VizWiz 达到 0.50。这说明多标签物体语义可能比部分单标签分类任务更接近 MLLM 所需能力。",
        "3. **FGVCAircraft 对 mix665k Flickr30K 很强（ρ=0.90）**，但对同一设置的 MMMU 为 -0.70；单任务的高相关不能直接推广到总体能力。",
        "4. **ImageNet full-shot 统一取 10-epoch checkpoint（iter 12499；TokLIP 日志为等价的 12500）**。其排名为 TokLIP-L > UniTok > VILA-U > TokLIP-S > MetaCLIP；修正后的相关性与用户上周截图完全一致。",
        "5. **ImageNet 2/4/8/16-shot 的 Spearman 结果完全相同**。原因是这四个 shot 下五个 tokenizer 的相对排名没有发生变化；Spearman 只看名次，不看分差增长。1-shot 的名次略有不同，因此相关矩阵不同。",
        "6. **两套 MLLM 金标准的排序差异是主导因素**：Overall 仅 ρ=0.10，MMMU 为 -0.80，MMBench 为 0.00。若目标是预测特定训练配置，应该分别选择 baseline，而不是期待一个全局代理指标。",
        "7. **多数 baseline 的任务宏平均为负**，表明传统分类 linear probing 排名整体上不能稳定代表当前下游 MLLM 排名；这也提示语言对齐、视觉 token 接口和训练数据可能比单纯分类可分性更关键。",
        "",
        "## 分析口径",
        "",
        "16 项包含 9 个下游 full-shot linear probing、旧 DINOv2 协议的 ImageNet full-shot、5 个 ImageNet K-shot，以及 VOC2007 multi-label。所有指标方向均为越高越好。旧 ImageNet full-shot 与 K-shot 的分类器、特征组合和训练长度不同，因此它的相关性可以参与横向排名，但不能解释为 K-shot 曲线的同协议终点。",
        "",
        "对每个 baseline，取 5 个 tokenizer 的 baseline 分数向量，分别与一个 MLLM setting 下 MMMU、MMBench、VQAv2、VizWiz、MSCOCO、Flickr30K 和截图总体均值的 5-tokenizer 分数向量计算 Spearman ρ。任务宏平均只平均六个具体任务的 ρ，不包含 Overall。并列值使用平均排名。",
        "",
        "## Spearman 公式与 ImageNet full-shot 审计",
        "",
        "实际计算使用 `scipy.stats.spearmanr`：先分别把 baseline 分数和 MLLM 分数转换为排名，再计算两个排名向量的 Pearson 相关，即 `ρs = Corr(rank(x), rank(y))`。无并列排名时，它等价于 `ρs = 1 - 6 Σdᵢ² / [n(n²-1)]`；这里 `n=5`。存在并列值时使用平均排名，并采用前一个通用定义。所有数据先按 tokenizer 名称对齐，而不是依赖 CSV 行顺序。",
        "",
        "ImageNet full-shot 文件包含多个 checkpoint。此前脚本错误地读取每个文件的最后一条记录，导致混用了 10、约 96 和 100 epoch。现统一选取 10-epoch 记录：",
        "",
        markdown_table(
            audit_display,
            {"10-epoch Top-1", "10-epoch rank", "Last stored Top-1"},
        ),
        "",
        "对应的 ImageNet full-shot Spearman 为：150k `[-0.3, -0.9, -0.2, -0.7, -0.9, -0.9, -0.9]`，mix665k `[0.3, -0.1, 0.0, 0.0, -0.6, 0.1, 0.0]`，顺序为 MMMU、MMBench、VQAv2、VizWiz、MSCOCO、Flickr30K、Overall。",
        "",
        "## 三类 linear probing 的实际设置",
        "",
        "### 1. Full-shot single-label linear probing",
        "",
        "- 冻结 tokenizer/backbone，只训练线性分类头；训练使用模型原生分辨率。UniTok、VILA-U、TokLIP-S 为 256，TokLIP-L 为 384，MetaCLIP 使用其 timm 原生配置。",
        "- `batch_size=128`、`num_workers=8`；训练增强为 bicubic RandomResizedCrop 和 `p=0.5` 水平翻转，验证使用 resize + center crop。",
        "- 优化器为 SGD，`momentum=0.9`、`weight_decay=0`，cosine learning-rate schedule。并行训练 1/4-block、是否 avg-pool 和 13 个基础学习率（`1e-5` 到 `0.1`，按 global batch/256 缩放）的分类头，再按验证集选择最佳头。",
        "- 本报告的 ImageNet full-shot 固定使用 10 epochs × 1250 iterations/epoch，即 iter 12499/12500。其余 full-shot 数据集按本地 launcher 设置：CIFAR100/Food101 20 epochs，OxfordPets/StanfordCars/FGVCAircraft/Caltech101 100，Flowers102/DTD 200，SUN397 15；每个 epoch 的迭代数为 `ceil(train_samples/128)`。",
        "",
        "### 2. ImageNet few-shot linear probing",
        "",
        "- 使用平衡且嵌套的 1/2/4/8/16-shot support set；本报告取 support seed 0。ImageNet train 的固定 10%（128,116 张）只用于选择正则化，support 从其余 90% 中采样，最终只在官方 50k validation 上报告。",
        "- 特征使用确定性的模型原生 resize + center crop，不做训练增强，也不做特征 L2 normalization。`batch_size=100`、`num_workers=8` 仅用于 GPU 特征提取。",
        "- 分类器为 sklearn multinomial LogisticRegression，solver 为 L-BFGS，`max_iter=1000`、`tol=1e-4`。它是全批 CPU 求解，没有 minibatch 和 epoch 参数。",
        "- C 从 `10^-6 ... 10^6` 的 7 个初始锚点开始，并在最优区域二分到 0.125 decade 分辨率；按 selection Top-1 选择，完全并列时取更小的 C。",
        "",
        "### 3. VOC2007 multi-label linear probing",
        "",
        "- 使用官方 train/val/test（2,501/2,510/4,952），冻结相同特征面；整张图 bicubic resize 到模型原生尺寸，不做 crop、增强或特征 normalization。`batch_size=100`、`num_workers=8` 同样只用于特征提取。",
        "- 对 20 个类别分别训练二元 L2 LogisticRegression（L-BFGS），`max_iter=1000`、`tol=1e-4`；没有 epoch、学习率或分类器 minibatch。difficult 标签 0 按类别忽略。",
        "- 在 45 个 `lambda=10^[5,...,-6]` 上以 strong-to-weak 顺序 warm start，使用 validation 11-point mAP 选择一个共享 lambda，并列时取更大的 lambda；随后在 train+val 重拟合，在 test 上报告官方 VOC2007 11-point mAP。",
        "",
        "## 汇总排名",
        "",
        markdown_table(summary_display, {column for column in summary_display.columns if column not in {"Rank", "Baseline"}}),
        "",
        "![Baseline correlation summary](./correlation_summary.png)",
        "",
        "## 完整相关性热力图",
        "",
        "红色为正相关，蓝色为负相关；ρ=1 表示 tokenizer 排名完全一致，ρ=-1 表示完全相反。",
        "",
        "![Spearman heatmaps](./spearman_heatmaps_corrected_10epoch.png)",
        "",
        "## 每个 MLLM 指标对应的最佳 baseline",
        "",
        markdown_table(best_display, {"150k rho", "mix665k rho"}),
        "",
        "## 两个 MLLM 金标准自身的排名一致性",
        "",
        markdown_table(agreement_display, {"150k vs mix665k rho"}),
        "",
        "该表直接比较同一任务在 150k 与 mix665k 下的 tokenizer 排名。如果该值较低或为负，就不能期待某个视觉 baseline 同时高度预测两个设置。",
        "",
        "## MLLM 金标准原始数据",
        "",
        markdown_table(gold_display, set(GOLD_LABELS.values())),
        "",
        "## 16 项 baseline 原始分数",
        "",
        markdown_table(baseline_display, set(baseline_display.columns) - {"Tokenizer"}),
        "",
        "## 完整 Spearman 矩阵",
        "",
    ]
    for scale, table in matrix_tables:
        lines.extend([f"### {scale}", "", table, ""])
    lines.extend(
        [
            "## 数据与可复现文件",
            "",
            "- [截图转录的 MLLM 金标准](./mllm_gold_scores.csv)",
            "- [16 项 baseline 宽表](./baseline_scores.csv)",
            "- [baseline tidy 数据](./baseline_scores_tidy.csv)",
            "- [16 项 baseline 排名（1 为最好）](./baseline_ranks.csv)",
            "- [MLLM 金标准排名（1 为最好）](./mllm_gold_ranks.csv)",
            "- [ImageNet full-shot checkpoint 审计](./imagenet_full_checkpoint_audit.csv)",
            "- [全部逐项 Spearman 与 p 值](./spearman_correlations.csv)",
            "- [16 项 baseline 汇总](./baseline_correlation_summary.csv)",
            "- [每个金标准指标的最佳 baseline](./best_baseline_by_gold_metric.csv)",
            "- [两个 MLLM 设置自身的一致性](./gold_scale_agreement.csv)",
            "- [分析脚本](./analyze_correlations.py)",
            "- [Full-shot DINOv2 linear evaluator](../../dinov2/dinov2/eval/linear.py)",
            "- [Full-shot dataset/epoch 设置](../../dinov2/knn_tools/linear_dataset_common.sh)",
            "- [Few-shot 实现](../../CLIP/linear_probe_tokenizers.py)",
            "- [VOC multi-label 实现](../../CLIP/linear_probe_voc2007.py)",
            "",
            "## 解读限制",
            "",
            "1. 样本量只有 5 个 tokenizer，单次名次交换就会显著改变 ρ；报告保留 p 值数据，但不依据 p 值做强结论。",
            "2. 同一 tokenizer 在两个 MLLM 设置中的相对强弱会受训练数据规模和任务构成影响，baseline–gold 相关性不是 tokenizer 的固有常数。",
            "3. 相关性衡量排序一致性，不代表 baseline 分数与 MLLM 绝对性能之间存在因果关系。",
        ]
    )
    (HERE / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    baseline_wide, baseline_tidy, dino_full_audit = load_baseline_scores()
    gold = pd.read_csv(GOLD_SOURCE)
    validate_inputs(baseline_wide, baseline_tidy, gold)
    baseline_ranks, gold_ranks = compute_rank_tables(baseline_wide, gold)
    correlations = compute_correlations(baseline_tidy, gold)
    expected_correlations = len(CORE_BASELINES) * len(SCALE_LABELS) * len(ALL_GOLD_METRICS)
    if len(correlations) != expected_correlations:
        raise ValueError(f"Expected {expected_correlations} correlations, got {len(correlations)}")
    summary = compute_summary(correlations, CORE_SCOPE)
    gold_agreement = compute_gold_agreement(gold)
    best = best_baselines_by_metric(correlations)

    baseline_wide.to_csv(HERE / "baseline_scores.csv", index=False, float_format="%.9f")
    baseline_tidy.to_csv(HERE / "baseline_scores_tidy.csv", index=False, float_format="%.9f")
    baseline_ranks.to_csv(HERE / "baseline_ranks.csv", index=False, float_format="%.6f")
    gold_ranks.to_csv(HERE / "mllm_gold_ranks.csv", index=False, float_format="%.6f")
    dino_full_audit.to_csv(HERE / "imagenet_full_checkpoint_audit.csv", index=False, float_format="%.9f")
    correlations.to_csv(HERE / "spearman_correlations.csv", index=False, float_format="%.9f")
    summary.to_csv(HERE / "baseline_correlation_summary.csv", index=False, float_format="%.9f")
    stale_supplement = HERE / "supplement_correlation_summary.csv"
    if stale_supplement.exists():
        stale_supplement.unlink()
    gold_agreement.to_csv(HERE / "gold_scale_agreement.csv", index=False, float_format="%.9f")
    best.to_csv(HERE / "best_baseline_by_gold_metric.csv", index=False, float_format="%.9f")
    plot_heatmaps(correlations)
    plot_summary(summary)
    write_report(baseline_wide, dino_full_audit, gold, correlations, summary, gold_agreement, best)
    print(HERE / "report.md")


if __name__ == "__main__":
    main()
