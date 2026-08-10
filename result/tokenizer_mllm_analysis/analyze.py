#!/usr/bin/env python3
"""Reproducible analysis of ImageNet linear probing vs. downstream MLLM scores.

Inputs (kept untouched):
  ../Tokenizer Accuracy by Epoch.md
  ../VisualTokenizer表现 - 主表.csv
  ../VisualTokenizer表现 - MLLM详细结果.csv

Outputs are written next to this script under data/, figures/, and README.md.
The analysis deliberately treats a missing Qwen score as NA, never as zero.
"""

from __future__ import annotations

import csv
import os
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import stats


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE.parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib otherwise tries to cache fonts in a read-only home directory here.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tokenizer_mllm_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/tokenizer_mllm_xdg_cache")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


MAIN_CSV = RESULT_DIR / "VisualTokenizer表现 - 主表.csv"
DETAIL_CSV = RESULT_DIR / "VisualTokenizer表现 - MLLM详细结果.csv"
EPOCH_MD = RESULT_DIR / "Tokenizer Accuracy by Epoch.md"

TASKS = [
    "MMMU",
    "MMBench",
    "VQAv2",
    "ScienceQA",
    "ChartQA",
    "DocVQA",
    "TextVQA",
    "POPE",
    "GQA",
    "COCO",
    "Flickr",
]
BACKBONES = ["Qwen3-1.7B", "Qwen2.5-1.5B"]
SEED = 20260810
N_BOOT = 10_000
N_PERM = 50_000

FAMILY_COLORS = {
    "SigLIP2": "#0072B2",
    "MetaCLIP1": "#D55E00",
    "MetaCLIP2": "#009E73",
    "OpenAI CLIP": "#CC79A7",
    "TokLIP": "#E69F00",
    "UniTok": "#56B4E9",
    "VILA-U": "#F0C808",
    "I-JEPA": "#7A5195",
    "RAE-v2": "#8C8C8C",
    "DINOv3": "#444444",
}


def as_float(value: str | float | int | None) -> float:
    """Parse spreadsheet-style numeric cells, returning NaN for blanks/dashes."""
    if value is None:
        return float("nan")
    try:
        text = str(value).strip()
        if text in {"", "-"}:
            return float("nan")
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def is_finite(value: float) -> bool:
    return bool(np.isfinite(value))


def mean_if_complete(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)) if np.all(np.isfinite(array)) else float("nan")


def family_of(name: str) -> str:
    if name.startswith("siglip2_"):
        return "SigLIP2"
    if name.startswith("mc1_"):
        return "MetaCLIP1"
    if name.startswith("mc2_"):
        return "MetaCLIP2"
    if name.startswith("toklip_"):
        return "TokLIP"
    return {
        "clip_openai__l14": "OpenAI CLIP",
        "unitok_attn": "UniTok",
        "vilau_256": "VILA-U",
        "I-JEPA": "I-JEPA",
        "raev2": "RAE-v2",
        "dinov3": "DINOv3",
    }.get(name, name)


def short_name(name: str) -> str:
    replacements = [
        ("siglip2_", "S2 "),
        ("mc1_", "MC1 "),
        ("mc2_", "MC2 "),
        ("clip_openai__", "CLIP "),
        ("toklip_", "TokLIP "),
        ("unitok_attn", "UniTok attn"),
        ("vilau_", "VILA-U "),
    ]
    result = name
    for old, new in replacements:
        if old in result:
            result = result.replace(old, new)
            break
    return result.replace("_", " ")


def read_two_header_csv(path: Path) -> tuple[list[list[str]], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3:
        raise ValueError(f"CSV has too few rows: {path}")
    return rows[:2], rows[2:]


def read_epochs(path: Path) -> dict[str, list[float]]:
    epochs: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---") or "tokenizer" in line:
            continue
        cells = [
            cell.strip().replace("\\_", "_").replace("\\.", ".")
            for cell in line.strip("|").split("|")
        ]
        if len(cells) != 11:
            raise ValueError(f"Unexpected epoch row: {line}")
        name = cells[0]
        if name in epochs:
            raise ValueError(f"Duplicate tokenizer in epoch table: {name}")
        epochs[name] = [float(value) for value in cells[1:]]
    return epochs


def load_records() -> tuple[list[dict], dict[str, list[float]], dict]:
    _, main_rows = read_two_header_csv(MAIN_CSV)
    _, detail_rows = read_two_header_csv(DETAIL_CSV)
    epochs = read_epochs(EPOCH_MD)

    main_by_name = {row[0]: row for row in main_rows}
    detail_by_name = {row[0]: row for row in detail_rows}
    if len(main_by_name) != len(main_rows) or len(detail_by_name) != len(detail_rows):
        raise ValueError("Duplicate tokenizer name found in a CSV")
    if set(main_by_name) != set(detail_by_name):
        raise ValueError("Tokenizer names differ between the two CSV files")
    if not set(epochs).issubset(main_by_name):
        raise ValueError("Epoch table contains tokenizer names absent from the CSV files")

    records: list[dict] = []
    epoch10_mismatches: list[str] = []
    avg_mismatches: list[dict] = []

    for name, main in main_by_name.items():
        detail = detail_by_name[name]
        if main[1] != detail[1]:
            raise ValueError(f"Family/type mismatch for {name}")

        q3_tasks = [as_float(value) for value in detail[2:13]]
        q25_tasks = [as_float(value) for value in detail[14:25]]
        q3_reported = as_float(main[2])
        q25_reported = as_float(main[3])
        q3_detail_reported = as_float(detail[13])
        q25_detail_reported = as_float(detail[25])
        q3_recomputed = mean_if_complete(q3_tasks)
        q25_recomputed = mean_if_complete(q25_tasks)
        probe = as_float(main[5])
        raw_combined = as_float(main[4])
        fair_combined = (
            float(np.mean([q3_reported, q25_reported]))
            if is_finite(q3_reported) and is_finite(q25_reported)
            else float("nan")
        )

        if name in epochs and (not is_finite(probe) or abs(probe - epochs[name][-1]) > 1e-9):
            epoch10_mismatches.append(name)
        if is_finite(q3_reported) and is_finite(q3_detail_reported) and abs(q3_reported - q3_detail_reported) > 1e-9:
            avg_mismatches.append({"tokenizer": name, "field": "Qwen3 CSV-to-CSV", "difference": q3_reported - q3_detail_reported})
        if is_finite(q25_reported) and is_finite(q25_detail_reported) and abs(q25_reported - q25_detail_reported) > 1e-9:
            avg_mismatches.append({"tokenizer": name, "field": "Qwen2.5 CSV-to-CSV", "difference": q25_reported - q25_detail_reported})
        if is_finite(q3_recomputed) and is_finite(q3_reported) and abs(q3_recomputed - q3_reported) > 0.006:
            avg_mismatches.append({"tokenizer": name, "field": "Qwen3 task mean", "difference": q3_reported - q3_recomputed})
        if is_finite(q25_recomputed) and is_finite(q25_reported) and abs(q25_recomputed - q25_reported) > 0.006:
            avg_mismatches.append({"tokenizer": name, "field": "Qwen2.5 task mean", "difference": q25_reported - q25_recomputed})

        history_status = "full_10_epoch" if name in epochs else ("final_only" if is_finite(probe) else "missing_probe")
        mllm_status = (
            "both_complete"
            if is_finite(q3_reported) and is_finite(q25_reported)
            else "qwen3_only"
            if is_finite(q3_reported)
            else "missing_both"
        )
        record = {
            "tokenizer": name,
            "visual_token_type": main[1],
            "model_family": family_of(name),
            "probe_final": probe,
            "probe_history_status": history_status,
            "qwen3_avg_reported": q3_reported,
            "qwen3_avg_recomputed": q3_recomputed,
            "qwen25_avg_reported": q25_reported,
            "qwen25_avg_recomputed": q25_recomputed,
            "mllm_avg_reported_raw": raw_combined,
            "mllm_avg_fair": fair_combined,
            "mllm_status": mllm_status,
            "qwen3_tasks": dict(zip(TASKS, q3_tasks)),
            "qwen25_tasks": dict(zip(TASKS, q25_tasks)),
            "epochs": epochs.get(name, [float("nan")] * 10),
        }
        record["in_primary_n38"] = is_finite(probe) and is_finite(fair_combined)
        record["in_qwen3_max_n44"] = is_finite(probe) and is_finite(q3_reported)
        record["in_epoch_combined_n29"] = name in epochs and is_finite(fair_combined)
        records.append(record)

    audit = {
        "epoch10_mismatches": epoch10_mismatches,
        "avg_mismatches": avg_mismatches,
        "main_names": set(main_by_name),
    }
    return records, epochs, audit


def stable_seed(label: str) -> int:
    return (SEED + zlib.crc32(label.encode("utf-8"))) % (2**32)


def rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator


def bootstrap_corr_ci(
    x: Sequence[float],
    y: Sequence[float],
    method: str,
    label: str,
    n_boot: int = N_BOOT,
) -> tuple[float, float]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    n = len(x_array)
    if n < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(stable_seed(label + method))
    indices = rng.integers(0, n, size=(n_boot, n))
    x_samples = x_array[indices]
    y_samples = y_array[indices]
    if method == "spearman":
        x_samples = stats.rankdata(x_samples, axis=1, method="average")
        y_samples = stats.rankdata(y_samples, axis=1, method="average")
    elif method != "pearson":
        raise ValueError(f"Unknown bootstrap method: {method}")
    values = rowwise_pearson(x_samples, y_samples)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_spearman_difference(
    x: Sequence[float], y_first: Sequence[float], y_second: Sequence[float], label: str
) -> tuple[float, float, float]:
    """Paired bootstrap CI for rho(x, y_second) - rho(x, y_first)."""
    x_array = np.asarray(x, dtype=float)
    first_array = np.asarray(y_first, dtype=float)
    second_array = np.asarray(y_second, dtype=float)
    observed = float(stats.spearmanr(x_array, second_array).statistic - stats.spearmanr(x_array, first_array).statistic)
    rng = np.random.default_rng(stable_seed(label + "paired-difference"))
    indices = rng.integers(0, len(x_array), size=(N_BOOT, len(x_array)))
    x_rank = stats.rankdata(x_array[indices], axis=1, method="average")
    first_rank = stats.rankdata(first_array[indices], axis=1, method="average")
    second_rank = stats.rankdata(second_array[indices], axis=1, method="average")
    difference = rowwise_pearson(x_rank, second_rank) - rowwise_pearson(x_rank, first_rank)
    difference = difference[np.isfinite(difference)]
    low, high = np.percentile(difference, [2.5, 97.5])
    return observed, float(low), float(high)


def permutation_pvalue(x: Sequence[float], y: Sequence[float], label: str, n_perm: int = N_PERM) -> float:
    x_rank = stats.rankdata(np.asarray(x, dtype=float), method="average")
    y_rank = stats.rankdata(np.asarray(y, dtype=float), method="average")
    observed = float(stats.pearsonr(x_rank, y_rank).statistic)
    rng = np.random.default_rng(stable_seed(label + "perm"))
    random_order = np.argsort(rng.random((n_perm, len(y_rank))), axis=1)
    x_matrix = np.broadcast_to(x_rank, random_order.shape)
    permuted = y_rank[random_order]
    null = rowwise_pearson(x_matrix, permuted)
    return float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))


def correlation_row(label: str, records: Sequence[dict], target_key: str, note: str = "") -> dict:
    valid = [record for record in records if is_finite(record["probe_final"]) and is_finite(record[target_key])]
    x = np.asarray([record["probe_final"] for record in valid], dtype=float)
    y = np.asarray([record[target_key] for record in valid], dtype=float)
    if len(valid) < 2:
        return {
            "analysis": label,
            "target": target_key,
            "n": len(valid),
            "spearman_rho": float("nan"),
            "spearman_ci_low": float("nan"),
            "spearman_ci_high": float("nan"),
            "spearman_p_asymptotic": float("nan"),
            "spearman_p_permutation": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "kendall_tau_b": float("nan"),
            "kendall_p": float("nan"),
            "note": note,
        }
    spearman = stats.spearmanr(x, y)
    pearson = stats.pearsonr(x, y)
    kendall = stats.kendalltau(x, y, variant="b")
    ci_low, ci_high = bootstrap_corr_ci(x, y, "spearman", label)
    return {
        "analysis": label,
        "target": target_key,
        "n": len(valid),
        "spearman_rho": float(spearman.statistic),
        "spearman_ci_low": ci_low,
        "spearman_ci_high": ci_high,
        "spearman_p_asymptotic": float(spearman.pvalue),
        "spearman_p_permutation": permutation_pvalue(x, y, label),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "kendall_tau_b": float(kendall.statistic),
        "kendall_p": float(kendall.pvalue),
        "note": note,
    }


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return [float(value) for value in result]


def family_adjusted_spearman(records: Sequence[dict]) -> dict:
    grouped = Counter(record["model_family"] for record in records)
    valid = [record for record in records if grouped[record["model_family"]] >= 2]
    x = stats.rankdata([record["probe_final"] for record in valid], method="average").astype(float)
    y = stats.rankdata([record["mllm_avg_fair"] for record in valid], method="average").astype(float)
    families = np.asarray([record["model_family"] for record in valid])
    family_indices = []
    for family in sorted(set(families)):
        index = np.flatnonzero(families == family)
        x[index] -= np.mean(x[index])
        y[index] -= np.mean(y[index])
        family_indices.append(index)
    observed = float(stats.pearsonr(x, y).statistic)

    rng = np.random.default_rng(stable_seed("family-adjusted"))
    exceed = 0
    for _ in range(N_PERM):
        permuted = y.copy()
        for index in family_indices:
            permuted[index] = rng.permutation(permuted[index])
        value = float(stats.pearsonr(x, permuted).statistic)
        exceed += abs(value) >= abs(observed)
    return {
        "analysis": "Family-adjusted pooled rank association",
        "target": "mllm_avg_fair",
        "n": len(valid),
        "spearman_rho": observed,
        "spearman_ci_low": float("nan"),
        "spearman_ci_high": float("nan"),
        "spearman_p_asymptotic": float("nan"),
        "spearman_p_permutation": (exceed + 1) / (N_PERM + 1),
        "pearson_r": float("nan"),
        "pearson_p": float("nan"),
        "kendall_tau_b": float("nan"),
        "kendall_p": float("nan"),
        "note": "Global ranks were centered within each family; singleton OpenAI CLIP contributes no within-family information.",
    }


def ols_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    train_matrix = np.column_stack([np.ones(len(train_x)), train_x])
    test_matrix = np.column_stack([np.ones(len(test_x)), test_x])
    coefficients = np.linalg.lstsq(train_matrix, train_y, rcond=None)[0]
    return test_matrix @ coefficients


def cross_validated_predictions(records: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    x = np.asarray([record["probe_final"] for record in records], dtype=float)
    y = np.asarray([record["mllm_avg_fair"] for record in records], dtype=float)
    families = np.asarray([record["model_family"] for record in records])
    n = len(records)
    loo = np.empty(n)
    loo_baseline = np.empty(n)
    for index in range(n):
        test = np.arange(n) == index
        train = ~test
        loo[index] = ols_predict(x[train, None], y[train], x[test, None])[0]
        loo_baseline[index] = float(np.mean(y[train]))

    # A singleton is not a meaningful held-out "family".  Cross-family
    # validation therefore evaluates the three replicated prefix families.
    major_families = {"SigLIP2", "MetaCLIP1", "MetaCLIP2"}
    major_mask = np.asarray([family in major_families for family in families])
    lofo = np.full(n, np.nan)
    lofo_baseline = np.full(n, np.nan)
    for family in sorted(major_families):
        test = families == family
        train = major_mask & ~test
        lofo[test] = ols_predict(x[train, None], y[train], x[test, None])
        lofo_baseline[test] = float(np.mean(y[train]))

    diagnostics = []
    for index, record in enumerate(records):
        diagnostics.append(
            {
                "tokenizer": record["tokenizer"],
                "model_family": record["model_family"],
                "probe_final": x[index],
                "mllm_avg_fair": y[index],
                "loocv_prediction": loo[index],
                "loocv_residual_observed_minus_predicted": y[index] - loo[index],
                "leave_family_out_prediction": lofo[index],
                "leave_family_out_residual_observed_minus_predicted": y[index] - lofo[index],
            }
        )

    metrics = []
    for label, prediction, baseline, mask in [
        ("Leave-one-tokenizer-out", loo, loo_baseline, np.ones(n, dtype=bool)),
        ("Leave-one-family-out", lofo, lofo_baseline, major_mask),
    ]:
        target = y[mask]
        prediction_used = prediction[mask]
        baseline_used = baseline[mask]
        metrics.append(
            {
                "validation": label,
                "n": int(np.count_nonzero(mask)),
                "mae": float(np.mean(np.abs(target - prediction_used))),
                "rmse": float(np.sqrt(np.mean((target - prediction_used) ** 2))),
                "r2_cv": float(1 - np.sum((target - prediction_used) ** 2) / np.sum((target - np.mean(target)) ** 2)),
                "baseline_mae": float(np.mean(np.abs(target - baseline_used))),
                "baseline_rmse": float(np.sqrt(np.mean((target - baseline_used) ** 2))),
            }
        )
    return diagnostics, metrics


def trajectory_cv_metrics(epoch_records: Sequence[dict]) -> tuple[list[dict], float, float]:
    epoch_matrix = np.asarray([record["epochs"] for record in epoch_records], dtype=float)
    y = np.asarray([record["mllm_avg_fair"] for record in epoch_records], dtype=float)
    feature_sets = {
        "Epoch 1 only": epoch_matrix[:, [0]],
        "Epoch 10 only": epoch_matrix[:, [9]],
        "Epoch 10 + gain (E10-E1)": np.column_stack([epoch_matrix[:, 9], epoch_matrix[:, 9] - epoch_matrix[:, 0]]),
    }
    rows = []
    for label, features in feature_sets.items():
        predictions = np.empty(len(y))
        for index in range(len(y)):
            test = np.arange(len(y)) == index
            train = ~test
            predictions[index] = ols_predict(features[train], y[train], features[test])[0]
        rows.append(
            {
                "validation": f"Trajectory subset LOOCV: {label}",
                "n": len(y),
                "mae": float(np.mean(np.abs(y - predictions))),
                "rmse": float(np.sqrt(np.mean((y - predictions) ** 2))),
                "r2_cv": float(1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)),
                "baseline_mae": float("nan"),
                "baseline_rmse": float("nan"),
            }
        )
    final_fit = ols_predict(epoch_matrix[:, [9]], y, epoch_matrix[:, [9]])
    gain = epoch_matrix[:, 9] - epoch_matrix[:, 0]
    gain_vs_target = float(stats.spearmanr(gain, y).statistic)
    gain_vs_residual = float(stats.spearmanr(gain, y - final_fit).statistic)
    return rows, gain_vs_target, gain_vs_residual


def csv_value(value):
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(value) else f"{float(value):.8g}"
    if isinstance(value, bool):
        return "1" if value else "0"
    return value


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "savefig.facecolor": "white",
        }
    )


def spread_labels(values: Sequence[float], low: float, high: float, gap: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    placed = values[order].copy()
    placed[0] = max(placed[0], low)
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + gap)
    if placed[-1] > high:
        placed -= placed[-1] - high
    for index in range(len(placed) - 2, -1, -1):
        placed[index] = min(placed[index], placed[index + 1] - gap)
    if placed[0] < low:
        placed += low - placed[0]
    result = np.empty_like(placed)
    result[order] = placed
    return result


def plot_epoch_trajectories(epoch_records: Sequence[dict]) -> None:
    plot_groups = {
        "SigLIP2": [record for record in epoch_records if record["model_family"] == "SigLIP2"],
        "MetaCLIP1": [record for record in epoch_records if record["model_family"] == "MetaCLIP1"],
        "MetaCLIP2": [record for record in epoch_records if record["model_family"] == "MetaCLIP2"],
        "Discrete tokenizers": [record for record in epoch_records if record["visual_token_type"] == "discrete"],
        "OpenAI CLIP": [record for record in epoch_records if record["model_family"] == "OpenAI CLIP"],
    }
    fig, axes = plt.subplots(3, 2, figsize=(17, 13), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    epochs_x = np.arange(1, 11)
    for axis, (group, records) in zip(axes_flat[:5], plot_groups.items()):
        records = sorted(records, key=lambda item: item["epochs"][-1])
        colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(len(records), 2)))
        endpoints = [record["epochs"][-1] for record in records]
        label_y = spread_labels(endpoints, 68.4, 89.1, 0.52) if len(records) > 1 else np.asarray(endpoints)
        for index, (record, y_label) in enumerate(zip(records, label_y)):
            color = colors[index]
            axis.plot(epochs_x, record["epochs"], marker="o", markersize=2.8, linewidth=1.35, color=color, alpha=0.95)
            axis.plot([10, 10.28], [record["epochs"][-1], y_label], color=color, linewidth=0.7, alpha=0.8)
            axis.text(10.34, y_label, short_name(record["tokenizer"]), fontsize=6.2, va="center", color="#202020")
        axis.set_title(f"{group} (n={len(records)})")
        axis.set_xlim(1, 13.1)
        axis.set_ylim(68, 89.6)
        axis.set_xticks(range(1, 11))
        axis.grid(axis="x", alpha=0.25)
    info_axis = axes_flat[5]
    info_axis.axis("off")
    gains = np.asarray([record["epochs"][-1] - record["epochs"][0] for record in epoch_records])
    family_counts = Counter(record["model_family"] for record in epoch_records)
    info_lines = [
        "Coverage",
        f"33 tokenizers with all 10 epochs",
        f"Median E1→E10 gain: {np.median(gains):.2f} points",
        f"Range of gains: {np.min(gains):.2f} to {np.max(gains):.2f}",
        "",
        "Trajectory counts",
    ] + [f"{family}: {count}" for family, count in family_counts.most_common()]
    info_axis.text(0.07, 0.93, "\n".join(info_lines), va="top", fontsize=11, linespacing=1.45, transform=info_axis.transAxes)
    fig.supxlabel("Linear-probe epoch")
    fig.supylabel("ImageNet Top-1 accuracy (%)")
    fig.suptitle("ImageNet linear-probe trajectories: every available tokenizer", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.98))
    fig.savefig(FIGURE_DIR / "01_epoch_accuracy_trajectories.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_epochs_by_tokenizer(epoch_records: Sequence[dict]) -> None:
    """Show all ten epochs across tokenizers, matching the requested overview style."""
    sorted_records = sorted(epoch_records, key=lambda record: np.mean(record["epochs"]), reverse=True)
    accuracy = np.asarray([record["epochs"] for record in sorted_records], dtype=float)
    x = np.arange(len(sorted_records))
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.08, 0.92, 10))

    fig, axis = plt.subplots(figsize=(15.5, 6.4))
    axis.fill_between(
        x,
        accuracy[:, 0],
        accuracy[:, -1],
        color="#4C78A8",
        alpha=0.12,
        linewidth=0,
        label="Epoch 1→10 gain band",
        zorder=1,
    )
    for epoch_index in range(10):
        endpoint = epoch_index in {0, 9}
        axis.plot(
            x,
            accuracy[:, epoch_index],
            color=colors[epoch_index],
            marker="o",
            markersize=2.25 if endpoint else 1.55,
            markeredgewidth=0,
            linewidth=1.05 if endpoint else 0.55,
            alpha=0.95 if endpoint else 0.62,
            zorder=3 if endpoint else 2,
        )

    norm = matplotlib.colors.Normalize(vmin=1, vmax=10)
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis, pad=0.015)
    colorbar.set_ticks(range(1, 11))
    colorbar.set_label("Linear-probe epoch")
    axis.set_xlim(-0.45, len(sorted_records) - 0.55)
    axis.set_xticks(x)
    axis.set_xticklabels([])
    axis.tick_params(axis="x", length=2.2, width=0.6)
    axis.grid(axis="x", visible=False)
    axis.set_xlabel("Tokenizer (sorted by 10-epoch mean accuracy; labels hidden)")
    axis.set_ylabel("ImageNet Top-1 accuracy (%)")
    axis.set_title(f"Ten linear-probe epochs by tokenizer (n={len(sorted_records)})")
    axis.text(
        0.985,
        0.965,
        f"mean E1→E10 gain = {np.mean(accuracy[:, -1] - accuracy[:, 0]):+.2f} pp",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01b_epoch_by_tokenizer_overview.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_rank_heatmap(epoch_records: Sequence[dict]) -> None:
    sorted_records = sorted(epoch_records, key=lambda record: record["epochs"][-1], reverse=True)
    accuracy = np.asarray([record["epochs"] for record in epoch_records], dtype=float)
    ranks = np.column_stack([stats.rankdata(-accuracy[:, index], method="average") for index in range(10)])
    index_by_name = {record["tokenizer"]: index for index, record in enumerate(epoch_records)}
    ordered_ranks = np.asarray([ranks[index_by_name[record["tokenizer"]]] for record in sorted_records])

    fig, axis = plt.subplots(figsize=(12.5, 13.5))
    image = axis.imshow(ordered_ranks, cmap="viridis_r", vmin=1, vmax=len(epoch_records), aspect="auto")
    for row in range(ordered_ranks.shape[0]):
        for column in range(ordered_ranks.shape[1]):
            value = ordered_ranks[row, column]
            text_value = f"{value:.0f}" if value.is_integer() else f"{value:.1f}"
            color = "white" if value > 18 else "#101010"
            axis.text(column, row, text_value, ha="center", va="center", fontsize=6.2, color=color)
    axis.set_xticks(range(10), [f"E{index}" for index in range(1, 11)])
    axis.set_yticks(range(len(sorted_records)), [record["tokenizer"] for record in sorted_records], fontsize=7.2)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Tokenizer (sorted by Epoch-10 rank)")
    axis.set_title("Relative-rank evolution (1 = best; ties use average rank)", pad=12)
    axis.grid(False)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Rank (lower is better)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_epoch_rank_heatmap.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_predictiveness(epoch_metrics: Sequence[dict], epoch_records: Sequence[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
    epoch_numbers = [row["epoch"] for row in epoch_metrics]
    axes[0].plot(epoch_numbers, [row["rank_stability_vs_epoch10_n33"] for row in epoch_metrics], marker="o", linewidth=2, label="Rank stability vs E10 (n=33)")
    axes[0].plot(epoch_numbers, [row["spearman_vs_qwen3_avg_n33"] for row in epoch_metrics], marker="o", linewidth=2, label="Qwen3 Avg (n=33)")
    axes[0].plot(epoch_numbers, [row["spearman_vs_fair_mllm_avg_n29"] for row in epoch_metrics], marker="o", linewidth=2, label="Two-Qwen fair Avg (n=29)")
    axes[0].set_xticks(range(1, 11))
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("Probe epoch")
    axes[0].set_ylabel("Spearman rho")
    axes[0].set_title("Ranking signal is already strong at Epoch 1")
    axes[0].legend(loc="lower right")

    groups = {
        "SigLIP2": [record for record in epoch_records if record["model_family"] == "SigLIP2"],
        "MetaCLIP1": [record for record in epoch_records if record["model_family"] == "MetaCLIP1"],
        "MetaCLIP2": [record for record in epoch_records if record["model_family"] == "MetaCLIP2"],
        "TokLIP": [record for record in epoch_records if record["model_family"] == "TokLIP"],
        "Other discrete": [record for record in epoch_records if record["visual_token_type"] == "discrete" and record["model_family"] != "TokLIP"],
        "OpenAI CLIP": [record for record in epoch_records if record["model_family"] == "OpenAI CLIP"],
    }
    rng = np.random.default_rng(SEED)
    for x_position, (group, records) in enumerate(groups.items()):
        gains = np.asarray([record["epochs"][-1] - record["epochs"][0] for record in records])
        jitter = rng.uniform(-0.13, 0.13, size=len(gains))
        color = FAMILY_COLORS.get(group, "#777777")
        if group == "Other discrete":
            color = "#56B4E9"
        axes[1].scatter(np.full(len(gains), x_position) + jitter, gains, s=45, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        if len(gains):
            axes[1].plot([x_position - 0.2, x_position + 0.2], [np.median(gains)] * 2, color="#222222", linewidth=2)
        for record, x_jitter, gain in zip(records, jitter, gains):
            if gain >= 5:
                axes[1].annotate(short_name(record["tokenizer"]), (x_position + x_jitter, gain), xytext=(3, 4), textcoords="offset points", fontsize=7)
    axes[1].set_xticks(range(len(groups)), list(groups), rotation=25, ha="right")
    axes[1].set_ylabel("Accuracy gain, E10 - E1 (points)")
    axes[1].set_title("TokLIP converges much more slowly")
    fig.suptitle("What the 10-epoch histories add", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_epoch_predictiveness.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def add_regression_line(axis, x: np.ndarray, y: np.ndarray) -> None:
    coefficients = np.polyfit(x, y, 1)
    x_line = np.linspace(np.min(x), np.max(x), 100)
    axis.plot(x_line, np.polyval(coefficients, x_line), color="#222222", linewidth=1.4, linestyle="--", alpha=0.8)


def plot_probe_vs_mllm(core_records: Sequence[dict], summary_by_label: dict[str, dict]) -> None:
    panels = [
        ("mllm_avg_fair", "Two-Qwen fair Avg", "Primary: fair two-backbone avg"),
        ("qwen3_avg_reported", "Qwen3-1.7B Avg", "Qwen3 Avg (matched n=38)"),
        ("qwen25_avg_reported", "Qwen2.5-1.5B Avg", "Qwen2.5 Avg (matched n=38)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharex=True)
    for axis, (target, y_label, summary_label) in zip(axes, panels):
        x = np.asarray([record["probe_final"] for record in core_records])
        y = np.asarray([record[target] for record in core_records])
        for record in core_records:
            full_history = record["probe_history_status"] == "full_10_epoch"
            axis.scatter(
                record["probe_final"],
                record[target],
                s=48,
                facecolor=FAMILY_COLORS[record["model_family"]] if full_history else "none",
                edgecolor="white" if full_history else FAMILY_COLORS[record["model_family"]],
                linewidth=0.7 if full_history else 1.3,
                alpha=0.92,
                zorder=3,
            )
        add_regression_line(axis, x, y)
        stats_row = summary_by_label[summary_label]
        axis.set_title(f"{y_label}\nn={stats_row['n']}, Spearman rho={stats_row['spearman_rho']:.3f}")
        axis.set_xlabel("ImageNet linear probe, Epoch 10 (%)")
        axis.set_ylabel(y_label)

    # Label the largest primary linear residuals, not every crowded point.
    x = np.asarray([record["probe_final"] for record in core_records])
    y = np.asarray([record["mllm_avg_fair"] for record in core_records])
    residuals = y - np.polyval(np.polyfit(x, y, 1), x)
    selected = np.argsort(np.abs(residuals))[-6:]
    y_limits = axes[0].get_ylim()
    label_positions = spread_labels(y[selected], y_limits[0] + 0.25, y_limits[1] - 0.25, 0.62)
    for index, label_y in zip(selected, label_positions):
        record = core_records[index]
        to_left = x[index] > np.median(x)
        label_x = x[index] - 0.18 if to_left else x[index] + 0.18
        axes[0].annotate(
            short_name(record["tokenizer"]),
            (x[index], y[index]),
            xytext=(label_x, label_y),
            textcoords="data",
            ha="right" if to_left else "left",
            va="center",
            fontsize=6.5,
            arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.5},
        )

    legend_families = ["SigLIP2", "MetaCLIP1", "MetaCLIP2", "OpenAI CLIP"]
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor=FAMILY_COLORS[family], markeredgecolor="white", label=family)
        for family in legend_families
    ]
    handles.extend(
        [
            Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor="#777777", markeredgecolor="white", label="Full 10-epoch history"),
            Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor="none", markeredgecolor="#777777", label="Final probe only"),
        ]
    )
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Linear probing predicts MLLM ranking, but strength depends on the Qwen backbone", fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(FIGURE_DIR / "04_probe_vs_mllm_avg.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_task_heatmap(task_rows: Sequence[dict], summary_by_label: dict[str, dict]) -> None:
    columns = TASKS + ["Avg"]
    plotted_backbones = BACKBONES + ["Two-Qwen task mean"]
    matrix = np.zeros((3, len(columns)))
    q_matrix = np.zeros_like(matrix)
    for row_index, backbone in enumerate(plotted_backbones):
        for column_index, task in enumerate(TASKS):
            row = next(item for item in task_rows if item["backbone"] == backbone and item["task"] == task)
            matrix[row_index, column_index] = row["matched_spearman_rho"]
            q_matrix[row_index, column_index] = row["matched_p"]
    matrix[0, -1] = summary_by_label["Qwen3 Avg (matched n=38)"]["spearman_rho"]
    matrix[1, -1] = summary_by_label["Qwen2.5 Avg (matched n=38)"]["spearman_rho"]
    matrix[2, -1] = summary_by_label["Primary: fair two-backbone avg"]["spearman_rho"]
    q_matrix[0, -1] = summary_by_label["Qwen3 Avg (matched n=38)"]["spearman_p_asymptotic"]
    q_matrix[1, -1] = summary_by_label["Qwen2.5 Avg (matched n=38)"]["spearman_p_asymptotic"]
    q_matrix[2, -1] = summary_by_label["Primary: fair two-backbone avg"]["spearman_p_asymptotic"]

    fig, axis = plt.subplots(figsize=(15, 3.8))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            suffix = "*" if q_matrix[row, column] < 0.05 else ""
            color = "white" if matrix[row, column] > 0.7 else "#111111"
            axis.text(column, row, f"{matrix[row, column]:.2f}{suffix}", ha="center", va="center", fontsize=8, color=color)
    axis.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
    axis.set_yticks(range(3), plotted_backbones)
    axis.set_title("Task-level Spearman correlation on the same 38-tokenizer cohort (* p < 0.05)")
    axis.grid(False)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Spearman rho")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_task_correlation_heatmap.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(rows: Sequence[dict]) -> None:
    plotted = [row for row in rows if is_finite(row["spearman_ci_low"])]
    labels = [f"{row['analysis']} (n={row['n']})" for row in plotted]
    rho = np.asarray([row["spearman_rho"] for row in plotted])
    low = np.asarray([row["spearman_ci_low"] for row in plotted])
    high = np.asarray([row["spearman_ci_high"] for row in plotted])
    y = np.arange(len(plotted))
    colors = ["#0072B2" if row["analysis"].startswith("Primary") else "#666666" for row in plotted]
    fig, axis = plt.subplots(figsize=(11, 7.3))
    axis.errorbar(rho, y, xerr=np.vstack([rho - low, high - rho]), fmt="none", ecolor="#888888", elinewidth=1.4, capsize=3)
    axis.scatter(rho, y, color=colors, s=55, zorder=3)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(max(0, float(np.nanmin(low)) - 0.05), 1.02)
    axis.set_xlabel("Spearman rho with 95% tokenizer-bootstrap interval")
    axis.set_title("The two-Qwen average result is stable across source and major families")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_family_and_source_robustness.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_validation(diagnostics: Sequence[dict], prediction_metrics: Sequence[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
    configurations = [
        ("loocv_prediction", "Leave-one-tokenizer-out", prediction_metrics[0]),
        ("leave_family_out_prediction", "Leave-one-family-out", prediction_metrics[1]),
    ]
    observed = np.asarray([row["mllm_avg_fair"] for row in diagnostics])
    limits = [min(observed) - 1, max(observed) + 1]
    for axis, (prediction_key, title, metrics) in zip(axes, configurations):
        predicted = np.asarray([row[prediction_key] for row in diagnostics])
        for row in diagnostics:
            if not is_finite(row[prediction_key]):
                continue
            axis.scatter(
                row[prediction_key],
                row["mllm_avg_fair"],
                s=48,
                color=FAMILY_COLORS[row["model_family"]],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.92,
                zorder=3,
            )
        axis.plot(limits, limits, linestyle="--", color="#222222", linewidth=1)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_xlabel("Out-of-sample predicted MLLM Avg")
        axis.set_ylabel("Observed MLLM Avg")
        axis.set_title(f"{title}\nMAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²cv={metrics['r2_cv']:.2f}")
        valid_indices = np.flatnonzero(np.isfinite(predicted))
        residuals = observed[valid_indices] - predicted[valid_indices]
        for index in valid_indices[np.argsort(np.abs(residuals))[-3:]]:
            row = diagnostics[index]
            axis.annotate(short_name(row["tokenizer"]), (predicted[index], observed[index]), xytext=(4, 4), textcoords="offset points", fontsize=6.5)
    fig.suptitle("A one-variable linear calibration retains useful out-of-sample accuracy", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_DIR / "07_prediction_validation.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_controlled_deltas(controlled_rows: Sequence[dict]) -> None:
    rows = [row for row in controlled_rows if row["comparison_type"] == "resolution"]
    fig, axis = plt.subplots(figsize=(8.8, 6.4))
    for row in rows:
        color = FAMILY_COLORS[row["model_family"]]
        axis.scatter(row["delta_probe"], row["delta_mllm_avg"], s=75, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        axis.annotate(row["pair_label"], (row["delta_probe"], row["delta_mllm_avg"]), xytext=(5, 4), textcoords="offset points", fontsize=7.5)
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_xlabel("Change in ImageNet probe (higher resolution minus lower)")
    axis.set_ylabel("Change in two-Qwen MLLM Avg")
    axis.set_title(f"Controlled resolution upgrades: direction agrees, magnitude does not (n={len(rows)})")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor=FAMILY_COLORS[family], markeredgecolor="white", label=family)
        for family in ["SigLIP2", "MetaCLIP2"]
    ]
    axis.legend(handles=handles, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "08_controlled_resolution_deltas.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def p_text(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def write_report(
    records: Sequence[dict],
    audit: dict,
    summary_rows: Sequence[dict],
    aggregation_rows: Sequence[dict],
    robustness_rows: Sequence[dict],
    task_rows: Sequence[dict],
    epoch_metrics: Sequence[dict],
    prediction_metrics: Sequence[dict],
    controlled_rows: Sequence[dict],
    gain_vs_target: float,
    gain_vs_residual: float,
) -> None:
    summary = {row["analysis"]: row for row in summary_rows}
    aggregation = {row["analysis"]: row for row in aggregation_rows}
    robust = {row["analysis"]: row for row in robustness_rows}
    primary = summary["Primary: fair two-backbone avg"]
    q3_matched = summary["Qwen3 Avg (matched n=38)"]
    q25_matched = summary["Qwen2.5 Avg (matched n=38)"]
    q3_max = summary["Qwen3 Avg (maximum coverage)"]
    q3_max_without_ijepa = summary["Qwen3 Avg (maximum coverage, excluding I-JEPA)"]
    q3_cont = summary["Qwen3 Avg (continuous only)"]
    q3_disc = summary["Qwen3 Avg (discrete only)"]
    history = summary["Two-Qwen Avg (full 10-epoch histories)"]
    final_only = summary["Two-Qwen Avg (final-only probing)"]

    core = [record for record in records if record["in_primary_n38"]]
    epoch_records = [record for record in records if record["probe_history_status"] == "full_10_epoch"]
    topk_lines = []
    probe_order = sorted(core, key=lambda record: record["probe_final"], reverse=True)
    mllm_order = sorted(core, key=lambda record: record["mllm_avg_fair"], reverse=True)
    for k in [3, 5, 10]:
        probe_top = {record["tokenizer"] for record in probe_order[:k]}
        mllm_top = {record["tokenizer"] for record in mllm_order[:k]}
        topk_lines.append(f"- Top-{k} 重合 {len(probe_top & mllm_top)}/{k}。")

    core_probe = np.asarray([record["probe_final"] for record in core])
    core_q3 = np.asarray([record["qwen3_avg_reported"] for record in core])
    core_q25 = np.asarray([record["qwen25_avg_reported"] for record in core])
    backbone_delta, backbone_delta_low, backbone_delta_high = bootstrap_spearman_difference(
        core_probe, core_q3, core_q25, "matched-backbone-rho"
    )
    concordant = discordant = tied = 0
    for first in range(len(core)):
        for second in range(first + 1, len(core)):
            product = (core[first]["probe_final"] - core[second]["probe_final"]) * (
                core[first]["mllm_avg_fair"] - core[second]["mllm_avg_fair"]
            )
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                tied += 1
    pairwise_accuracy = concordant / (concordant + discordant)
    leave_one_rhos = []
    for index in range(len(core)):
        kept = [record for row, record in enumerate(core) if row != index]
        leave_one_rhos.append(
            float(stats.spearmanr([record["probe_final"] for record in kept], [record["mllm_avg_fair"] for record in kept]).statistic)
        )

    task_by_backbone = defaultdict(list)
    for row in task_rows:
        task_by_backbone[row["backbone"]].append(row)
    q3_best = max(task_by_backbone["Qwen3-1.7B"], key=lambda row: row["matched_spearman_rho"])
    q3_worst = min(task_by_backbone["Qwen3-1.7B"], key=lambda row: row["matched_spearman_rho"])
    q25_best = max(task_by_backbone["Qwen2.5-1.5B"], key=lambda row: row["matched_spearman_rho"])
    q25_worst = min(task_by_backbone["Qwen2.5-1.5B"], key=lambda row: row["matched_spearman_rho"])
    combined_best = max(task_by_backbone["Two-Qwen task mean"], key=lambda row: row["matched_spearman_rho"])
    combined_worst = min(task_by_backbone["Two-Qwen task mean"], key=lambda row: row["matched_spearman_rho"])
    q3_sig = sum(row["matched_p"] < 0.05 for row in task_by_backbone["Qwen3-1.7B"])
    q25_sig = sum(row["matched_p"] < 0.05 for row in task_by_backbone["Qwen2.5-1.5B"])
    bh_significant = sum(
        row["max_q_bh_22"] < 0.05
        for row in task_rows
        if row["backbone"] in BACKBONES
    )

    aggregation_leave_out = [row for row in aggregation_rows if row["analysis"].startswith("Leave out ")]
    aggregation_leave_min = min(row["spearman_rho"] for row in aggregation_leave_out)
    aggregation_leave_max = max(row["spearman_rho"] for row in aggregation_leave_out)

    epoch1 = epoch_metrics[0]
    epoch10 = epoch_metrics[-1]
    gains = np.asarray([record["epochs"][-1] - record["epochs"][0] for record in epoch_records])
    rank_e1 = stats.rankdata(-np.asarray([record["epochs"][0] for record in epoch_records]), method="average")
    rank_e10 = stats.rankdata(-np.asarray([record["epochs"][-1] for record in epoch_records]), method="average")
    movement = np.asarray(rank_e1) - np.asarray(rank_e10)
    biggest_movers = sorted(
        zip(epoch_records, movement),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:2]
    epoch_core = [record for record in epoch_records if record["in_epoch_combined_n29"]]
    epoch_delta, epoch_delta_low, epoch_delta_high = bootstrap_spearman_difference(
        [record["mllm_avg_fair"] for record in epoch_core],
        [record["epochs"][0] for record in epoch_core],
        [record["epochs"][-1] for record in epoch_core],
        "epoch10-minus-epoch1-rho",
    )
    epoch_spearman_lines = [
        "| "
        + " | ".join(
            [
                str(row["epoch"]),
                f"{row['spearman_vs_qwen3_avg_n33']:.3f}",
                f"{row['spearman_vs_qwen3_avg_matched_n29']:.3f}",
                f"{row['spearman_vs_qwen25_avg_n29']:.3f}",
                f"{row['spearman_vs_fair_mllm_avg_n29']:.3f}",
            ]
        )
        + " |"
        for row in epoch_metrics
    ]

    family_lines = []
    for label in ["Within SigLIP2", "Within MetaCLIP1", "Within MetaCLIP2"]:
        row = robust[label]
        family_lines.append(
            f"| {label.replace('Within ', '')} | {row['n']} | {row['spearman_rho']:.3f} | [{row['spearman_ci_low']:.3f}, {row['spearman_ci_high']:.3f}] |"
        )
    adjusted = robust["Family-adjusted pooled rank association"]
    top_half = probe_order[: len(core) // 2]
    top_half_rho = float(
        stats.spearmanr([record["probe_final"] for record in top_half], [record["mllm_avg_fair"] for record in top_half]).statistic
    )
    high_probe = [record for record in core if record["probe_final"] >= 87]
    high_probe_result = stats.spearmanr(
        [record["probe_final"] for record in high_probe], [record["mllm_avg_fair"] for record in high_probe]
    )

    resolution_rows = [row for row in controlled_rows if row["comparison_type"] == "resolution"]
    delta_x = [row["delta_probe"] for row in resolution_rows]
    delta_y = [row["delta_mllm_avg"] for row in resolution_rows]
    delta_rho = float(stats.spearmanr(delta_x, delta_y).statistic)
    delta_p = float(stats.spearmanr(delta_x, delta_y).pvalue)
    positive_both = sum(row["delta_probe"] > 0 and row["delta_mllm_avg"] > 0 for row in resolution_rows)

    loo = prediction_metrics[0]
    lofo = prediction_metrics[1]
    trajectory_metric = {row["validation"]: row for row in prediction_metrics[2:]}
    epoch1_cv = trajectory_metric["Trajectory subset LOOCV: Epoch 1 only"]
    epoch10_cv = trajectory_metric["Trajectory subset LOOCV: Epoch 10 only"]
    gain_cv = trajectory_metric["Trajectory subset LOOCV: Epoch 10 + gain (E10-E1)"]

    no_history = [record for record in records if record["probe_history_status"] == "final_only"]
    no_probe = [record for record in records if record["probe_history_status"] == "missing_probe"]
    q25_missing = [record for record in records if not is_finite(record["qwen25_avg_reported"])]
    names_markdown = lambda items: ", ".join(f"`{record['tokenizer']}`" for record in items)  # noqa: E731
    q3_missing = [record for record in records if not is_finite(record["qwen3_avg_reported"])]

    report = f"""# Linear probing 对 MLLM 表现的预测力

## 结论先行

在公平的主队列上，ImageNet Epoch-10 linear probing 对两个 Qwen MLLM 的平均排名预测力很强：**n={primary['n']}，Spearman rho={primary['spearman_rho']:.3f}**（tokenizer 自助法 95% CI [{primary['spearman_ci_low']:.3f}, {primary['spearman_ci_high']:.3f}]），Pearson r={primary['pearson_r']:.3f}，Kendall tau-b={primary['kendall_tau_b']:.3f}，置换检验 p={p_text(primary['spearman_p_permutation'])}。这就是“约 0.94”的正确口径。

但结论需要加两个限定：

- 这 38 个点全是 **continuous tokenizer**；4 个 discrete tokenizer 都缺 Qwen2.5，因此 0.944 不能直接外推到 discrete。
- 预测强度对 MLLM backbone 敏感：在完全相同的 38 个 tokenizer 上，Qwen3 rho={q3_matched['spearman_rho']:.3f}，Qwen2.5 rho={q25_matched['spearman_rho']:.3f}，差 {backbone_delta:.3f}（配对 bootstrap 95% CI [{backbone_delta_low:.3f}, {backbone_delta_high:.3f}]）。两者均为强相关，但 Qwen2.5 更贴合 probing 排名。

![Probing vs MLLM](figures/04_probe_vs_mllm_avg.png)

## 数据口径与样本数

| 用途 | 使用数 | 公平性口径 | 排除 |
|---|---:|---|---|
| 主分析：probing × 两 Qwen 公平 Avg | 38 | probing、Qwen3、Qwen2.5 都完整；Avg 为两个 backbone Avg 等权平均 | 2 个无 probing；6 个缺 Qwen2.5；DINOv3 缺两个 MLLM |
| Qwen3 最大覆盖 | 44 | 仅要求 probing + Qwen3 | 2 个无 probing；DINOv3 无 MLLM |
| Qwen2.5 分析 | 38 | 仅要求 probing + Qwen2.5 | 同主队列；现有 Qwen2.5 均为 continuous |
| 10-epoch 轨迹可视化 | 33 | 必须 10 轮全齐 | {len(no_history)} 个只有最终 probing；{len(no_probe)} 个 probing 全缺 |
| 10-epoch × 两 Qwen Avg | 29 | 轨迹和两 backbone 都完整 | 上述双重完整病例交集 |

主表有 47 个 tokenizer：45 个有最终 probing，33 个有全部 10 轮。完整排除列表在 [`data/exclusions.csv`](data/exclusions.csv)，逐 tokenizer 的合并审计表在 [`data/analysis_cohort.csv`](data/analysis_cohort.csv)。

- probing 完全缺失（2）：{names_markdown(no_probe)}。
- 有最终 probing，但前 9 轮不在 epoch 文件（12）：{names_markdown(no_history)}。
- Qwen2.5 整块缺失（7）：{names_markdown(q25_missing)}。
- Qwen3 整块缺失（1）：{names_markdown(q3_missing)}。

## 主相关性与稳健性

| 目标/子集 | n | Spearman rho | 95% CI | Pearson r | Kendall tau-b |
|---|---:|---:|---:|---:|---:|
| 两 Qwen 公平 Avg（主结果） | {primary['n']} | {primary['spearman_rho']:.3f} | [{primary['spearman_ci_low']:.3f}, {primary['spearman_ci_high']:.3f}] | {primary['pearson_r']:.3f} | {primary['kendall_tau_b']:.3f} |
| Qwen3 Avg（同一主队列） | {q3_matched['n']} | {q3_matched['spearman_rho']:.3f} | [{q3_matched['spearman_ci_low']:.3f}, {q3_matched['spearman_ci_high']:.3f}] | {q3_matched['pearson_r']:.3f} | {q3_matched['kendall_tau_b']:.3f} |
| Qwen2.5 Avg（同一主队列） | {q25_matched['n']} | {q25_matched['spearman_rho']:.3f} | [{q25_matched['spearman_ci_low']:.3f}, {q25_matched['spearman_ci_high']:.3f}] | {q25_matched['pearson_r']:.3f} | {q25_matched['kendall_tau_b']:.3f} |
| Qwen3 Avg（最大覆盖） | {q3_max['n']} | {q3_max['spearman_rho']:.3f} | [{q3_max['spearman_ci_low']:.3f}, {q3_max['spearman_ci_high']:.3f}] | {q3_max['pearson_r']:.3f} | {q3_max['kendall_tau_b']:.3f} |
| Qwen3 Avg（continuous only） | {q3_cont['n']} | {q3_cont['spearman_rho']:.3f} | [{q3_cont['spearman_ci_low']:.3f}, {q3_cont['spearman_ci_high']:.3f}] | {q3_cont['pearson_r']:.3f} | {q3_cont['kendall_tau_b']:.3f} |
| Qwen3 Avg（discrete only） | {q3_disc['n']} | {q3_disc['spearman_rho']:.3f} | [{q3_disc['spearman_ci_low']:.3f}, {q3_disc['spearman_ci_high']:.3f}] | {q3_disc['pearson_r']:.3f} | {q3_disc['kendall_tau_b']:.3f} |
| 两 Qwen Avg（10-epoch 子集） | {history['n']} | {history['spearman_rho']:.3f} | [{history['spearman_ci_low']:.3f}, {history['spearman_ci_high']:.3f}] | {history['pearson_r']:.3f} | {history['kendall_tau_b']:.3f} |
| 两 Qwen Avg（只有最终 probing 的子集） | {final_only['n']} | {final_only['spearman_rho']:.3f} | [{final_only['spearman_ci_low']:.3f}, {final_only['spearman_ci_high']:.3f}] | {final_only['pearson_r']:.3f} | {final_only['kendall_tau_b']:.3f} |

“完整历史”子集 rho={history['spearman_rho']:.3f}，“只有最终分数”子集 rho={final_only['spearman_rho']:.3f}，说明 0.944 不是由两类 probing 数据源混合才人为造成的。但后者仅 n={final_only['n']}，区间会更不稳定。

高相关也不依赖任务的原始分数量纲：22 个任务单元直接平均 rho={aggregation['Raw mean across 22 task cells']['spearman_rho']:.3f}，先对每个任务 z-score 再平均为 {aggregation['Mean after per-task z-scoring']['spearman_rho']:.3f}，任务内 rank 再平均为 {aggregation['Mean of within-task ranks']['spearman_rho']:.3f}，z-score 中位数为 {aggregation['Median after per-task z-scoring']['spearman_rho']:.3f}。每次留掉 22 个任务中的一个，rho 只在 [{aggregation_leave_min:.3f}, {aggregation_leave_max:.3f}] 之间，因此不是某一个任务单独驱动。逐项结果见 [`data/task_aggregation_robustness.csv`](data/task_aggregation_robustness.csv)。

再逐一删除 tokenizer，rho 范围为 [{min(leave_one_rhos):.3f}, {max(leave_one_rhos):.3f}]，主结果不由任何单点驱动。

![Robustness](figures/06_family_and_source_robustness.png)

## 具体 Qwen backbone 与任务

为了直接比较两个 Qwen，热图固定用同一批 n=38 tokenizer；CSV 表另外保留 Qwen3 的最大覆盖口径（通常 n=44）。

- Qwen3：任务级 rho 从 {q3_worst['task']}={q3_worst['matched_spearman_rho']:.3f} 到 {q3_best['task']}={q3_best['matched_spearman_rho']:.3f}，11 项中 {q3_sig}/11 在未校正 p<0.05。
- Qwen2.5：从 {q25_worst['task']}={q25_worst['matched_spearman_rho']:.3f} 到 {q25_best['task']}={q25_best['matched_spearman_rho']:.3f}，11/11 在未校正 p<0.05。
- 两 backbone 的同名任务先平均后，从 {combined_worst['task']}={combined_worst['matched_spearman_rho']:.3f} 到 {combined_best['task']}={combined_best['matched_spearman_rho']:.3f}。
- 同队列下，Qwen2.5 在 11/11 个任务上的 rho 都高于 Qwen3。Flickr、COCO、TextVQA/VQAv2 等任务与视觉 tokenizer 表征质量的相关性最高；MMMU 最弱，说明多学科推理能力的瓶颈不只是视觉表征。这是相关性解释，不是因果结论。

按各 backbone 的最大可用队列对 22 个任务检验统一做 Benjamini-Hochberg 校正后，{bh_significant}/22 仍显著。

![Task correlations](figures/05_task_correlation_heatmap.png)

完整数值、自助区间、p 值与 22 项检验的 Benjamini-Hochberg q 值见 [`data/task_correlations.csv`](data/task_correlations.csv)。Qwen3/ScienceQA 的最大覆盖分析保守排除了 I-JEPA，因为该行存在明确的 Avg/任务均值冲突；匹配 n=38 队列本来就不含 I-JEPA。Qwen3 Avg 最大覆盖的结果对此不敏感：含 I-JEPA 时 rho={q3_max['spearman_rho']:.3f}，排除后 rho={q3_max_without_ijepa['spearman_rho']:.3f}。

## 10 个 epoch：分数、排名与早停信号

下表是每个 probing epoch 与下游 MLLM Avg 的 Spearman rho。“匹配”三列固定用同一批 n=29 continuous tokenizer，可公平比较两个 Qwen；Qwen3 最大覆盖列另外纳入 4 个缺 Qwen2.5 的 discrete tokenizer，n=33。

| Epoch | Qwen3 Avg（最大覆盖 n=33） | Qwen3 Avg（匹配 n=29） | Qwen2.5 Avg（匹配 n=29） | 两 Qwen 公平 Avg（n=29） |
|---:|---:|---:|---:|---:|
{chr(10).join(epoch_spearman_lines)}

对应的 p 值、每轮平均/中位准确率以及与 Epoch 10 的排名稳定性见 [`data/epoch_metrics.csv`](data/epoch_metrics.csv)。

![Epoch trajectories](figures/01_epoch_accuracy_trajectories.png)

下图按 10 轮平均准确率对同一批 33 个 tokenizer 排序，横轴隐去具体名称；每个 tokenizer 的 10 个细点分别对应 Epoch 1–10，浅色带表示 Epoch 1 到 Epoch 10 的增益范围。

![Epochs by tokenizer](figures/01b_epoch_by_tokenizer_overview.png)

![Epoch rank heatmap](figures/02_epoch_rank_heatmap.png)

33 条完整轨迹中，E1→E10 提升的中位数是 {np.median(gains):.2f} 个点。Epoch 1 与 Epoch 10 排名已有 rho={epoch1['rank_stability_vs_epoch10_n33']:.3f}，到 Epoch 9 为 {epoch_metrics[8]['rank_stability_vs_epoch10_n33']:.3f}。对完整两-Qwen 子集，Epoch 1 对 MLLM Avg 的 rho={epoch1['spearman_vs_fair_mllm_avg_n29']:.3f}，Epoch 10 为 {epoch10['spearman_vs_fair_mllm_avg_n29']:.3f}，差 {epoch_delta:.3f}（配对 bootstrap 95% CI [{epoch_delta_low:.3f}, {epoch_delta_high:.3f}]）。区间跨 0，没有证据说 10 轮比 1 轮更能预测 MLLM；且相关不随 epoch 单调提升。

这表明：在当前 continuous 模型范围内，**1 个 epoch 已经可以做粗筛排名**；但不能用它替代最终分数。{biggest_movers[0][0]['tokenizer']} 和 {biggest_movers[1][0]['tokenizer']} 从 E1 到 E10 都上升了 {abs(biggest_movers[0][1]):.0f} 个名次，且 TokLIP 的 E1→E10 增益分别约 8.3 点；早停对这类收敛慢的 discrete tokenizer 不公平。

![Epoch predictiveness](figures/03_epoch_predictiveness.png)

轨迹的“增益”本身与 MLLM Avg 负相关（rho={gain_vs_target:.3f}），主要因为低起点模型可上升空间更大；gain 与 E10-only 全样本线性拟合所得 MLLM 残差的 rho={gain_vs_residual:.3f}，几乎无关。LOOCV 也一致：E1-only MAE={epoch1_cv['mae']:.2f}，E10-only MAE={epoch10_cv['mae']:.2f}，E10+增益 MAE={gain_cv['mae']:.2f}；目前没证据说 10 轮动力学比单个准确率能额外预测 MLLM。

## 家族内部与受控对比

| 家族 | n | 家族内 Spearman rho | 95% CI |
|---|---:|---:|---:|
{chr(10).join(family_lines)}

把全局秩在家族内去均值后，pooled family-adjusted 关联仍为 rho={adjusted['spearman_rho']:.3f}（n={adjusted['n']}，家族内置换 p={p_text(adjusted['spearman_p_permutation'])}）。所以整体高相关不只是 SigLIP2/MetaCLIP 家族均值之间的差异。OpenAI CLIP 只有 1 个点，不能算家族内相关。

不过，当只在高分段做精细选型时，关系会因取值范围收窄而变弱：probing 排名前半 n={len(top_half)} 的 rho={top_half_rho:.3f}；probing≥87 的 n={len(high_probe)} 个点中 rho={float(high_probe_result.statistic):.3f}，p={p_text(float(high_probe_result.pvalue))}。所以 0.944 更适合解读为跨较广质量范围的排名 proxy，不是顶尖模型之间细微差异的完美判别器。

更严格地固定架构、只比较分辨率升级时，{positive_both}/{len(resolution_rows)} 对都同时提高 probing 和 MLLM Avg；但两种增益幅度的 Spearman 只有 {delta_rho:.3f}（n={len(resolution_rows)}，p={p_text(delta_p)}）。因此 probing 对“方向”很好，却不宜解读为局部改动的精确增益估计器。n={len(resolution_rows)} 且对比对来自两个家族，这一点应视为探索性结论。

![Controlled deltas](figures/08_controlled_resolution_deltas.png)

## “预测”而不只是同样本相关

用一元线性校准 `MLLM Avg ~ probing`：

- leave-one-tokenizer-out（n={loo['n']}）：MAE={loo['mae']:.2f}，RMSE={loo['rmse']:.2f}，R²cv={loo['r2_cv']:.3f}；不用 probing 的训练集均值 baseline MAE={loo['baseline_mae']:.2f}。
- leave-one-family-out（只评估三个有重复样本的主家族，n={lofo['n']}）：MAE={lofo['mae']:.2f}，RMSE={lofo['rmse']:.2f}，R²cv={lofo['r2_cv']:.3f}；baseline MAE={lofo['baseline_mae']:.2f}。

这说明 probing 不只能在全样本上“拟合得好看”；对留出点、甚至留出整个家族仍有明显预测信号。但家族只有 SigLIP2、MetaCLIP1、MetaCLIP2 三个大组加一个单点 CLIP，leave-family-out 数字仍需要新家族验证。

{chr(10).join(topk_lines)}

全部 {concordant + discordant} 个非并列 tokenizer pair 中，{concordant} 对的 probing 与 MLLM 排序方向一致，即 {pairwise_accuracy * 100:.1f}%；另有 {tied} 对至少一边并列。

![Prediction validation](figures/07_prediction_validation.png)

## 数据质量问题与不可越过的边界

1. **不能直接用主表 `Avg` 列的缺失行。** 6 个只有 Qwen3 的 tokenizer 被写成 `Qwen3 Avg / 2`，DINOv3 两个 MLLM 都缺却写成 0.00。脚本已将这些值当 NA，只在两 backbone 都齐时重算公平 Avg。
2. **I-JEPA 行内不一致。** 11 个 Qwen3 任务的算术均值是 35.09，CSV Avg 是 36.56，差 1.47；其 VQAv2 和 ScienceQA 又恰好都为 47.08，建议回查原始日志。主 n=38 分析不含 I-JEPA，因此不受影响。
3. **完整 epoch 与最终分数来源。** 33 个 epoch 表的 E10 与主表逐项完全一致；另有 {len(no_history)} 个只有主表最终分数，以及 {len(no_probe)} 个完全没有 probing。脚本不伪造前 9 轮，轨迹分析只用真实 33 个点。
4. **不是因果证明。** 容量、预训练数据、分辨率和 tokenizer 家族同时影响 probing 与 MLLM；家族内分析能缓解，不能消除所有混杂。
5. **相关性不代表差值可直接换算。** 整体排名很强，但局部分辨率增益的幅度相关很弱，且个别残差可超过 4 个 MLLM 分数点。

## 文件索引与复现

- [`analyze.py`](analyze.py)：唯一分析脚本；解析、审计、统计、画图和报告生成都在这里。
- [`data/analysis_cohort.csv`](data/analysis_cohort.csv) 与 [`data/exclusions.csv`](data/exclusions.csv)：47 个 tokenizer 的合并口径、入选标记与逐项排除原因。
- [`data/correlation_summary.csv`](data/correlation_summary.csv)：主相关性与不同队列。
- [`data/task_aggregation_robustness.csv`](data/task_aggregation_robustness.csv)：任务标准化、rank 聚合与 leave-one-task-out。
- [`data/task_correlations.csv`](data/task_correlations.csv)：两个 Qwen 的逐任务结果。
- [`data/family_robustness.csv`](data/family_robustness.csv)：家族内、留一家族和来源敏感性。
- [`data/epoch_metrics.csv`](data/epoch_metrics.csv)：逐 epoch 排名稳定性与 MLLM 相关性。
- [`data/prediction_metrics.csv`](data/prediction_metrics.csv) 与 [`data/prediction_diagnostics.csv`](data/prediction_diagnostics.csv)：样本外误差及逐 tokenizer 残差。
- [`data/controlled_comparisons.csv`](data/controlled_comparisons.csv)：分辨率、预训练规模和 MT5 的成对对比。

复现命令（需 NumPy、SciPy、Matplotlib）：

```bash
conda run -n TokBench python result/tokenizer_mllm_analysis/analyze.py
```

所有 bootstrap/permutation 都使用固定种子 {SEED}，重跑可得到相同结果。
"""
    (HERE / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    set_plot_style()
    records, _epochs, audit = load_records()
    by_name = {record["tokenizer"]: record for record in records}
    core = [record for record in records if record["in_primary_n38"]]
    q3_max = [record for record in records if record["in_qwen3_max_n44"]]
    epoch_records = [record for record in records if record["probe_history_status"] == "full_10_epoch"]
    epoch_core = [record for record in records if record["in_epoch_combined_n29"]]
    final_only_core = [record for record in core if record["probe_history_status"] == "final_only"]

    if len(records) != 47 or len(core) != 38 or len(q3_max) != 44 or len(epoch_records) != 33 or len(epoch_core) != 29:
        raise ValueError("Unexpected cohort size; inspect the inputs before interpreting regenerated results")
    if audit["epoch10_mismatches"]:
        raise ValueError(f"Epoch-10 values disagree with main table: {audit['epoch10_mismatches']}")

    # Test whether the headline result depends on raw task scales.  The four
    # aggregate targets use exactly the same 38 x 22 complete task matrix.
    core_task_matrix = np.asarray(
        [
            [record["qwen3_tasks"][task] for task in TASKS]
            + [record["qwen25_tasks"][task] for task in TASKS]
            for record in core
        ],
        dtype=float,
    )
    task_z = (core_task_matrix - np.mean(core_task_matrix, axis=0)) / np.std(core_task_matrix, axis=0)
    task_rank = np.column_stack(
        [stats.rankdata(core_task_matrix[:, index], method="average") for index in range(core_task_matrix.shape[1])]
    )
    aggregate_values = {
        "_agg_raw_22_task_mean": np.mean(core_task_matrix, axis=1),
        "_agg_mean_task_z": np.mean(task_z, axis=1),
        "_agg_mean_task_rank": np.mean(task_rank, axis=1),
        "_agg_median_task_z": np.median(task_z, axis=1),
    }
    for key, values in aggregate_values.items():
        for record, value in zip(core, values):
            record[key] = float(value)
    aggregation_rows = [
        correlation_row("Raw mean across 22 task cells", core, "_agg_raw_22_task_mean", "Same cohort; recomputed from detailed task cells."),
        correlation_row("Mean after per-task z-scoring", core, "_agg_mean_task_z", "Removes task scale and variance differences."),
        correlation_row("Mean of within-task ranks", core, "_agg_mean_task_rank", "Uses only each task's tokenizer ordering."),
        correlation_row("Median after per-task z-scoring", core, "_agg_median_task_z", "Robust aggregation across the 22 task cells."),
    ]
    for column in range(core_task_matrix.shape[1]):
        backbone = "Qwen3" if column < len(TASKS) else "Qwen2.5"
        task = TASKS[column % len(TASKS)]
        key = "_agg_leave_one_task_out"
        values = np.mean(np.delete(core_task_matrix, column, axis=1), axis=1)
        for record, value in zip(core, values):
            record[key] = float(value)
        aggregation_rows.append(
            correlation_row(f"Leave out {backbone}/{task}", core, key, "One of 22 task cells removed before averaging.")
        )

    summary_rows = [
        correlation_row(
            "Primary: fair two-backbone avg",
            core,
            "mllm_avg_fair",
            "Complete-case intersection: ImageNet probe plus both Qwen backbone averages; all 38 are continuous.",
        ),
        correlation_row("Qwen3 Avg (matched n=38)", core, "qwen3_avg_reported", "Same tokenizer cohort as the primary result."),
        correlation_row("Qwen2.5 Avg (matched n=38)", core, "qwen25_avg_reported", "Same tokenizer cohort as the primary result."),
        correlation_row(
            "Qwen3 Avg (maximum coverage)",
            q3_max,
            "qwen3_avg_reported",
            "Maximum coverage: 40 continuous plus 4 discrete tokenizers; I-JEPA uses its reported Avg and is checked by exclusion sensitivity.",
        ),
        correlation_row(
            "Qwen3 Avg (maximum coverage, excluding I-JEPA)",
            [record for record in q3_max if record["tokenizer"] != "I-JEPA"],
            "qwen3_avg_reported",
            "Sensitivity to I-JEPA's inconsistent task mean versus reported Avg.",
        ),
        correlation_row(
            "Qwen3 Avg (continuous only)",
            [record for record in q3_max if record["visual_token_type"] == "continuous"],
            "qwen3_avg_reported",
            "Maximum continuous-only coverage.",
        ),
        correlation_row(
            "Qwen3 Avg (discrete only)",
            [record for record in q3_max if record["visual_token_type"] == "discrete"],
            "qwen3_avg_reported",
            "Exploratory only: four tokenizers and no Qwen2.5 results.",
        ),
        correlation_row(
            "Two-Qwen Avg (full 10-epoch histories)",
            epoch_core,
            "mllm_avg_fair",
            "Restricts the primary analysis to tokenizers with all ten probing epochs.",
        ),
        correlation_row(
            "Two-Qwen Avg (final-only probing)",
            final_only_core,
            "mllm_avg_fair",
            "Tokenizers with a final probe in the main table but no earlier epochs in the Markdown file.",
        ),
    ]
    summary_by_label = {row["analysis"]: row for row in summary_rows}

    robustness_rows = []
    for source_label, robustness_label in [
        ("Primary: fair two-backbone avg", "Primary overall"),
        ("Two-Qwen Avg (full 10-epoch histories)", "Full 10-epoch source only"),
        ("Two-Qwen Avg (final-only probing)", "Final-only source only"),
    ]:
        reused = dict(summary_by_label[source_label])
        reused["analysis"] = robustness_label
        robustness_rows.append(reused)
    for family in ["SigLIP2", "MetaCLIP1", "MetaCLIP2"]:
        robustness_rows.append(correlation_row(f"Within {family}", [record for record in core if record["model_family"] == family], "mllm_avg_fair"))
    for family in ["SigLIP2", "MetaCLIP1", "MetaCLIP2", "OpenAI CLIP"]:
        robustness_rows.append(correlation_row(f"Leave {family} out", [record for record in core if record["model_family"] != family], "mllm_avg_fair"))
    robustness_rows.append(family_adjusted_spearman(core))

    task_rows = []
    for backbone, task_key in [("Qwen3-1.7B", "qwen3_tasks"), ("Qwen2.5-1.5B", "qwen25_tasks")]:
        for task in TASKS:
            maximum = [
                record
                for record in records
                if is_finite(record["probe_final"])
                and is_finite(record[task_key][task])
                and not (backbone == "Qwen3-1.7B" and task == "ScienceQA" and record["tokenizer"] == "I-JEPA")
            ]
            matched = [record for record in core if is_finite(record[task_key][task])]
            max_x = [record["probe_final"] for record in maximum]
            max_y = [record[task_key][task] for record in maximum]
            matched_x = [record["probe_final"] for record in matched]
            matched_y = [record[task_key][task] for record in matched]
            max_result = stats.spearmanr(max_x, max_y)
            matched_result = stats.spearmanr(matched_x, matched_y)
            ci_low, ci_high = bootstrap_corr_ci(max_x, max_y, "spearman", f"task-{backbone}-{task}")
            task_rows.append(
                {
                    "backbone": backbone,
                    "task": task,
                    "max_n": len(maximum),
                    "max_spearman_rho": float(max_result.statistic),
                    "max_ci_low": ci_low,
                    "max_ci_high": ci_high,
                    "max_p": float(max_result.pvalue),
                    "max_q_bh_22": float("nan"),
                    "matched_n": len(matched),
                    "matched_spearman_rho": float(matched_result.statistic),
                    "matched_p": float(matched_result.pvalue),
                    "note": "I-JEPA excluded due row inconsistency" if backbone == "Qwen3-1.7B" and task == "ScienceQA" else "",
                }
            )
    qvalues = benjamini_hochberg([row["max_p"] for row in task_rows])
    for row, qvalue in zip(task_rows, qvalues):
        row["max_q_bh_22"] = qvalue
    for task in TASKS:
        x = [record["probe_final"] for record in core]
        y = [float(np.mean([record["qwen3_tasks"][task], record["qwen25_tasks"][task]])) for record in core]
        result = stats.spearmanr(x, y)
        ci_low, ci_high = bootstrap_corr_ci(x, y, "spearman", f"task-combined-{task}")
        task_rows.append(
            {
                "backbone": "Two-Qwen task mean",
                "task": task,
                "max_n": len(core),
                "max_spearman_rho": float(result.statistic),
                "max_ci_low": ci_low,
                "max_ci_high": ci_high,
                "max_p": float(result.pvalue),
                "max_q_bh_22": float("nan"),
                "matched_n": len(core),
                "matched_spearman_rho": float(result.statistic),
                "matched_p": float(result.pvalue),
                "note": "Matched descriptive aggregate; excluded from the 22 backbone-specific BH tests.",
            }
        )

    accuracy_matrix = np.asarray([record["epochs"] for record in epoch_records], dtype=float)
    final_accuracy = accuracy_matrix[:, -1]
    q3_epoch = np.asarray([record["qwen3_avg_reported"] for record in epoch_records])
    epoch_core_names = {record["tokenizer"] for record in epoch_core}
    epoch_metrics = []
    for index in range(10):
        current = accuracy_matrix[:, index]
        matched_records = [record for record in epoch_records if record["tokenizer"] in epoch_core_names]
        matched_current = [record["epochs"][index] for record in matched_records]
        matched_q3 = [record["qwen3_avg_reported"] for record in matched_records]
        matched_q25 = [record["qwen25_avg_reported"] for record in matched_records]
        matched_fair_avg = [record["mllm_avg_fair"] for record in matched_records]
        q3_max_result = stats.spearmanr(current, q3_epoch)
        q3_matched_result = stats.spearmanr(matched_current, matched_q3)
        q25_matched_result = stats.spearmanr(matched_current, matched_q25)
        fair_avg_result = stats.spearmanr(matched_current, matched_fair_avg)
        epoch_metrics.append(
            {
                "epoch": index + 1,
                "n_all_histories": len(epoch_records),
                "mean_accuracy": float(np.mean(current)),
                "median_accuracy": float(np.median(current)),
                "rank_stability_vs_epoch10_n33": float(stats.spearmanr(current, final_accuracy).statistic),
                "spearman_vs_qwen3_avg_n33": float(q3_max_result.statistic),
                "p_vs_qwen3_avg_n33": float(q3_max_result.pvalue),
                "n_fair_mllm_avg": len(epoch_core),
                "spearman_vs_qwen3_avg_matched_n29": float(q3_matched_result.statistic),
                "p_vs_qwen3_avg_matched_n29": float(q3_matched_result.pvalue),
                "spearman_vs_qwen25_avg_n29": float(q25_matched_result.statistic),
                "p_vs_qwen25_avg_n29": float(q25_matched_result.pvalue),
                "spearman_vs_fair_mllm_avg_n29": float(fair_avg_result.statistic),
                "p_vs_fair_mllm_avg_n29": float(fair_avg_result.pvalue),
            }
        )

    diagnostics, prediction_metrics = cross_validated_predictions(core)
    trajectory_rows, gain_vs_target, gain_vs_residual = trajectory_cv_metrics(epoch_core)
    prediction_metrics.extend(trajectory_rows)

    controlled_pairs = [
        ("resolution", "SigLIP2 b16 224→512", "siglip2_b16_224", "siglip2_b16_512"),
        ("resolution", "SigLIP2 sm14 224→384", "siglip2_sm14_224", "siglip2_sm14_384"),
        ("resolution", "SigLIP2 sm16 256→384", "siglip2_sm16_256", "siglip2_sm16_384"),
        ("resolution", "SigLIP2 l16 256→384", "siglip2_l16_256", "siglip2_l16_384"),
        ("resolution", "SigLIP2 g16 256→384", "siglip2_g16_256", "siglip2_g16_384"),
        ("resolution", "MetaCLIP2 g14 224→378", "mc2_g14_224", "mc2_g14_378"),
        ("resolution", "MetaCLIP2 b16 224→384", "mc2_b16_224", "mc2_b16_384"),
        ("resolution", "MetaCLIP2 b32 224→384", "mc2_b32_224", "mc2_b32_384"),
        ("resolution", "MetaCLIP2 m16 224→384", "mc2_m16_224", "mc2_m16_384"),
        ("resolution", "MetaCLIP2 s16 224→384", "mc2_s16_224", "mc2_s16_384"),
        ("pretraining_scale", "MetaCLIP1 b32 400m→2.5b", "mc1_b32_224_400m", "mc1_b32_224_2.5b"),
        ("pretraining_scale", "MetaCLIP1 b16 400m→2.5b", "mc1_b16_224_400m", "mc1_b16_224_2.5b"),
        ("pretraining_scale", "MetaCLIP1 l14 400m→2.5b", "mc1_l14_224_400m", "mc1_l14_224_2.5b"),
        ("mt5_variant", "MetaCLIP2 s16 base→mt5", "mc2_s16_224", "mc2_s16_224_mt5"),
        ("mt5_variant", "MetaCLIP2 m16 base→mt5", "mc2_m16_224", "mc2_m16_224_mt5"),
        ("mt5_variant", "MetaCLIP2 b32 base→mt5", "mc2_b32_224", "mc2_b32_224_mt5"),
    ]
    controlled_rows = []
    for comparison_type, label, baseline_name, changed_name in controlled_pairs:
        baseline = by_name[baseline_name]
        changed = by_name[changed_name]
        controlled_rows.append(
            {
                "comparison_type": comparison_type,
                "pair_label": label,
                "model_family": baseline["model_family"],
                "baseline_tokenizer": baseline_name,
                "changed_tokenizer": changed_name,
                "baseline_probe": baseline["probe_final"],
                "changed_probe": changed["probe_final"],
                "delta_probe": changed["probe_final"] - baseline["probe_final"],
                "baseline_mllm_avg": baseline["mllm_avg_fair"],
                "changed_mllm_avg": changed["mllm_avg_fair"],
                "delta_mllm_avg": changed["mllm_avg_fair"] - baseline["mllm_avg_fair"],
            }
        )

    cohort_rows = []
    for record in records:
        if not is_finite(record["probe_final"]):
            primary_reason = "missing ImageNet probe"
        elif not is_finite(record["qwen3_avg_reported"]):
            primary_reason = "missing Qwen3 and Qwen2.5"
        elif not is_finite(record["qwen25_avg_reported"]):
            primary_reason = "missing Qwen2.5"
        else:
            primary_reason = "included"
        row = {
            "tokenizer": record["tokenizer"],
            "visual_token_type": record["visual_token_type"],
            "model_family": record["model_family"],
            "probe_final": record["probe_final"],
            "probe_history_status": record["probe_history_status"],
            **{f"epoch_{index + 1}": value for index, value in enumerate(record["epochs"])},
            "qwen3_avg_reported": record["qwen3_avg_reported"],
            "qwen3_avg_recomputed_from_tasks": record["qwen3_avg_recomputed"],
            "qwen25_avg_reported": record["qwen25_avg_reported"],
            "qwen25_avg_recomputed_from_tasks": record["qwen25_avg_recomputed"],
            "mllm_avg_reported_raw": record["mllm_avg_reported_raw"],
            "mllm_avg_fair": record["mllm_avg_fair"],
            "mllm_status": record["mllm_status"],
            "in_primary_n38": record["in_primary_n38"],
            "in_qwen3_max_n44": record["in_qwen3_max_n44"],
            "in_epoch_combined_n29": record["in_epoch_combined_n29"],
            "primary_status_or_exclusion": primary_reason,
        }
        cohort_rows.append(row)

    exclusions = []
    for record in records:
        if not record["in_primary_n38"]:
            if not is_finite(record["probe_final"]):
                reason = "ImageNet linear probe missing"
            elif not is_finite(record["qwen3_avg_reported"]):
                reason = "Both Qwen3 and Qwen2.5 MLLM results missing"
            else:
                reason = "Qwen2.5 MLLM result missing; raw main-table Avg is not a fair two-backbone average"
            exclusions.append({"analysis": "Primary two-Qwen average (n=38)", "tokenizer": record["tokenizer"], "reason": reason})
        if record["probe_history_status"] != "full_10_epoch":
            reason = (
                "Final probing exists, but Epochs 1-9 are absent from the epoch Markdown"
                if record["probe_history_status"] == "final_only"
                else "ImageNet probing was not run / is entirely missing"
            )
            exclusions.append({"analysis": "Ten-epoch trajectory (n=33)", "tokenizer": record["tokenizer"], "reason": reason})
        if not record["in_qwen3_max_n44"]:
            reason = "ImageNet linear probe missing" if not is_finite(record["probe_final"]) else "Qwen3 MLLM result missing"
            exclusions.append({"analysis": "Qwen3 maximum coverage (n=44)", "tokenizer": record["tokenizer"], "reason": reason})
    exclusions.append(
        {
            "analysis": "Qwen3 ScienceQA maximum coverage (n=43)",
            "tokenizer": "I-JEPA",
            "reason": "Conservative exclusion: Qwen3 task mean (35.09) conflicts with reported Avg (36.56), and ScienceQA duplicates VQAv2 at 47.08",
        }
    )

    cohort_fields = [
        "tokenizer",
        "visual_token_type",
        "model_family",
        "probe_final",
        "probe_history_status",
        *[f"epoch_{index}" for index in range(1, 11)],
        "qwen3_avg_reported",
        "qwen3_avg_recomputed_from_tasks",
        "qwen25_avg_reported",
        "qwen25_avg_recomputed_from_tasks",
        "mllm_avg_reported_raw",
        "mllm_avg_fair",
        "mllm_status",
        "in_primary_n38",
        "in_qwen3_max_n44",
        "in_epoch_combined_n29",
        "primary_status_or_exclusion",
    ]
    corr_fields = [
        "analysis",
        "target",
        "n",
        "spearman_rho",
        "spearman_ci_low",
        "spearman_ci_high",
        "spearman_p_asymptotic",
        "spearman_p_permutation",
        "pearson_r",
        "pearson_p",
        "kendall_tau_b",
        "kendall_p",
        "note",
    ]
    task_fields = [
        "backbone",
        "task",
        "max_n",
        "max_spearman_rho",
        "max_ci_low",
        "max_ci_high",
        "max_p",
        "max_q_bh_22",
        "matched_n",
        "matched_spearman_rho",
        "matched_p",
        "note",
    ]
    epoch_fields = list(epoch_metrics[0])
    prediction_fields = list(diagnostics[0])
    prediction_metric_fields = list(prediction_metrics[0])
    controlled_fields = list(controlled_rows[0])

    write_csv(DATA_DIR / "analysis_cohort.csv", cohort_fields, cohort_rows)
    write_csv(DATA_DIR / "exclusions.csv", ["analysis", "tokenizer", "reason"], exclusions)
    write_csv(DATA_DIR / "correlation_summary.csv", corr_fields, summary_rows)
    write_csv(DATA_DIR / "task_aggregation_robustness.csv", corr_fields, aggregation_rows)
    write_csv(DATA_DIR / "family_robustness.csv", corr_fields, robustness_rows)
    write_csv(DATA_DIR / "task_correlations.csv", task_fields, task_rows)
    write_csv(DATA_DIR / "epoch_metrics.csv", epoch_fields, epoch_metrics)
    write_csv(DATA_DIR / "prediction_diagnostics.csv", prediction_fields, diagnostics)
    write_csv(DATA_DIR / "prediction_metrics.csv", prediction_metric_fields, prediction_metrics)
    write_csv(DATA_DIR / "controlled_comparisons.csv", controlled_fields, controlled_rows)

    plot_epoch_trajectories(epoch_records)
    plot_epochs_by_tokenizer(epoch_records)
    plot_epoch_rank_heatmap(epoch_records)
    plot_epoch_predictiveness(epoch_metrics, epoch_records)
    plot_probe_vs_mllm(core, summary_by_label)
    plot_task_heatmap(task_rows, summary_by_label)
    plot_robustness(robustness_rows[:-1])
    plot_prediction_validation(diagnostics, prediction_metrics)
    plot_controlled_deltas(controlled_rows)
    write_report(
        records,
        audit,
        summary_rows,
        aggregation_rows,
        robustness_rows,
        task_rows,
        epoch_metrics,
        prediction_metrics,
        controlled_rows,
        gain_vs_target,
        gain_vs_residual,
    )

    print(f"Wrote analysis to {HERE}")
    print(f"Primary n={len(core)}, Spearman rho={summary_by_label['Primary: fair two-backbone avg']['spearman_rho']:.6f}")


if __name__ == "__main__":
    main()
