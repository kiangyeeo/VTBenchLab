#!/usr/bin/env python3
"""Reproducible analysis of ImageNet linear probing vs. downstream MLLM scores.

Inputs (kept untouched):
  ../Tokenizer Accuracy by Epoch.md
  ../VisualTokenizer表现 - 主表 (1).csv
  ../VisualTokenizer表现 - MLLM详细结果 (1).csv

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


MAIN_CSV = RESULT_DIR / "VisualTokenizer表现 - 主表 (1).csv"
DETAIL_CSV = RESULT_DIR / "VisualTokenizer表现 - MLLM详细结果 (1).csv"
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

PLOT_GROUP_COLORS = {
    "SigLIP2": "#0072B2",
    "MetaCLIP1": "#D55E00",
    "MetaCLIP2": "#009E73",
    "OpenAI CLIP": "#CC79A7",
    "Other continuous": "#666666",
    "Discrete": "#E69F00",
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


def plot_group(record: dict) -> str:
    if record["model_family"] in {"SigLIP2", "MetaCLIP1", "MetaCLIP2", "OpenAI CLIP"}:
        return record["model_family"]
    if record["visual_token_type"] == "discrete":
        return "Discrete"
    return "Other continuous"


def read_two_header_csv(path: Path) -> tuple[list[list[str]], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3:
        raise ValueError(f"CSV has too few rows: {path}")
    return rows[:2], rows[2:]


def header_column(headers: Sequence[Sequence[str]], group: str, field: str) -> int:
    """Locate a two-row spreadsheet header, honoring merged cells exported as blanks."""
    if len(headers) != 2:
        raise ValueError("Expected exactly two CSV header rows")
    top: list[str] = []
    active = ""
    for value in headers[0]:
        if value.strip():
            active = value.strip()
        top.append(active)
    matches = [
        index
        for index, (section, subfield) in enumerate(zip(top, headers[1]))
        if section == group and subfield.strip() == field
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one column for {group!r}/{field!r}, found {matches}")
    return matches[0]


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
    main_headers, main_rows = read_two_header_csv(MAIN_CSV)
    detail_headers, detail_rows = read_two_header_csv(DETAIL_CSV)
    epochs = read_epochs(EPOCH_MD)

    main_columns = {
        "probe": header_column(main_headers, "Probing", "ImageNet"),
        "q3": header_column(main_headers, "MLLM Avg Performance", "qwen3 1.7B"),
        "q25": header_column(main_headers, "MLLM Avg Performance", "qwen2.5 1.5B"),
        "combined": header_column(main_headers, "MLLM Avg Performance", "Avg"),
    }
    detail_q3_columns = {task: header_column(detail_headers, "qwen3 1.7B", task) for task in TASKS}
    detail_q25_columns = {task: header_column(detail_headers, "qwen2.5 1.5B", task) for task in TASKS}
    detail_q3_avg = header_column(detail_headers, "qwen3 1.7B", "Avg")
    detail_q25_avg = header_column(detail_headers, "qwen2.5 1.5B", "Avg")

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

        q3_tasks = [as_float(detail[detail_q3_columns[task]]) for task in TASKS]
        q25_tasks = [as_float(detail[detail_q25_columns[task]]) for task in TASKS]
        q3_reported = as_float(main[main_columns["q3"]])
        q25_reported = as_float(main[main_columns["q25"]])
        q3_detail_reported = as_float(detail[detail_q3_avg])
        q25_detail_reported = as_float(detail[detail_q25_avg])
        q3_recomputed = mean_if_complete(q3_tasks)
        q25_recomputed = mean_if_complete(q25_tasks)
        probe = as_float(main[main_columns["probe"]])
        raw_combined = as_float(main[main_columns["combined"]])
        fair_combined = (
            float(np.mean([q3_reported, q25_reported]))
            if is_finite(q3_reported) and is_finite(q25_reported)
            else float("nan")
        )
        task_recomputed_combined = (
            float(np.mean([q3_recomputed, q25_recomputed]))
            if is_finite(q3_recomputed) and is_finite(q25_recomputed)
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
        if is_finite(raw_combined) and is_finite(fair_combined) and abs(raw_combined - fair_combined) > 0.006:
            avg_mismatches.append({"tokenizer": name, "field": "main-table combined Avg", "difference": raw_combined - fair_combined})

        history_status = "full_10_epoch" if name in epochs else ("final_only" if is_finite(probe) else "missing_probe")
        if is_finite(q3_reported) and is_finite(q25_reported):
            mllm_status = "both_complete"
        elif is_finite(q3_reported):
            mllm_status = "qwen3_only"
        elif is_finite(q25_reported):
            mllm_status = "qwen25_only"
        else:
            mllm_status = "missing_both"
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
            "mllm_avg_tasks_recomputed": task_recomputed_combined,
            "mllm_status": mllm_status,
            "qwen3_tasks": dict(zip(TASKS, q3_tasks)),
            "qwen25_tasks": dict(zip(TASKS, q25_tasks)),
            "epochs": epochs.get(name, [float("nan")] * 10),
        }
        record["in_primary"] = is_finite(probe) and is_finite(fair_combined)
        record["in_qwen3_max"] = is_finite(probe) and is_finite(q3_reported)
        record["in_qwen25_max"] = is_finite(probe) and is_finite(q25_reported)
        record["in_epoch_combined"] = name in epochs and is_finite(probe) and is_finite(fair_combined)
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
    singleton_families = sorted(family for family, count in grouped.items() if count < 2)
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
        "note": "Global ranks were centered within replicated families; singleton families excluded: " + ", ".join(singleton_families),
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

    lofo_all = np.full(n, np.nan)
    lofo_all_baseline = np.full(n, np.nan)
    for family in sorted(set(families)):
        test = families == family
        train = ~test
        lofo_all[test] = ols_predict(x[train, None], y[train], x[test, None])
        lofo_all_baseline[test] = float(np.mean(y[train]))

    # Retain a like-for-like stress test on the three large prefix families.
    major_families = {"SigLIP2", "MetaCLIP1", "MetaCLIP2"}
    major_mask = np.asarray([family in major_families for family in families])
    lofo_major = np.full(n, np.nan)
    lofo_major_baseline = np.full(n, np.nan)
    for family in sorted(major_families):
        test = families == family
        train = major_mask & ~test
        lofo_major[test] = ols_predict(x[train, None], y[train], x[test, None])
        lofo_major_baseline[test] = float(np.mean(y[train]))

    diagnostics = []
    for index, record in enumerate(records):
        diagnostics.append(
            {
                "tokenizer": record["tokenizer"],
                "model_family": record["model_family"],
                "visual_token_type": record["visual_token_type"],
                "probe_final": x[index],
                "mllm_avg_fair": y[index],
                "loocv_prediction": loo[index],
                "loocv_residual_observed_minus_predicted": y[index] - loo[index],
                "leave_family_out_prediction": lofo_all[index],
                "leave_family_out_residual_observed_minus_predicted": y[index] - lofo_all[index],
                "leave_major_family_out_prediction": lofo_major[index],
                "leave_major_family_out_residual_observed_minus_predicted": y[index] - lofo_major[index],
            }
        )

    metrics = []
    for label, prediction, baseline, mask in [
        ("Leave-one-tokenizer-out", loo, loo_baseline, np.ones(n, dtype=bool)),
        ("Leave-one-family-out", lofo_all, lofo_all_baseline, np.ones(n, dtype=bool)),
        ("Leave-one-major-prefix-family-out", lofo_major, lofo_major_baseline, major_mask),
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
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
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
        f"{len(epoch_records)} tokenizers with all 10 epochs",
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
            color = "white" if value > len(epoch_records) / 2 else "#101010"
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
    n_history = epoch_metrics[0]["n_all_histories"]
    n_mllm = epoch_metrics[0]["n_mllm_matched"]
    axes[0].plot(epoch_numbers, [row["rank_stability_vs_epoch10"] for row in epoch_metrics], marker="o", linewidth=2, label=f"Rank stability vs E10 (n={n_history})")
    axes[0].plot(epoch_numbers, [row["spearman_vs_qwen3_avg"] for row in epoch_metrics], marker="o", linewidth=2, label=f"Qwen3 Avg (n={n_mllm})")
    axes[0].plot(epoch_numbers, [row["spearman_vs_qwen25_avg"] for row in epoch_metrics], marker="o", linewidth=2, label=f"Qwen2.5 Avg (n={n_mllm})")
    axes[0].plot(epoch_numbers, [row["spearman_vs_fair_mllm_avg"] for row in epoch_metrics], marker="o", linewidth=2, label=f"Two-Qwen fair Avg (n={n_mllm})")
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
        ("qwen3_avg_reported", "Qwen3-1.7B Avg", "Qwen3 Avg (matched primary)"),
        ("qwen25_avg_reported", "Qwen2.5-1.5B Avg", "Qwen2.5 Avg (matched primary)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharex=True)
    for axis, (target, y_label, summary_label) in zip(axes, panels):
        x = np.asarray([record["probe_final"] for record in core_records])
        y = np.asarray([record[target] for record in core_records])
        for record in core_records:
            full_history = record["probe_history_status"] == "full_10_epoch"
            color = PLOT_GROUP_COLORS[plot_group(record)]
            axis.scatter(
                record["probe_final"],
                record[target],
                s=48,
                facecolor=color if full_history else "none",
                edgecolor="white" if full_history else color,
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

    legend_families = list(PLOT_GROUP_COLORS)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor=PLOT_GROUP_COLORS[family], markeredgecolor="white", label=family)
        for family in legend_families
    ]
    handles.extend(
        [
            Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor="#111111", markeredgecolor="#111111", label="Full 10-epoch history"),
            Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor="none", markeredgecolor="#111111", label="Final probe only"),
        ]
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.055))
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
    matrix[0, -1] = summary_by_label["Qwen3 Avg (matched primary)"]["spearman_rho"]
    matrix[1, -1] = summary_by_label["Qwen2.5 Avg (matched primary)"]["spearman_rho"]
    matrix[2, -1] = summary_by_label["Primary: fair two-backbone avg"]["spearman_rho"]
    q_matrix[0, -1] = summary_by_label["Qwen3 Avg (matched primary)"]["spearman_p_asymptotic"]
    q_matrix[1, -1] = summary_by_label["Qwen2.5 Avg (matched primary)"]["spearman_p_asymptotic"]
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
    matched_sizes = [row["matched_n"] for row in task_rows]
    matched_text = str(min(matched_sizes)) if min(matched_sizes) == max(matched_sizes) else f"{min(matched_sizes)}–{max(matched_sizes)}"
    axis.set_title(f"Task-level Spearman correlation on matched cohorts (n={matched_text}; * p < 0.05)")
    axis.grid(False)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Spearman rho")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_task_correlation_heatmap.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(rows: Sequence[dict]) -> None:
    # n<5 bootstrap intervals are essentially uninformative and would compress
    # every other cohort; those exploratory results remain in the CSV/report.
    plotted = [
        row
        for row in rows
        if is_finite(row["spearman_ci_low"]) and row["n"] >= 5
    ]
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
    axis.set_xlim(max(-1.02, float(np.nanmin(low)) - 0.05), 1.02)
    axis.set_xlabel("Spearman rho with 95% tokenizer-bootstrap interval")
    axis.set_title("Sensitivity of the probing–MLLM association across cohorts")
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
                color=PLOT_GROUP_COLORS[plot_group(row)],
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
    fig.suptitle("Out-of-sample calibration is weaker for unseen tokenizer families", fontsize=14)
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
    q3 = summary["Qwen3 Avg (matched primary)"]
    q25 = summary["Qwen2.5 Avg (matched primary)"]
    primary_no_ijepa = summary["Primary excluding I-JEPA"]
    primary_task_mean = summary["Primary using task-recomputed averages"]
    continuous = summary["Primary continuous only"]
    discrete = summary["Primary discrete only"]
    clip_like = summary["CLIP-like benchmark families"]
    no_dino = summary["Primary excluding DINOv3"]
    no_rae = summary["Primary excluding RAE-v2"]
    history = summary["Two-Qwen Avg (full 10-epoch histories)"]
    final_only = summary["Two-Qwen Avg (final-only probing)"]

    core = [record for record in records if record["in_primary"]]
    epoch_records = [record for record in records if record["probe_history_status"] == "full_10_epoch"]
    epoch_core = [record for record in epoch_records if record["in_epoch_combined"]]
    no_history = [record for record in records if record["probe_history_status"] == "final_only"]
    no_probe = [record for record in records if record["probe_history_status"] == "missing_probe"]
    q3_missing = [record for record in records if not is_finite(record["qwen3_avg_reported"])]
    q25_missing = [record for record in records if not is_finite(record["qwen25_avg_reported"])]

    core_probe = np.asarray([record["probe_final"] for record in core])
    core_q3 = np.asarray([record["qwen3_avg_reported"] for record in core])
    core_q25 = np.asarray([record["qwen25_avg_reported"] for record in core])
    backbone_delta, backbone_delta_low, backbone_delta_high = bootstrap_spearman_difference(
        core_probe, core_q3, core_q25, "matched-backbone-rho"
    )

    probe_order = sorted(core, key=lambda record: record["probe_final"], reverse=True)
    mllm_order = sorted(core, key=lambda record: record["mllm_avg_fair"], reverse=True)
    topk_rows = []
    for k in [3, 5, 10]:
        overlap = len(
            {record["tokenizer"] for record in probe_order[:k]}
            & {record["tokenizer"] for record in mllm_order[:k]}
        )
        topk_rows.append((k, overlap))

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
            float(
                stats.spearmanr(
                    [record["probe_final"] for record in kept],
                    [record["mllm_avg_fair"] for record in kept],
                ).statistic
            )
        )

    task_by_backbone = defaultdict(list)
    for row in task_rows:
        task_by_backbone[row["backbone"]].append(row)
    task_summary = {}
    for backbone in [*BACKBONES, "Two-Qwen task mean"]:
        rows = task_by_backbone[backbone]
        task_summary[backbone] = (
            min(rows, key=lambda row: row["matched_spearman_rho"]),
            max(rows, key=lambda row: row["matched_spearman_rho"]),
        )
    q3_sig = sum(row["matched_p"] < 0.05 for row in task_by_backbone[BACKBONES[0]])
    q25_sig = sum(row["matched_p"] < 0.05 for row in task_by_backbone[BACKBONES[1]])
    q25_stronger = sum(
        q25_row["matched_spearman_rho"] > q3_row["matched_spearman_rho"]
        for q3_row, q25_row in zip(task_by_backbone[BACKBONES[0]], task_by_backbone[BACKBONES[1]])
    )
    bh_significant = sum(
        row["max_q_bh"] < 0.05
        for row in task_rows
        if row["backbone"] in BACKBONES
    )

    aggregation_leave_out = [row for row in aggregation_rows if row["analysis"].startswith("Leave out ")]
    aggregation_leave_min = min(row["spearman_rho"] for row in aggregation_leave_out)
    aggregation_leave_max = max(row["spearman_rho"] for row in aggregation_leave_out)

    epoch1 = epoch_metrics[0]
    epoch10 = epoch_metrics[-1]
    gains = np.asarray([record["epochs"][-1] - record["epochs"][0] for record in epoch_records])
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
                f"{row['spearman_vs_qwen3_avg']:.3f}",
                f"{row['spearman_vs_qwen25_avg']:.3f}",
                f"{row['spearman_vs_fair_mllm_avg']:.3f}",
                f"{row['rank_stability_vs_epoch10']:.3f}",
            ]
        )
        + " |"
        for row in epoch_metrics
    ]

    family_lines = []
    for label in ["Within SigLIP2", "Within MetaCLIP1", "Within MetaCLIP2"]:
        row = robust[label]
        family_lines.append(
            f"| {label.replace('Within ', '')} | {row['n']} | {row['spearman_rho']:.3f} | "
            f"[{row['spearman_ci_low']:.3f}, {row['spearman_ci_high']:.3f}] |"
        )
    adjusted = robust["Family-adjusted pooled rank association"]

    top_half = probe_order[: len(core) // 2]
    top_half_rho = float(
        stats.spearmanr(
            [record["probe_final"] for record in top_half],
            [record["mllm_avg_fair"] for record in top_half],
        ).statistic
    )
    high_probe = [record for record in core if record["probe_final"] >= 87]
    high_probe_result = stats.spearmanr(
        [record["probe_final"] for record in high_probe],
        [record["mllm_avg_fair"] for record in high_probe],
    )

    resolution_rows = [row for row in controlled_rows if row["comparison_type"] == "resolution"]
    delta_result = stats.spearmanr(
        [row["delta_probe"] for row in resolution_rows],
        [row["delta_mllm_avg"] for row in resolution_rows],
    )
    positive_both = sum(
        row["delta_probe"] > 0 and row["delta_mllm_avg"] > 0 for row in resolution_rows
    )

    metrics = {row["validation"]: row for row in prediction_metrics}
    loo = metrics["Leave-one-tokenizer-out"]
    lofo_all = metrics["Leave-one-family-out"]
    lofo_major = metrics["Leave-one-major-prefix-family-out"]
    epoch1_cv = metrics["Trajectory subset LOOCV: Epoch 1 only"]
    epoch10_cv = metrics["Trajectory subset LOOCV: Epoch 10 only"]
    gain_cv = metrics["Trajectory subset LOOCV: Epoch 10 + gain (E10-E1)"]

    names = lambda items: ", ".join(record["tokenizer"] for record in items) if items else "无"
    task_n_values = [row["matched_n"] for row in task_rows]
    task_n_text = (
        str(min(task_n_values))
        if min(task_n_values) == max(task_n_values)
        else f"{min(task_n_values)}–{max(task_n_values)}"
    )
    combined_mismatches = [
        row for row in audit["avg_mismatches"] if row["field"] == "main-table combined Avg"
    ]
    ijepa_mismatches = [
        row for row in audit["avg_mismatches"] if row["tokenizer"] == "I-JEPA"
    ]

    report = f"""# Linear probing 对 MLLM 表现的预测力（新版完整数据）

## 结论先行

新版公平主队列使用所有同时具备 ImageNet Epoch-10 probing、Qwen3 Avg 和 Qwen2.5 Avg 的 tokenizer。当前是 **n={primary['n']}，Spearman rho={primary['spearman_rho']:.3f}**（tokenizer bootstrap 95% CI [{primary['spearman_ci_low']:.3f}, {primary['spearman_ci_high']:.3f}]），Pearson r={primary['pearson_r']:.3f}，Kendall tau-b={primary['kendall_tau_b']:.3f}，置换检验 p={p_text(primary['spearman_p_permutation'])}。

这仍是很强的全局排序信号，但不再是旧版完整病例的 0.94。旧版对应的 CLIP-like 四类家族子集现在仍为 n={clip_like['n']}、rho={clip_like['spearman_rho']:.3f}；把新补齐的 discrete、I-JEPA、RAE-v2、DINOv3 纳入后，主结果变成 rho={primary['spearman_rho']:.3f}。所以变化主要说明 **跨新 tokenizer 家族的泛化比家族内排序难**，不是旧数据或计算突然失效。

- 同一 n={primary['n']} 队列上，Qwen3 rho={q3['spearman_rho']:.3f}，Qwen2.5 rho={q25['spearman_rho']:.3f}；后者高 {backbone_delta:.3f}，配对 bootstrap 95% CI [{backbone_delta_low:.3f}, {backbone_delta_high:.3f}]。
- continuous tokenizer（n={continuous['n']}）rho={continuous['spearman_rho']:.3f}；discrete（n={discrete['n']}）rho={discrete['spearman_rho']:.3f}，但后者仅 4 点，只能描述，不能据此判定“没有关系”。
- I-JEPA 的 Qwen3 Avg 存在行内不一致。排除它后 rho={primary_no_ijepa['spearman_rho']:.3f}；统一从 22 个任务重算双 Qwen Avg 后 rho={primary_task_mean['spearman_rho']:.3f}，主结论基本不变。
- DINOv3 与 RAE-v2 是合法但明显的跨家族残差点；仅作敏感性诊断，分别删除时 rho={no_dino['spearman_rho']:.3f} 与 {no_rae['spearman_rho']:.3f}，不作为主分析排除规则。

![Probing vs MLLM](figures/04_probe_vs_mllm_avg.png)

## 数据覆盖与公平口径

| 项目 | 可用数 | 说明 |
|---|---:|---|
| 原始 tokenizer | {len(records)} | 两个 CSV 名称集合与 Family 完全一致，按 tokenizer 名称关联 |
| Qwen3 / Qwen2.5 全任务与 Avg | {len(records)} / {len(records)} | 两套 11 个任务均已补齐；Qwen3 缺失 {len(q3_missing)}，Qwen2.5 缺失 {len(q25_missing)} |
| 最终 ImageNet probing | {sum(is_finite(record['probe_final']) for record in records)} | 仍有 {len(no_probe)} 个完全缺 probing |
| 主分析：probing × 两 Qwen 公平 Avg | {len(core)} | 两个 backbone Avg 等权平均；只排除无 probing 的点 |
| 完整 10-epoch 轨迹 | {len(epoch_records)} | 10 轮固定同一 tokenizer 队列 |
| 10-epoch × 两 Qwen Avg | {len(epoch_core)} | 当前所有轨迹点都有两套 MLLM 数据 |

- probing 完全缺失：{names(no_probe)}。
- 有最终 probing、但没有前 9 轮：{names(no_history)}。
- 旧版因 MLLM 缺失排除的 UniTok、VILA-U、TokLIP、I-JEPA、RAE-v2、DINOv3 已全部补齐，不再排除。
- 完整逐项口径见 [analysis_cohort.csv](data/analysis_cohort.csv)，排除原因见 [exclusions.csv](data/exclusions.csv)。

## 总体相关性、分组与稳健性

| 目标/子集 | n | Spearman rho | 95% CI | Pearson r | Kendall tau-b |
|---|---:|---:|---:|---:|---:|
| 两 Qwen 公平 Avg（主结果） | {primary['n']} | {primary['spearman_rho']:.3f} | [{primary['spearman_ci_low']:.3f}, {primary['spearman_ci_high']:.3f}] | {primary['pearson_r']:.3f} | {primary['kendall_tau_b']:.3f} |
| Qwen3 Avg（同队列） | {q3['n']} | {q3['spearman_rho']:.3f} | [{q3['spearman_ci_low']:.3f}, {q3['spearman_ci_high']:.3f}] | {q3['pearson_r']:.3f} | {q3['kendall_tau_b']:.3f} |
| Qwen2.5 Avg（同队列） | {q25['n']} | {q25['spearman_rho']:.3f} | [{q25['spearman_ci_low']:.3f}, {q25['spearman_ci_high']:.3f}] | {q25['pearson_r']:.3f} | {q25['kendall_tau_b']:.3f} |
| continuous only | {continuous['n']} | {continuous['spearman_rho']:.3f} | [{continuous['spearman_ci_low']:.3f}, {continuous['spearman_ci_high']:.3f}] | {continuous['pearson_r']:.3f} | {continuous['kendall_tau_b']:.3f} |
| discrete only（探索性） | {discrete['n']} | {discrete['spearman_rho']:.3f} | [{discrete['spearman_ci_low']:.3f}, {discrete['spearman_ci_high']:.3f}] | {discrete['pearson_r']:.3f} | {discrete['kendall_tau_b']:.3f} |
| CLIP-like benchmark families | {clip_like['n']} | {clip_like['spearman_rho']:.3f} | [{clip_like['spearman_ci_low']:.3f}, {clip_like['spearman_ci_high']:.3f}] | {clip_like['pearson_r']:.3f} | {clip_like['kendall_tau_b']:.3f} |
| 完整 10-epoch 来源 | {history['n']} | {history['spearman_rho']:.3f} | [{history['spearman_ci_low']:.3f}, {history['spearman_ci_high']:.3f}] | {history['pearson_r']:.3f} | {history['kendall_tau_b']:.3f} |
| 只有最终 probing 来源 | {final_only['n']} | {final_only['spearman_rho']:.3f} | [{final_only['spearman_ci_low']:.3f}, {final_only['spearman_ci_high']:.3f}] | {final_only['pearson_r']:.3f} | {final_only['kendall_tau_b']:.3f} |

完整轨迹与 final-only 两组的家族组成不同，因此两者差异不能归因为“协作者数据质量”。它更像一个来源与模型家族共同变化的敏感性分析。

从 22 个详细任务直接重算均值时，rho={aggregation['Raw mean across task cells']['spearman_rho']:.3f}；先逐任务 z-score 再平均为 {aggregation['Mean after per-task z-scoring']['spearman_rho']:.3f}；逐任务 rank 后平均为 {aggregation['Mean of within-task ranks']['spearman_rho']:.3f}；z-score 中位数为 {aggregation['Median after per-task z-scoring']['spearman_rho']:.3f}。每次留掉一个 backbone-task 单元，rho 范围 [{aggregation_leave_min:.3f}, {aggregation_leave_max:.3f}]，说明结果不是单一任务或量纲驱动。完整表见 [task_aggregation_robustness.csv](data/task_aggregation_robustness.csv)。

逐一删除 tokenizer 后，主 rho 范围为 [{min(leave_one_rhos):.3f}, {max(leave_one_rhos):.3f}]。上界来自删除强跨家族残差点，说明新版结论比旧版更依赖“是否要求跨家族泛化”，应保留这个限定。

![Robustness](figures/06_family_and_source_robustness.png)

## 具体 backbone 与任务

逐任务热图尽量使用全部 probing 可用点：通常 n={len(core)}；Qwen3 ScienceQA 及其双模型任务均值因 I-JEPA 疑似复制单元保守用 n={min(task_n_values)}。因此热图的任务级 n 范围为 {task_n_text}，每个格子的精确 n 在 [task_correlations.csv](data/task_correlations.csv)。

- Qwen3：从 {task_summary[BACKBONES[0]][0]['task']}={task_summary[BACKBONES[0]][0]['matched_spearman_rho']:.3f} 到 {task_summary[BACKBONES[0]][1]['task']}={task_summary[BACKBONES[0]][1]['matched_spearman_rho']:.3f}；{q3_sig}/11 个任务未经多重校正时 p<0.05。
- Qwen2.5：从 {task_summary[BACKBONES[1]][0]['task']}={task_summary[BACKBONES[1]][0]['matched_spearman_rho']:.3f} 到 {task_summary[BACKBONES[1]][1]['task']}={task_summary[BACKBONES[1]][1]['matched_spearman_rho']:.3f}；{q25_sig}/11 个任务未经多重校正时 p<0.05。
- 两 backbone 同名任务等权平均后，从 {task_summary['Two-Qwen task mean'][0]['task']}={task_summary['Two-Qwen task mean'][0]['matched_spearman_rho']:.3f} 到 {task_summary['Two-Qwen task mean'][1]['task']}={task_summary['Two-Qwen task mean'][1]['matched_spearman_rho']:.3f}。
- Qwen2.5 在 {q25_stronger}/11 个任务上的 rho 高于 Qwen3。对 22 个 backbone-task 检验统一做 Benjamini-Hochberg 校正后，{bh_significant}/22 仍显著。
- Flickr、COCO 等任务与 probing 关联最强，MMMU 最弱。合理解释是多学科推理还受语言与推理瓶颈限制；这只是相关性解释，不是因果证明。

![Task correlations](figures/05_task_correlation_heatmap.png)

## 10 个 epoch：逐轮 Spearman 表

以下每一轮都固定使用同一批 n={len(epoch_core)} tokenizer，同时计算 Qwen3、Qwen2.5 和两者公平平均，因此跨 epoch 和跨 backbone 可直接比较。

| Epoch | Qwen3 Avg rho | Qwen2.5 Avg rho | 两 Qwen公平 Avg rho | 与 Epoch-10 probing 排名 rho |
|---:|---:|---:|---:|---:|
{chr(10).join(epoch_spearman_lines)}

完整 p 值、每轮均值/中位准确率和样本数见 [epoch_metrics.csv](data/epoch_metrics.csv)。

![Epoch trajectories](figures/01_epoch_accuracy_trajectories.png)

下图按同一批 tokenizer 的 10 轮平均准确率排序，横轴不写 tokenizer 名；每条细线上的 10 个小点就是 Epoch 1–10。

![Epochs by tokenizer](figures/01b_epoch_by_tokenizer_overview.png)

![Epoch rank heatmap](figures/02_epoch_rank_heatmap.png)

E1→E10 的 probing 增益中位数为 {np.median(gains):.2f} pp；E1 与 E10 的全体排名 rho={epoch1['rank_stability_vs_epoch10']:.3f}。对 MLLM 公平 Avg，E1 rho={epoch1['spearman_vs_fair_mllm_avg']:.3f}，E10 rho={epoch10['spearman_vs_fair_mllm_avg']:.3f}，变化 {epoch_delta:+.3f}（配对 bootstrap 95% CI [{epoch_delta_low:.3f}, {epoch_delta_high:.3f}]）。区间跨 0，不能声称训练到第 10 轮会显著提高对 MLLM 的排序预测；当前数据中 E1 数值反而略高。

这不等于 E1 可以替代 E10：TokLIP 等慢收敛 discrete tokenizer 的绝对 probing 会继续大幅上升。轨迹 LOOCV 中 E1-only MAE={epoch1_cv['mae']:.2f}，E10-only MAE={epoch10_cv['mae']:.2f}，E10+gain MAE={gain_cv['mae']:.2f}。gain 与 MLLM Avg 的 rho={gain_vs_target:.3f}，与 E10-only 线性拟合残差的 rho={gain_vs_residual:.3f}；目前没有稳定的额外动力学预测收益。

![Epoch predictiveness](figures/03_epoch_predictiveness.png)

## 家族内部、局部选型和受控对比

| 家族 | n | 家族内 Spearman rho | 95% CI |
|---|---:|---:|---:|
{chr(10).join(family_lines)}

三大主家族内部仍分别很强。把全局秩在所有至少有 2 点的家族内中心化后，pooled family-adjusted rho={adjusted['spearman_rho']:.3f}（n={adjusted['n']}，家族内置换 p={p_text(adjusted['spearman_p_permutation'])}）。单点家族不能贡献家族内证据，这正是 RAE-v2、DINOv3 等新家族外推仍不确定的原因。

在 probing 排名前半 n={len(top_half)} 中 rho={top_half_rho:.3f}；probing≥87 的 n={len(high_probe)} 中 rho={float(high_probe_result.statistic):.3f}（p={p_text(float(high_probe_result.pvalue))}）。范围收窄后区分力下降，不能把全局 rho 直接理解成顶尖模型之间细微差值的精确预测。

固定架构、只比较分辨率时，{positive_both}/{len(resolution_rows)} 对的 probing 与 MLLM Avg 同方向增加；但增益幅度的 rho={float(delta_result.statistic):.3f}（p={p_text(float(delta_result.pvalue))}）。因此 probing 更适合判断整体方向，不适合把局部 probing 增益一比一换算成 MLLM 增益。

![Controlled deltas](figures/08_controlled_resolution_deltas.png)

## 真正的样本外预测

一元线性校准 MLLM Avg ~ probing 的结果：

| 验证方式 | n | MAE | RMSE | CV R² | 不用 probing 的 baseline MAE |
|---|---:|---:|---:|---:|---:|
| Leave-one-tokenizer-out | {loo['n']} | {loo['mae']:.2f} | {loo['rmse']:.2f} | {loo['r2_cv']:.3f} | {loo['baseline_mae']:.2f} |
| Leave-one-family-out（全部家族） | {lofo_all['n']} | {lofo_all['mae']:.2f} | {lofo_all['rmse']:.2f} | {lofo_all['r2_cv']:.3f} | {lofo_all['baseline_mae']:.2f} |
| Leave-one-major-prefix-family-out（三大主家族） | {lofo_major['n']} | {lofo_major['mae']:.2f} | {lofo_major['rmse']:.2f} | {lofo_major['r2_cv']:.3f} | {lofo_major['baseline_mae']:.2f} |

Leave-one-tokenizer-out 仍明显优于均值 baseline，但把整个家族留出后误差上升、R²下降。这和相关性部分一致：probing 是很好的表内排序 proxy，但遇到 RAE/DINO 一类新表征范式时，单变量校准不够。

- Top-3 重合 {topk_rows[0][1]}/3，Top-5 重合 {topk_rows[1][1]}/5，Top-10 重合 {topk_rows[2][1]}/10。
- 全部非并列 pair 中，{concordant}/{concordant + discordant} 方向一致，即 {pairwise_accuracy * 100:.1f}%；另有 {tied} 对并列。

![Prediction validation](figures/07_prediction_validation.png)

## 数据质量与边界

1. 新主表总 Avg 已修复：47/47 都等于两个 backbone Avg 的等权平均（仅有两位小数舍入），检测到异常行数 {len(combined_mismatches)}。
2. I-JEPA/Qwen3 仍不一致：11 个任务均值 35.09，而 reported Avg 36.56，差 1.47；VQAv2 与 ScienceQA 又同为 47.08。当前检测到相关异常 {len(ijepa_mismatches)} 条。主分析保留 reported Avg，并同时给出排除与任务重算敏感性；Qwen3 ScienceQA 单格分析保守排除 I-JEPA。
3. 33 条完整轨迹的 E10 与主表逐项一致；12 个 tokenizer 只有最终 probing，2 个完全缺 probing。脚本不补造前 9 轮。
4. 相关性不是因果证明。预训练数据、容量、分辨率和 tokenizer 范式会同时影响 probing 与 MLLM。
5. discrete 只有 4 点，I-JEPA/RAE-v2/DINOv3 等多个家族只有单点，跨家族结论仍需更多同类 tokenizer 验证。

## 文件与复现

- [analyze.py](analyze.py)：唯一分析脚本，按双行表头解析字段并按 tokenizer 名称关联。
- [correlation_summary.csv](data/correlation_summary.csv)：总体、连续/离散、旧 CLIP-like 子集和异常点敏感性。
- [task_correlations.csv](data/task_correlations.csv)：逐 backbone、逐任务结果与 BH 校正。
- [epoch_metrics.csv](data/epoch_metrics.csv)：10 个 epoch 各自与下游 MLLM 的 Spearman 和 p 值。
- [family_robustness.csv](data/family_robustness.csv)：家族内、留一家族和来源敏感性。
- [prediction_metrics.csv](data/prediction_metrics.csv)：LOOCV 与两种 leave-family-out 预测误差。
- [analysis_cohort.csv](data/analysis_cohort.csv) / [exclusions.csv](data/exclusions.csv)：逐 tokenizer 入选与排除口径。
- 其余结构化结果均在 data/，九张图均在 figures/。

复现命令：

    conda run -n TokBench python result/tokenizer_mllm_analysis/analyze.py

所有 bootstrap/permutation 使用固定种子 {SEED}。
"""
    (HERE / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    set_plot_style()
    records, _epochs, audit = load_records()
    by_name = {record["tokenizer"]: record for record in records}

    core = [record for record in records if record["in_primary"]]
    q3_max = [record for record in records if record["in_qwen3_max"]]
    q25_max = [record for record in records if record["in_qwen25_max"]]
    epoch_records = [
        record for record in records if record["probe_history_status"] == "full_10_epoch"
    ]
    epoch_core = [record for record in records if record["in_epoch_combined"]]
    final_only_core = [
        record for record in core if record["probe_history_status"] == "final_only"
    ]

    if not records or len(by_name) != len(records):
        raise ValueError("No records or duplicate tokenizer names after loading")
    if len(core) < 3 or len(epoch_records) < 3 or len(epoch_core) < 3:
        raise ValueError("Too few complete records for the requested correlations")
    if audit["epoch10_mismatches"]:
        raise ValueError(
            f"Epoch-10 values disagree with main table: {audit['epoch10_mismatches']}"
        )

    complete_task_records = [
        record
        for record in core
        if all(
            is_finite(record[key][task])
            for key in ["qwen3_tasks", "qwen25_tasks"]
            for task in TASKS
        )
    ]
    if len(complete_task_records) < 3:
        raise ValueError("Too few complete 22-task records for aggregation robustness")
    task_matrix = np.asarray(
        [
            [record["qwen3_tasks"][task] for task in TASKS]
            + [record["qwen25_tasks"][task] for task in TASKS]
            for record in complete_task_records
        ],
        dtype=float,
    )
    task_z = (task_matrix - np.mean(task_matrix, axis=0)) / np.std(task_matrix, axis=0)
    task_rank = np.column_stack(
        [
            stats.rankdata(task_matrix[:, index], method="average")
            for index in range(task_matrix.shape[1])
        ]
    )
    aggregate_values = {
        "_agg_raw_task_mean": np.mean(task_matrix, axis=1),
        "_agg_mean_task_z": np.mean(task_z, axis=1),
        "_agg_mean_task_rank": np.mean(task_rank, axis=1),
        "_agg_median_task_z": np.median(task_z, axis=1),
    }
    for key, values in aggregate_values.items():
        for record, value in zip(complete_task_records, values):
            record[key] = float(value)

    aggregation_rows = [
        correlation_row(
            "Raw mean across task cells",
            complete_task_records,
            "_agg_raw_task_mean",
            f"Same n={len(complete_task_records)} cohort; recomputed from {task_matrix.shape[1]} detailed cells.",
        ),
        correlation_row(
            "Mean after per-task z-scoring",
            complete_task_records,
            "_agg_mean_task_z",
            "Removes task scale and variance differences.",
        ),
        correlation_row(
            "Mean of within-task ranks",
            complete_task_records,
            "_agg_mean_task_rank",
            "Uses only each task's tokenizer ordering.",
        ),
        correlation_row(
            "Median after per-task z-scoring",
            complete_task_records,
            "_agg_median_task_z",
            "Robust aggregation across all task cells.",
        ),
    ]
    for column in range(task_matrix.shape[1]):
        backbone = "Qwen3" if column < len(TASKS) else "Qwen2.5"
        task = TASKS[column % len(TASKS)]
        values = np.mean(np.delete(task_matrix, column, axis=1), axis=1)
        key = f"_agg_leave_{column}"
        for record, value in zip(complete_task_records, values):
            record[key] = float(value)
        aggregation_rows.append(
            correlation_row(
                f"Leave out {backbone}/{task}",
                complete_task_records,
                key,
                f"One of {task_matrix.shape[1]} task cells removed.",
            )
        )

    clip_like_families = {"SigLIP2", "MetaCLIP1", "MetaCLIP2", "OpenAI CLIP"}
    summary_rows = [
        correlation_row(
            "Primary: fair two-backbone avg",
            core,
            "mllm_avg_fair",
            "Maximum fair coverage: probe plus both reported backbone averages.",
        ),
        correlation_row(
            "Qwen3 Avg (matched primary)",
            core,
            "qwen3_avg_reported",
            "Same tokenizer cohort as the primary result.",
        ),
        correlation_row(
            "Qwen2.5 Avg (matched primary)",
            core,
            "qwen25_avg_reported",
            "Same tokenizer cohort as the primary result.",
        ),
        correlation_row(
            "Primary excluding I-JEPA",
            [record for record in core if record["tokenizer"] != "I-JEPA"],
            "mllm_avg_fair",
            "Sensitivity to I-JEPA's reported Avg/task-mean inconsistency.",
        ),
        correlation_row(
            "Primary using task-recomputed averages",
            core,
            "mllm_avg_tasks_recomputed",
            "Both backbone averages recomputed from their 11 detailed task cells.",
        ),
        correlation_row(
            "Primary continuous only",
            [record for record in core if record["visual_token_type"] == "continuous"],
            "mllm_avg_fair",
        ),
        correlation_row(
            "Primary discrete only",
            [record for record in core if record["visual_token_type"] == "discrete"],
            "mllm_avg_fair",
            "Exploratory: only four discrete tokenizers.",
        ),
        correlation_row(
            "CLIP-like benchmark families",
            [record for record in core if record["model_family"] in clip_like_families],
            "mllm_avg_fair",
            "Like-for-like sensitivity matching the families represented in the former n=38 complete-case cohort.",
        ),
        correlation_row(
            "Primary excluding DINOv3",
            [record for record in core if record["tokenizer"] != "dinov3"],
            "mllm_avg_fair",
            "Diagnostic only; DINOv3 is valid and remains in the primary analysis.",
        ),
        correlation_row(
            "Primary excluding RAE-v2",
            [record for record in core if record["tokenizer"] != "raev2"],
            "mllm_avg_fair",
            "Diagnostic only; RAE-v2 is valid and remains in the primary analysis.",
        ),
        correlation_row(
            "Two-Qwen Avg (full 10-epoch histories)",
            epoch_core,
            "mllm_avg_fair",
            "All tokenizers with complete ten-epoch histories and both MLLM averages.",
        ),
        correlation_row(
            "Two-Qwen Avg (final-only probing)",
            final_only_core,
            "mllm_avg_fair",
            "Final probe exists in the main table but earlier epochs are unavailable.",
        ),
        correlation_row(
            "Qwen3 Avg (discrete only)",
            [record for record in q3_max if record["visual_token_type"] == "discrete"],
            "qwen3_avg_reported",
            "Exploratory.",
        ),
        correlation_row(
            "Qwen2.5 Avg (discrete only)",
            [record for record in q25_max if record["visual_token_type"] == "discrete"],
            "qwen25_avg_reported",
            "Exploratory.",
        ),
    ]
    summary_by_label = {row["analysis"]: row for row in summary_rows}

    robustness_rows = []
    for source_label, robustness_label in [
        ("Primary: fair two-backbone avg", "Primary overall"),
        ("Primary continuous only", "Continuous only"),
        ("Primary discrete only", "Discrete only"),
        ("CLIP-like benchmark families", "CLIP-like families"),
        ("Two-Qwen Avg (full 10-epoch histories)", "Full 10-epoch source"),
        ("Two-Qwen Avg (final-only probing)", "Final-only source"),
        ("Primary excluding DINOv3", "Exclude DINOv3 (diagnostic)"),
        ("Primary excluding RAE-v2", "Exclude RAE-v2 (diagnostic)"),
    ]:
        reused = dict(summary_by_label[source_label])
        reused["analysis"] = robustness_label
        robustness_rows.append(reused)
    for family in ["SigLIP2", "MetaCLIP1", "MetaCLIP2"]:
        robustness_rows.append(
            correlation_row(
                f"Within {family}",
                [record for record in core if record["model_family"] == family],
                "mllm_avg_fair",
            )
        )
    for family in ["SigLIP2", "MetaCLIP1", "MetaCLIP2"]:
        robustness_rows.append(
            correlation_row(
                f"Leave {family} out",
                [record for record in core if record["model_family"] != family],
                "mllm_avg_fair",
            )
        )
    robustness_rows.append(family_adjusted_spearman(core))

    ijepa_q3_anomaly = any(
        row["tokenizer"] == "I-JEPA" and row["field"] == "Qwen3 task mean"
        for row in audit["avg_mismatches"]
    )

    def valid_task_cell(record: dict, backbone: str, task: str) -> bool:
        return not (
            ijepa_q3_anomaly
            and record["tokenizer"] == "I-JEPA"
            and backbone == BACKBONES[0]
            and task == "ScienceQA"
        )

    task_rows = []
    for backbone, task_key in [
        (BACKBONES[0], "qwen3_tasks"),
        (BACKBONES[1], "qwen25_tasks"),
    ]:
        for task in TASKS:
            maximum = [
                record
                for record in records
                if is_finite(record["probe_final"])
                and is_finite(record[task_key][task])
                and valid_task_cell(record, backbone, task)
            ]
            matched = [
                record
                for record in core
                if is_finite(record[task_key][task])
                and valid_task_cell(record, backbone, task)
            ]
            max_x = [record["probe_final"] for record in maximum]
            max_y = [record[task_key][task] for record in maximum]
            matched_x = [record["probe_final"] for record in matched]
            matched_y = [record[task_key][task] for record in matched]
            max_result = stats.spearmanr(max_x, max_y)
            matched_result = stats.spearmanr(matched_x, matched_y)
            ci_low, ci_high = bootstrap_corr_ci(
                max_x, max_y, "spearman", f"task-{backbone}-{task}"
            )
            task_rows.append(
                {
                    "backbone": backbone,
                    "task": task,
                    "max_n": len(maximum),
                    "max_spearman_rho": float(max_result.statistic),
                    "max_ci_low": ci_low,
                    "max_ci_high": ci_high,
                    "max_p": float(max_result.pvalue),
                    "max_q_bh": float("nan"),
                    "matched_n": len(matched),
                    "matched_spearman_rho": float(matched_result.statistic),
                    "matched_p": float(matched_result.pvalue),
                    "note": (
                        "I-JEPA excluded because its Qwen3 task mean conflicts with the reported Avg and ScienceQA duplicates VQAv2"
                        if backbone == BACKBONES[0]
                        and task == "ScienceQA"
                        and ijepa_q3_anomaly
                        else ""
                    ),
                }
            )
    qvalues = benjamini_hochberg([row["max_p"] for row in task_rows])
    for row, qvalue in zip(task_rows, qvalues):
        row["max_q_bh"] = qvalue

    for task in TASKS:
        matched = [
            record
            for record in core
            if is_finite(record["qwen3_tasks"][task])
            and is_finite(record["qwen25_tasks"][task])
            and valid_task_cell(record, BACKBONES[0], task)
        ]
        x = [record["probe_final"] for record in matched]
        y = [
            float(
                np.mean(
                    [record["qwen3_tasks"][task], record["qwen25_tasks"][task]]
                )
            )
            for record in matched
        ]
        result = stats.spearmanr(x, y)
        ci_low, ci_high = bootstrap_corr_ci(
            x, y, "spearman", f"task-combined-{task}"
        )
        task_rows.append(
            {
                "backbone": "Two-Qwen task mean",
                "task": task,
                "max_n": len(matched),
                "max_spearman_rho": float(result.statistic),
                "max_ci_low": ci_low,
                "max_ci_high": ci_high,
                "max_p": float(result.pvalue),
                "max_q_bh": float("nan"),
                "matched_n": len(matched),
                "matched_spearman_rho": float(result.statistic),
                "matched_p": float(result.pvalue),
                "note": (
                    "I-JEPA excluded because the Qwen3 ScienceQA cell is under QC"
                    if task == "ScienceQA" and ijepa_q3_anomaly
                    else "Descriptive aggregate; excluded from backbone-specific BH tests."
                ),
            }
        )

    accuracy_matrix = np.asarray(
        [record["epochs"] for record in epoch_records], dtype=float
    )
    final_accuracy = accuracy_matrix[:, -1]
    epoch_metrics = []
    for index in range(10):
        matched = [
            record
            for record in epoch_core
            if is_finite(record["epochs"][index])
            and is_finite(record["qwen3_avg_reported"])
            and is_finite(record["qwen25_avg_reported"])
        ]
        current = np.asarray([record["epochs"][index] for record in matched])
        q3_values = np.asarray([record["qwen3_avg_reported"] for record in matched])
        q25_values = np.asarray([record["qwen25_avg_reported"] for record in matched])
        fair_values = np.asarray([record["mllm_avg_fair"] for record in matched])
        q3_result = stats.spearmanr(current, q3_values)
        q25_result = stats.spearmanr(current, q25_values)
        fair_result = stats.spearmanr(current, fair_values)
        all_current = accuracy_matrix[:, index]
        epoch_metrics.append(
            {
                "epoch": index + 1,
                "n_all_histories": len(epoch_records),
                "mean_accuracy": float(np.mean(all_current)),
                "median_accuracy": float(np.median(all_current)),
                "rank_stability_vs_epoch10": float(
                    stats.spearmanr(all_current, final_accuracy).statistic
                ),
                "n_mllm_matched": len(matched),
                "spearman_vs_qwen3_avg": float(q3_result.statistic),
                "p_vs_qwen3_avg": float(q3_result.pvalue),
                "spearman_vs_qwen25_avg": float(q25_result.statistic),
                "p_vs_qwen25_avg": float(q25_result.pvalue),
                "spearman_vs_fair_mllm_avg": float(fair_result.statistic),
                "p_vs_fair_mllm_avg": float(fair_result.pvalue),
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
        if not all(
            is_finite(value)
            for value in [
                baseline["probe_final"],
                changed["probe_final"],
                baseline["mllm_avg_fair"],
                changed["mllm_avg_fair"],
            ]
        ):
            continue
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
                "delta_mllm_avg": changed["mllm_avg_fair"]
                - baseline["mllm_avg_fair"],
            }
        )

    cohort_rows = []
    for record in records:
        if not is_finite(record["probe_final"]):
            primary_reason = "missing ImageNet probe"
        elif not is_finite(record["qwen3_avg_reported"]):
            primary_reason = "missing Qwen3"
        elif not is_finite(record["qwen25_avg_reported"]):
            primary_reason = "missing Qwen2.5"
        else:
            primary_reason = "included"
        cohort_rows.append(
            {
                "tokenizer": record["tokenizer"],
                "visual_token_type": record["visual_token_type"],
                "model_family": record["model_family"],
                "probe_final": record["probe_final"],
                "probe_history_status": record["probe_history_status"],
                **{
                    f"epoch_{index + 1}": value
                    for index, value in enumerate(record["epochs"])
                },
                "qwen3_avg_reported": record["qwen3_avg_reported"],
                "qwen3_avg_recomputed_from_tasks": record["qwen3_avg_recomputed"],
                "qwen25_avg_reported": record["qwen25_avg_reported"],
                "qwen25_avg_recomputed_from_tasks": record["qwen25_avg_recomputed"],
                "mllm_avg_reported_raw": record["mllm_avg_reported_raw"],
                "mllm_avg_fair": record["mllm_avg_fair"],
                "mllm_avg_tasks_recomputed": record["mllm_avg_tasks_recomputed"],
                "mllm_status": record["mllm_status"],
                "in_primary": record["in_primary"],
                "in_qwen3_max": record["in_qwen3_max"],
                "in_qwen25_max": record["in_qwen25_max"],
                "in_epoch_combined": record["in_epoch_combined"],
                "primary_status_or_exclusion": primary_reason,
            }
        )

    exclusions = []
    for record in records:
        if not record["in_primary"]:
            if not is_finite(record["probe_final"]):
                reason = "ImageNet linear probe missing"
            elif not is_finite(record["qwen3_avg_reported"]):
                reason = "Qwen3 MLLM result missing"
            else:
                reason = "Qwen2.5 MLLM result missing"
            exclusions.append(
                {
                    "analysis": "Primary two-Qwen average",
                    "analysis_n": len(core),
                    "tokenizer": record["tokenizer"],
                    "reason": reason,
                }
            )
        if record["probe_history_status"] != "full_10_epoch":
            reason = (
                "Final probing exists, but Epochs 1-9 are absent from the epoch Markdown"
                if record["probe_history_status"] == "final_only"
                else "ImageNet probing is entirely missing"
            )
            exclusions.append(
                {
                    "analysis": "Ten-epoch trajectory",
                    "analysis_n": len(epoch_records),
                    "tokenizer": record["tokenizer"],
                    "reason": reason,
                }
            )
    if ijepa_q3_anomaly:
        exclusions.append(
            {
                "analysis": "Qwen3 ScienceQA task correlation",
                "analysis_n": next(
                    row["matched_n"]
                    for row in task_rows
                    if row["backbone"] == BACKBONES[0]
                    and row["task"] == "ScienceQA"
                ),
                "tokenizer": "I-JEPA",
                "reason": "Conservative cell-level QC exclusion: Qwen3 task mean conflicts with reported Avg and ScienceQA duplicates VQAv2",
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
        "mllm_avg_tasks_recomputed",
        "mllm_status",
        "in_primary",
        "in_qwen3_max",
        "in_qwen25_max",
        "in_epoch_combined",
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
        "max_q_bh",
        "matched_n",
        "matched_spearman_rho",
        "matched_p",
        "note",
    ]

    write_csv(DATA_DIR / "analysis_cohort.csv", cohort_fields, cohort_rows)
    write_csv(
        DATA_DIR / "exclusions.csv",
        ["analysis", "analysis_n", "tokenizer", "reason"],
        exclusions,
    )
    write_csv(DATA_DIR / "correlation_summary.csv", corr_fields, summary_rows)
    write_csv(
        DATA_DIR / "task_aggregation_robustness.csv",
        corr_fields,
        aggregation_rows,
    )
    write_csv(DATA_DIR / "family_robustness.csv", corr_fields, robustness_rows)
    write_csv(DATA_DIR / "task_correlations.csv", task_fields, task_rows)
    write_csv(
        DATA_DIR / "epoch_metrics.csv",
        list(epoch_metrics[0]),
        epoch_metrics,
    )
    write_csv(
        DATA_DIR / "prediction_diagnostics.csv",
        list(diagnostics[0]),
        diagnostics,
    )
    write_csv(
        DATA_DIR / "prediction_metrics.csv",
        list(prediction_metrics[0]),
        prediction_metrics,
    )
    write_csv(
        DATA_DIR / "controlled_comparisons.csv",
        list(controlled_rows[0]),
        controlled_rows,
    )

    plot_epoch_trajectories(epoch_records)
    plot_epochs_by_tokenizer(epoch_records)
    plot_epoch_rank_heatmap(epoch_records)
    plot_epoch_predictiveness(epoch_metrics, epoch_records)
    plot_probe_vs_mllm(core, summary_by_label)
    plot_task_heatmap(task_rows, summary_by_label)
    plot_robustness(robustness_rows)
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
    print(
        f"Primary n={len(core)}, "
        f"Spearman rho={summary_by_label['Primary: fair two-backbone avg']['spearman_rho']:.6f}"
    )


if __name__ == "__main__":
    main()
