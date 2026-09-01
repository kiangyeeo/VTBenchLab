"""Small CSV/statistics helpers for LAR evaluation scripts."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: str | float | None) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def rho(x: list[float], y: list[float]) -> tuple[float, float]:
    if len(x) < 2:
        return math.nan, math.nan
    result = spearmanr(np.asarray(x), np.asarray(y))
    return float(result.statistic), float(result.pvalue)


def write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")

