"""Single source of truth for metric selection directions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

try:
    from .data import WORKSPACE
except ImportError:  # Direct execution
    from data import WORKSPACE


DEFAULT_METRICS_CONFIG = WORKSPACE / "lar" / "configs" / "metrics.yaml"


def load_metric_config(path: Path = DEFAULT_METRICS_CONFIG) -> dict[str, dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise RuntimeError(f"Expected a non-empty metrics mapping in {path}")
    output: dict[str, dict[str, object]] = {}
    for name, settings in metrics.items():
        if not isinstance(settings, dict) or not isinstance(
            settings.get("higher_is_better"), bool
        ):
            raise RuntimeError(
                f"Metric {name!r} in {path} must declare boolean higher_is_better"
            )
        output[str(name)] = dict(settings)
    return output


def higher_is_better(name: str, config: dict[str, dict[str, object]]) -> bool:
    if name not in config:
        raise KeyError(f"Metric {name!r} has no entry in metrics configuration")
    return bool(config[name]["higher_is_better"])


def oriented(
    name: str,
    values: np.ndarray,
    config: dict[str, dict[str, object]],
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array if higher_is_better(name, config) else -array
