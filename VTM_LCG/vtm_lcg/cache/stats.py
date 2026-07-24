from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from .io import ShardDescriptor, validate_shard


def _iter_values(
    cache_dir: Path,
    descriptors: Iterable[ShardDescriptor],
) -> Iterable[torch.Tensor]:
    for descriptor in descriptors:
        values, _image_ids = validate_shard(
            cache_dir / "shards" / descriptor.filename,
            descriptor,
            verify_checksum=True,
        )
        yield values


def compute_cache_stats(
    cache_dir: Path,
    descriptors: list[ShardDescriptor],
    *,
    epsilon: float,
) -> dict[str, Any]:
    if not descriptors:
        raise ValueError("Cannot compute statistics for an empty cache")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    started_at = time.perf_counter()
    channel_sum: torch.Tensor | None = None
    channel_square_sum: torch.Tensor | None = None
    token_variance_sum = 0.0
    image_count = 0
    token_observation_count = 0
    nan_count = 0
    inf_count = 0

    for values in _iter_values(cache_dir, descriptors):
        nan_count += int(torch.isnan(values).sum().item())
        inf_count += int(torch.isinf(values).sum().item())
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Cache contains non-finite values")
        values64 = values.to(torch.float64)
        current_sum = values64.sum(dim=(0, 1))
        current_square_sum = values64.square().sum(dim=(0, 1))
        channel_sum = current_sum if channel_sum is None else channel_sum + current_sum
        channel_square_sum = (
            current_square_sum
            if channel_square_sum is None
            else channel_square_sum + current_square_sum
        )
        token_variance_sum += float(
            values64.var(dim=1, correction=0).mean(dim=1).sum().item()
        )
        image_count += int(values.shape[0])
        token_observation_count += int(values.shape[0] * values.shape[1])

    assert channel_sum is not None and channel_square_sum is not None
    mean = channel_sum / token_observation_count
    variance = channel_square_sum / token_observation_count - mean.square()
    variance = variance.clamp_min(0.0)
    std = torch.sqrt(variance + epsilon)

    normalized_sum = torch.zeros_like(mean)
    normalized_square_sum = torch.zeros_like(mean)
    for values in _iter_values(cache_dir, descriptors):
        normalized = (values.to(torch.float64) - mean) / std
        normalized_sum += normalized.sum(dim=(0, 1))
        normalized_square_sum += normalized.square().sum(dim=(0, 1))
    normalized_mean = normalized_sum / token_observation_count
    normalized_variance = (
        normalized_square_sum / token_observation_count - normalized_mean.square()
    ).clamp_min(0.0)
    normalized_std = torch.sqrt(normalized_variance)

    return {
        "schema_version": 1,
        "image_count": image_count,
        "token_observation_count": token_observation_count,
        "hidden_dim": int(mean.numel()),
        "epsilon": float(epsilon),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "channel_mean": [float(value) for value in mean.tolist()],
        "channel_std": [float(value) for value in std.tolist()],
        "channel_mean_min": float(mean.min().item()),
        "channel_mean_max": float(mean.max().item()),
        "channel_std_min": float(std.min().item()),
        "channel_std_max": float(std.max().item()),
        "mean_token_variance": token_variance_sum / image_count,
        "normalized_max_abs_channel_mean": float(normalized_mean.abs().max().item()),
        "normalized_max_abs_channel_std_error": float(
            (normalized_std - 1.0).abs().max().item()
        ),
        "elapsed_seconds": time.perf_counter() - started_at,
        "finite": nan_count == 0 and inf_count == 0,
        "acceptance": {
            "finite": nan_count == 0 and inf_count == 0,
            "all_channel_std_above_1e-6": bool(std.min().item() > 1.0e-6),
            "token_variance_positive": token_variance_sum > 0.0,
            "normalized_mean_within_5e-3": bool(
                normalized_mean.abs().max().item() < 5.0e-3
            ),
            "normalized_std_within_5e-3": bool(
                (normalized_std - 1.0).abs().max().item() < 5.0e-3
            ),
        },
    }


def validate_stats_acceptance(stats: dict[str, Any]) -> None:
    failures = [
        name for name, passed in stats["acceptance"].items() if not bool(passed)
    ]
    if failures:
        raise RuntimeError(f"Phase 0 statistics failed acceptance checks: {failures}")

