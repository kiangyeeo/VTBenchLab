from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor


def make_deterministic_block_mask(
    record_indices: Sequence[int] | Tensor,
    *,
    grid_shape: tuple[int, int],
    block_shape: tuple[int, int],
    mask_ratio: float,
    seed: int,
    epoch: int,
    device: torch.device | str,
) -> Tensor:
    """Select complete non-overlapping blocks with deterministic per-record seeds."""
    rows, columns = grid_shape
    block_rows, block_columns = block_shape
    if rows % block_rows != 0 or columns % block_columns != 0:
        raise ValueError("block shape must tile the token grid exactly")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between zero and one")
    coarse_rows = rows // block_rows
    coarse_columns = columns // block_columns
    block_count = coarse_rows * coarse_columns
    masked_blocks = int(round(block_count * mask_ratio))
    if masked_blocks <= 0 or masked_blocks >= block_count:
        raise ValueError("mask_ratio produces an invalid number of masked blocks")
    actual_ratio = masked_blocks / block_count
    if abs(actual_ratio - mask_ratio) > 1.0e-9:
        raise ValueError(
            f"mask_ratio={mask_ratio} is not exactly representable by "
            f"{block_count} blocks"
        )

    if isinstance(record_indices, Tensor):
        indices = [int(value) for value in record_indices.tolist()]
    else:
        indices = [int(value) for value in record_indices]
    mask = torch.zeros(len(indices), rows, columns, dtype=torch.bool)
    for batch_row, record_index in enumerate(indices):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(seed) + int(epoch) * 1_000_003 + record_index * 97
        )
        selected = torch.randperm(
            block_count,
            generator=generator,
        )[:masked_blocks]
        for block_index in selected.tolist():
            coarse_row = block_index // coarse_columns
            coarse_column = block_index % coarse_columns
            row_start = coarse_row * block_rows
            column_start = coarse_column * block_columns
            mask[
                batch_row,
                row_start : row_start + block_rows,
                column_start : column_start + block_columns,
            ] = True
    return mask.reshape(len(indices), rows * columns).to(device)


def residualize_cross_view(
    source: Tensor,
    target: Tensor,
    masked_positions: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Remove the source visible-token mean without consulting masked targets."""
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError("source and target must have matching shape [B,N,D]")
    if tuple(masked_positions.shape) != tuple(source.shape[:2]):
        raise ValueError("masked_positions must have shape [B,N]")
    visible = ~masked_positions
    visible_count = visible.sum(dim=1, keepdim=True)
    if bool((visible_count == 0).any()):
        raise ValueError("each example must retain at least one visible token")
    source_center = (
        (source * visible.unsqueeze(-1)).sum(dim=1, keepdim=True)
        / visible_count.unsqueeze(-1)
    )
    return source - source_center, target - source_center, source_center


def cross_view_loss_sums(
    prediction: Tensor,
    target: Tensor,
    target_residual: Tensor,
    masked_positions: Tensor,
) -> dict[str, float | int]:
    if prediction.shape != target.shape or prediction.shape != target_residual.shape:
        raise ValueError("prediction, target, and target_residual shapes must match")
    selected_prediction = prediction[masked_positions].float()
    selected_target = target[masked_positions].float()
    selected_residual = target_residual[masked_positions].float()
    return {
        "total_sum": float(selected_target.square().sum().item()),
        "residual_null_sum": float(selected_residual.square().sum().item()),
        "residual_prediction_sum": float(
            (selected_prediction - selected_residual).square().sum().item()
        ),
        "element_count": selected_target.numel(),
    }


def compute_cvrvtm_scores(
    losses: dict[str, float],
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    required = {"L_total", "L_residual_null", "L_residual_prediction"}
    missing = required - set(losses)
    if missing:
        raise KeyError(f"Missing required losses: {sorted(missing)}")
    total = float(losses["L_total"])
    residual_null = float(losses["L_residual_null"])
    residual_prediction = float(losses["L_residual_prediction"])
    if total <= 0 or residual_null < 0 or residual_prediction < 0:
        raise ValueError(f"Invalid CV-RVTM losses: {losses}")
    residual_gain = residual_null - residual_prediction
    result: dict[str, Any] = {
        "CVRVTM": residual_gain / max(total, epsilon),
        "residual_energy_ratio": residual_null / max(total, epsilon),
        "residual_predictability": (
            residual_gain / max(residual_null, epsilon)
        ),
    }
    result["CVRVTM_positive"] = result["CVRVTM"] > 0
    return result
