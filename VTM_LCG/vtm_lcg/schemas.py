from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class TokenBatch:
    """A dense batch of spatial visual tokens on one representation surface."""

    values: Tensor
    mask: Tensor
    coords: Tensor
    special_mask: Tensor
    grid_shape: tuple[int, int]
    surface: str
    input_size: tuple[int, int]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError(f"values must have shape [B,N,D], got {tuple(self.values.shape)}")
        batch_size, token_count, hidden_dim = self.values.shape
        if batch_size < 1 or token_count < 1 or hidden_dim < 1:
            raise ValueError(f"values dimensions must be positive, got {tuple(self.values.shape)}")

        expected_bn = (batch_size, token_count)
        if tuple(self.mask.shape) != expected_bn or self.mask.dtype is not torch.bool:
            raise ValueError(
                f"mask must be bool with shape {expected_bn}, got "
                f"{tuple(self.mask.shape)} {self.mask.dtype}"
            )
        if tuple(self.special_mask.shape) != expected_bn or self.special_mask.dtype is not torch.bool:
            raise ValueError(
                f"special_mask must be bool with shape {expected_bn}, got "
                f"{tuple(self.special_mask.shape)} {self.special_mask.dtype}"
            )
        if tuple(self.coords.shape) != (batch_size, token_count, 2):
            raise ValueError(
                "coords must have shape "
                f"{(batch_size, token_count, 2)}, got {tuple(self.coords.shape)}"
            )
        if self.coords.dtype not in (torch.int16, torch.int32, torch.int64):
            raise ValueError(f"coords must use an integer dtype, got {self.coords.dtype}")

        rows, columns = self.grid_shape
        if rows <= 0 or columns <= 0 or rows * columns != token_count:
            raise ValueError(
                f"grid_shape {self.grid_shape} does not match token_count={token_count}"
            )
        if any(dimension <= 0 for dimension in self.input_size):
            raise ValueError(f"input_size must be positive, got {self.input_size}")
        if not self.surface.strip():
            raise ValueError("surface must be non-empty")
        if not bool(self.mask.all()):
            raise ValueError("Phase 0 TokenBatch must be dense: every mask entry must be true")
        if bool(self.special_mask.any()):
            raise ValueError("Phase 0 adapters must remove all special tokens")

        expected_coords = row_major_coords(self.grid_shape, device=self.coords.device)
        expected_coords = expected_coords.unsqueeze(0).expand(batch_size, -1, -1)
        if not torch.equal(self.coords.to(torch.int64), expected_coords):
            raise ValueError("coords must be a row-major grid matching grid_shape")

        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.values.shape[1])

    @property
    def hidden_dim(self) -> int:
        return int(self.values.shape[2])


def row_major_coords(
    grid_shape: tuple[int, int],
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    rows, columns = grid_shape
    if rows <= 0 or columns <= 0:
        raise ValueError(f"grid_shape must be positive, got {grid_shape}")
    row_ids = torch.arange(rows, dtype=torch.int64, device=device)
    column_ids = torch.arange(columns, dtype=torch.int64, device=device)
    row_grid, column_grid = torch.meshgrid(row_ids, column_ids, indexing="ij")
    return torch.stack((row_grid, column_grid), dim=-1).reshape(rows * columns, 2)

