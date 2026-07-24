from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor


def make_deterministic_mask(
    record_indices: Sequence[int] | Tensor,
    *,
    token_count: int,
    mask_ratio: float,
    seed: int,
    epoch: int,
    device: torch.device | str,
) -> Tensor:
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between zero and one")
    masked_count = int(round(token_count * mask_ratio))
    if masked_count <= 0 or masked_count >= token_count:
        raise ValueError("mask_ratio produces an invalid number of masked tokens")
    if isinstance(record_indices, Tensor):
        indices = [int(value) for value in record_indices.tolist()]
    else:
        indices = [int(value) for value in record_indices]
    mask = torch.zeros(len(indices), token_count, dtype=torch.bool)
    for row, record_index in enumerate(indices):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(seed) + int(epoch) * 1_000_003 + record_index * 97
        )
        positions = torch.randperm(token_count, generator=generator)[:masked_count]
        mask[row, positions] = True
    return mask.to(device)


def select_caption_ids(
    records: list[dict[str, Any]],
    record_indices: Sequence[int] | Tensor,
    *,
    seed: int,
    epoch: int,
    fixed_first: bool,
) -> list[int]:
    if isinstance(record_indices, Tensor):
        indices = [int(value) for value in record_indices.tolist()]
    else:
        indices = [int(value) for value in record_indices]
    selected: list[int] = []
    for record_index in indices:
        caption_ids = records[record_index]["caption_ids"]
        if not caption_ids:
            raise ValueError(f"Record {record_index} has no captions")
        if fixed_first:
            caption_position = 0
        else:
            caption_position = (
                int(seed) + int(epoch) * 104_729 + record_index * 17
            ) % len(caption_ids)
        selected.append(int(caption_ids[caption_position]))
    return selected


def select_caption_texts(
    records: list[dict[str, Any]],
    record_indices: Sequence[int] | Tensor,
    *,
    seed: int,
    epoch: int,
    fixed_first: bool,
) -> list[str]:
    if isinstance(record_indices, Tensor):
        indices = [int(value) for value in record_indices.tolist()]
    else:
        indices = [int(value) for value in record_indices]
    selected: list[str] = []
    for record_index in indices:
        captions = records[record_index]["captions"]
        if not captions:
            raise ValueError(f"Record {record_index} has no captions")
        if fixed_first:
            caption_position = 0
        else:
            caption_position = (
                int(seed) + int(epoch) * 104_729 + record_index * 17
            ) % len(captions)
        selected.append(str(captions[caption_position]))
    return selected


def caption_keep_mask(
    record_indices: Sequence[int] | Tensor,
    *,
    dropout: float,
    seed: int,
    epoch: int,
    device: torch.device | str,
) -> Tensor:
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("caption dropout must be in [0,1]")
    if isinstance(record_indices, Tensor):
        indices = [int(value) for value in record_indices.tolist()]
    else:
        indices = [int(value) for value in record_indices]
    keep = torch.empty(len(indices), dtype=torch.bool)
    for row, record_index in enumerate(indices):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(seed) + int(epoch) * 1_000_033 + record_index * 193
        )
        keep[row] = bool(torch.rand((), generator=generator).item() >= dropout)
    return keep.to(device)


def spatially_shuffle_visible_tokens(
    visual_values: Tensor,
    masked_positions: Tensor,
    record_indices: Sequence[int] | Tensor,
    *,
    seed: int,
) -> Tensor:
    if visual_values.ndim != 3:
        raise ValueError("visual_values must have shape [B,N,D]")
    if tuple(masked_positions.shape) != tuple(visual_values.shape[:2]):
        raise ValueError("masked_positions must have shape [B,N]")
    if isinstance(record_indices, Tensor):
        indices = [int(value) for value in record_indices.tolist()]
    else:
        indices = [int(value) for value in record_indices]
    shuffled = visual_values.clone()
    for row, record_index in enumerate(indices):
        visible = torch.nonzero(~masked_positions[row], as_tuple=False).flatten()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + record_index * 389)
        permutation = torch.randperm(visible.numel(), generator=generator).to(
            visible.device
        )
        shuffled[row, visible] = visual_values[row, visible[permutation]]
    return shuffled


def shuffled_caption_id_map(
    records: list[dict[str, Any]],
    evaluation_indices: Sequence[int],
) -> dict[int, int]:
    indices = [int(index) for index in evaluation_indices]
    if len(indices) < 2:
        raise ValueError("Shuffled-caption evaluation requires at least two records")
    result: dict[int, int] = {}
    for position, record_index in enumerate(indices):
        next_index = indices[(position + 1) % len(indices)]
        result[record_index] = int(records[next_index]["caption_ids"][0])
    return result
