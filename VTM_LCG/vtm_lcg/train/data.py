from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from vtm_lcg.cache.io import ShardDescriptor, validate_shard


@dataclass
class Phase0VisualCache:
    tokenizer_id: str
    values: torch.Tensor
    records: list[dict[str, Any]]
    dataset_fingerprint: str
    cache_dir: Path


class IndexedVisualDataset(Dataset):
    def __init__(self, values: torch.Tensor, indices: Sequence[int]) -> None:
        self.values = values
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        record_index = self.indices[item]
        return self.values[record_index], record_index


def load_phase0_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("all_acceptance_checks_passed"):
        raise RuntimeError(f"Phase 0 summary did not pass acceptance checks: {path}")
    return summary


def load_phase0_visual_cache(
    summary: dict[str, Any],
    tokenizer_id: str,
) -> Phase0VisualCache:
    result_by_id = {
        item["tokenizer_id"]: item for item in summary["tokenizers"]
    }
    if tokenizer_id not in result_by_id:
        raise KeyError(f"Tokenizer {tokenizer_id!r} is absent from the Phase 0 summary")
    cache_dir = Path(result_by_id[tokenizer_id]["cache_dir"])
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    records_payload = json.loads((cache_dir / "records.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError(f"Phase 0 cache is incomplete: {cache_dir}")
    values: list[torch.Tensor] = []
    image_ids: list[int] = []
    for raw_descriptor in manifest["shards"]:
        descriptor = ShardDescriptor.from_dict(raw_descriptor)
        shard_values, shard_ids = validate_shard(
            cache_dir / "shards" / descriptor.filename,
            descriptor,
            verify_checksum=True,
        )
        values.append(shard_values)
        image_ids.extend(int(value) for value in shard_ids.tolist())
    combined = torch.cat(values, dim=0)
    records = records_payload["records"]
    if combined.shape[0] != len(records):
        raise RuntimeError("Phase 0 tensor count does not match records.json")
    if image_ids != [int(record["image_id"]) for record in records]:
        raise RuntimeError("Phase 0 shard order does not match records.json")
    return Phase0VisualCache(
        tokenizer_id=tokenizer_id,
        values=combined,
        records=records,
        dataset_fingerprint=records_payload["dataset"]["dataset_fingerprint"],
        cache_dir=cache_dir,
    )


def make_split_indices(
    record_count: int,
    *,
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int,
) -> dict[str, list[int]]:
    if train_count + validation_count + test_count != record_count:
        raise ValueError(
            "train + validation + test counts must equal the Phase 0 record count"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(record_count, generator=generator).tolist()
    train_stop = train_count
    validation_stop = train_stop + validation_count
    return {
        "train": permutation[:train_stop],
        "validation": permutation[train_stop:validation_stop],
        "test": permutation[validation_stop:],
    }


def fit_channel_standardization(
    values: torch.Tensor,
    train_indices: Sequence[int],
    *,
    epsilon: float,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    channel_sum = torch.zeros(values.shape[-1], dtype=torch.float64)
    channel_square_sum = torch.zeros_like(channel_sum)
    observation_count = 0
    indices = list(train_indices)
    for start in range(0, len(indices), chunk_size):
        chunk_indices = torch.tensor(indices[start : start + chunk_size], dtype=torch.int64)
        chunk = values[chunk_indices].to(torch.float64)
        channel_sum += chunk.sum(dim=(0, 1))
        channel_square_sum += chunk.square().sum(dim=(0, 1))
        observation_count += int(chunk.shape[0] * chunk.shape[1])
    mean = channel_sum / observation_count
    variance = (
        channel_square_sum / observation_count - mean.square()
    ).clamp_min(0.0)
    std = torch.sqrt(variance + epsilon)
    return mean.to(torch.float32), std.to(torch.float32)

