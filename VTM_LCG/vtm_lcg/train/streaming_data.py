from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors.torch import load_file

from vtm_lcg.cache.io import ShardDescriptor, validate_shard


@dataclass
class Phase0ShardCache:
    split_name: str
    tokenizer_id: str
    cache_key: str
    cache_dir: Path
    manifest: dict[str, Any]
    records: list[dict[str, Any]]
    dataset_metadata: dict[str, Any]
    stats: dict[str, Any]
    descriptors: list[ShardDescriptor]

    @classmethod
    def from_summary(
        cls,
        summary_path: Path,
        tokenizer_id: str,
        *,
        split_name: str,
    ) -> "Phase0ShardCache":
        summary_path = summary_path.resolve()
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if not summary.get("all_acceptance_checks_passed"):
            raise RuntimeError(f"Phase 0 {split_name} summary failed: {summary_path}")
        results = {
            result["tokenizer_id"]: result for result in summary["tokenizers"]
        }
        if tokenizer_id not in results:
            raise KeyError(
                f"{tokenizer_id!r} is absent from Phase 0 {split_name} summary"
            )
        result = results[tokenizer_id]
        cache_dir = Path(result["cache_dir"])
        manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        records_payload = json.loads(
            (cache_dir / "records.json").read_text(encoding="utf-8")
        )
        stats = json.loads((cache_dir / "stats.json").read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise RuntimeError(f"Incomplete Phase 0 cache: {cache_dir}")
        descriptors = sorted(
            (
                ShardDescriptor.from_dict(payload)
                for payload in manifest["shards"]
            ),
            key=lambda descriptor: descriptor.index,
        )
        records = records_payload["records"]
        if manifest["record_count"] != len(records):
            raise RuntimeError(f"Manifest/record count mismatch: {cache_dir}")
        if sum(len(descriptor.image_ids) for descriptor in descriptors) != len(records):
            raise RuntimeError(f"Shard/record count mismatch: {cache_dir}")
        return cls(
            split_name=split_name,
            tokenizer_id=tokenizer_id,
            cache_key=str(result["cache_key"]),
            cache_dir=cache_dir,
            manifest=manifest,
            records=records,
            dataset_metadata=records_payload["dataset"],
            stats=stats,
            descriptors=descriptors,
        )

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def shard_size(self) -> int:
        return int(self.manifest["shard_size"])

    def batch_count(self, batch_size: int) -> int:
        return sum(
            math.ceil(len(descriptor.image_ids) / batch_size)
            for descriptor in self.descriptors
        )

    def train_normalization(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor(self.stats["channel_mean"], dtype=torch.float32)
        std = torch.tensor(self.stats["channel_std"], dtype=torch.float32)
        if mean.numel() != int(self.manifest["hidden_dim"]):
            raise RuntimeError(f"Invalid Phase 0 channel statistics: {self.cache_dir}")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
            raise RuntimeError(f"Non-finite Phase 0 channel statistics: {self.cache_dir}")
        if not bool((std > 1.0e-6).all()):
            raise RuntimeError(f"Collapsed Phase 0 channel statistics: {self.cache_dir}")
        return mean, std

    def verify_all(self) -> None:
        for descriptor in self.descriptors:
            validate_shard(
                self.cache_dir / "shards" / descriptor.filename,
                descriptor,
                verify_checksum=True,
            )

    def iter_batches(
        self,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        verify_checksum: bool,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if batch_size <= 0 or batch_size > self.shard_size:
            raise ValueError(
                f"batch_size must be in [1,{self.shard_size}] for shard streaming"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        if shuffle:
            descriptor_order = torch.randperm(
                len(self.descriptors),
                generator=generator,
            ).tolist()
        else:
            descriptor_order = list(range(len(self.descriptors)))

        for descriptor_position in descriptor_order:
            descriptor = self.descriptors[descriptor_position]
            path = self.cache_dir / "shards" / descriptor.filename
            if verify_checksum:
                values, image_ids = validate_shard(
                    path,
                    descriptor,
                    verify_checksum=True,
                )
            else:
                tensors = load_file(str(path), device="cpu")
                values = tensors["values"]
                image_ids = tensors["image_ids"]
            record_offset = descriptor.index * self.shard_size
            record_indices = torch.arange(
                record_offset,
                record_offset + values.shape[0],
                dtype=torch.int64,
            )
            expected_ids = [
                int(self.records[index]["image_id"]) for index in record_indices.tolist()
            ]
            if [int(value) for value in image_ids.tolist()] != expected_ids:
                raise RuntimeError(f"Shard record order mismatch: {path}")
            if shuffle:
                row_order = torch.randperm(values.shape[0], generator=generator)
                values = values[row_order]
                record_indices = record_indices[row_order]
            for start in range(0, values.shape[0], batch_size):
                stop = min(start + batch_size, values.shape[0])
                yield values[start:stop], record_indices[start:stop]


def validate_karpathy_split_caches(
    train_cache: Phase0ShardCache,
    validation_cache: Phase0ShardCache,
    test_cache: Phase0ShardCache,
) -> None:
    caches = (train_cache, validation_cache, test_cache)
    tokenizer_ids = {cache.tokenizer_id for cache in caches}
    if len(tokenizer_ids) != 1:
        raise RuntimeError("Karpathy split caches use different tokenizers")
    hidden_dims = {int(cache.manifest["hidden_dim"]) for cache in caches}
    token_counts = {int(cache.manifest["token_count"]) for cache in caches}
    if len(hidden_dims) != 1 or len(token_counts) != 1:
        raise RuntimeError("Karpathy split caches use different token shapes")
    image_id_sets = [
        {int(record["image_id"]) for record in cache.records}
        for cache in caches
    ]
    if image_id_sets[0] & image_id_sets[1]:
        raise RuntimeError("Karpathy train and validation image ids overlap")
    if image_id_sets[0] & image_id_sets[2]:
        raise RuntimeError("Karpathy train and test image ids overlap")
    if image_id_sets[1] & image_id_sets[2]:
        raise RuntimeError("Karpathy validation and test image ids overlap")

