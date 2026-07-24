from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from vtm_lcg.utils import sha256_file


@dataclass(frozen=True)
class ShardDescriptor:
    index: int
    filename: str
    image_ids: list[int]
    shape: list[int]
    dtype: str
    byte_size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShardDescriptor":
        return cls(
            index=int(payload["index"]),
            filename=str(payload["filename"]),
            image_ids=[int(value) for value in payload["image_ids"]],
            shape=[int(value) for value in payload["shape"]],
            dtype=str(payload["dtype"]),
            byte_size=int(payload["byte_size"]),
            sha256=str(payload["sha256"]),
        )


def canonical_tensor_dtype(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def write_shard_atomic(
    path: Path,
    *,
    shard_index: int,
    values: torch.Tensor,
    image_ids: torch.Tensor,
) -> ShardDescriptor:
    if values.device.type != "cpu" or image_ids.device.type != "cpu":
        raise ValueError("Cache tensors must be moved to CPU before writing")
    if values.ndim != 3 or image_ids.ndim != 1 or values.shape[0] != image_ids.shape[0]:
        raise ValueError(
            f"Invalid shard tensors: values={tuple(values.shape)}, "
            f"image_ids={tuple(image_ids.shape)}"
        )
    if values.dtype is not torch.float16:
        raise ValueError(f"Phase 0 cache values must be float16, got {values.dtype}")
    if image_ids.dtype is not torch.int64:
        raise ValueError(f"image_ids must be int64, got {image_ids.dtype}")

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        save_file(
            {
                "values": values.contiguous(),
                "image_ids": image_ids.contiguous(),
            },
            str(temporary_path),
            metadata={"schema_version": "1", "shard_index": str(shard_index)},
        )
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return ShardDescriptor(
        index=shard_index,
        filename=path.name,
        image_ids=[int(value) for value in image_ids.tolist()],
        shape=[int(value) for value in values.shape],
        dtype=canonical_tensor_dtype(values.dtype),
        byte_size=path.stat().st_size,
        sha256=sha256_file(path),
    )


def validate_shard(
    path: Path,
    descriptor: ShardDescriptor,
    *,
    verify_checksum: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache shard: {path}")
    if path.stat().st_size != descriptor.byte_size:
        raise ValueError(f"Shard byte-size mismatch: {path}")
    if verify_checksum and sha256_file(path) != descriptor.sha256:
        raise ValueError(f"Shard checksum mismatch: {path}")
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != {"values", "image_ids"}:
        raise ValueError(f"Unexpected tensors in {path}: {sorted(tensors)}")
    values = tensors["values"]
    image_ids = tensors["image_ids"]
    if list(values.shape) != descriptor.shape:
        raise ValueError(
            f"Shard shape mismatch in {path}: {list(values.shape)} != {descriptor.shape}"
        )
    if canonical_tensor_dtype(values.dtype) != descriptor.dtype:
        raise ValueError(f"Shard dtype mismatch in {path}: {values.dtype}")
    if image_ids.dtype is not torch.int64:
        raise ValueError(f"Shard image_ids must be int64: {path}")
    if [int(value) for value in image_ids.tolist()] != descriptor.image_ids:
        raise ValueError(f"Shard image ids do not match manifest: {path}")
    return values, image_ids

