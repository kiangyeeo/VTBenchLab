from __future__ import annotations

import gc
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from vtm_lcg.config import canonical_dtype_name, torch_dtype_from_name
from vtm_lcg.models import FrozenClipTextConditioner
from vtm_lcg.utils import (
    atomic_write_json,
    code_fingerprint,
    sha256_file,
    sha256_json,
)


@dataclass
class CaptionEmbeddingStore:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    caption_ids: torch.Tensor
    image_ids: torch.Tensor
    _row_by_caption_id: dict[int, int]

    @classmethod
    def load(cls, path: Path) -> "CaptionEmbeddingStore":
        tensors = load_file(str(path), device="cpu")
        required = {"embeddings", "attention_mask", "caption_ids", "image_ids"}
        if set(tensors) != required:
            raise ValueError(f"Unexpected text-cache tensors: {sorted(tensors)}")
        caption_ids = tensors["caption_ids"]
        if caption_ids.dtype is not torch.int64:
            raise ValueError("caption_ids must be int64")
        row_by_caption_id = {
            int(caption_id): row
            for row, caption_id in enumerate(caption_ids.tolist())
        }
        if len(row_by_caption_id) != caption_ids.numel():
            raise ValueError("Text cache contains duplicate caption ids")
        return cls(
            embeddings=tensors["embeddings"],
            attention_mask=tensors["attention_mask"].to(torch.bool),
            caption_ids=caption_ids,
            image_ids=tensors["image_ids"],
            _row_by_caption_id=row_by_caption_id,
        )

    def get(self, caption_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            rows = [self._row_by_caption_id[int(caption_id)] for caption_id in caption_ids]
        except KeyError as error:
            raise KeyError(f"Caption id is missing from the text cache: {error}") from error
        row_tensor = torch.tensor(rows, dtype=torch.int64)
        return self.embeddings[row_tensor], self.attention_mask[row_tensor]


def _resolve_weight_file(checkpoint: Path) -> Path:
    for filename in ("model.safetensors", "pytorch_model.bin"):
        candidate = checkpoint / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No CLIP text weight file found in {checkpoint}")


def _tokenizer_asset_hashes(checkpoint: Path) -> dict[str, str]:
    filenames = (
        "config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    result: dict[str, str] = {}
    for filename in filenames:
        path = checkpoint / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing CLIP tokenizer asset: {path}")
        result[filename] = sha256_file(path)
    return result


def _atomic_save_text_cache(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        save_file(
            {key: value.contiguous() for key, value in tensors.items()},
            str(temporary_path),
            metadata={"schema_version": "1"},
        )
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def ensure_text_cache(
    *,
    records: list[dict[str, Any]],
    dataset_fingerprint: str,
    text_config: dict[str, Any],
    artifact_root: Path,
    project_root: Path,
    device: torch.device,
    precision_name: str,
) -> tuple[CaptionEmbeddingStore, dict[str, Any], Path]:
    checkpoint = Path(text_config["checkpoint"]).resolve()
    weight_file = _resolve_weight_file(checkpoint)
    identity = {
        "schema_version": 1,
        "dataset_fingerprint": dataset_fingerprint,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(weight_file),
        "tokenizer_assets": _tokenizer_asset_hashes(checkpoint),
        "max_length": int(text_config["max_length"]),
        "cache_dtype": canonical_dtype_name(text_config["cache_dtype"]),
        "conditioner_code_sha256": code_fingerprint(project_root / "vtm_lcg" / "models"),
    }
    cache_key = sha256_json(identity)
    identity["cache_key"] = cache_key
    cache_dir = artifact_root / "text_cache" / cache_key
    identity_path = cache_dir / "identity.json"
    data_path = cache_dir / "caption_embeddings.safetensors"
    manifest_path = cache_dir / "manifest.json"

    if identity_path.is_file() and data_path.is_file():
        existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing_identity == identity:
            data_sha256 = sha256_file(data_path)
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file()
                else None
            )
            checksum_matches = (
                manifest is None or manifest.get("sha256") == data_sha256
            )
            if checksum_matches:
                store = CaptionEmbeddingStore.load(data_path)
                if bool(torch.isfinite(store.embeddings).all()):
                    if manifest is None:
                        manifest = {
                            "schema_version": 1,
                            "cache_key": cache_key,
                            "caption_count": int(store.caption_ids.numel()),
                            "shape": list(store.embeddings.shape),
                            "dtype": canonical_dtype_name(store.embeddings.dtype),
                            "byte_size": data_path.stat().st_size,
                            "sha256": data_sha256,
                        }
                        atomic_write_json(manifest_path, manifest)
                    print("text cache: valid shared caption embeddings reused")
                    return store, identity, data_path

    flattened: list[tuple[int, int, str]] = []
    for record in records:
        caption_ids = record["caption_ids"]
        captions = record["captions"]
        if len(caption_ids) != len(captions):
            raise ValueError(f"Caption metadata mismatch for image {record['image_id']}")
        flattened.extend(
            (int(caption_id), int(record["image_id"]), str(caption))
            for caption_id, caption in zip(caption_ids, captions)
        )
    caption_ids = [item[0] for item in flattened]
    if len(set(caption_ids)) != len(caption_ids):
        raise ValueError("COCO caption ids must be unique")

    backbone_dtype = torch_dtype_from_name(precision_name)
    if device.type == "cpu":
        backbone_dtype = torch.float32
    print(
        f"text cache: encoding {len(flattened)} captions with shared CLIP text tower "
        f"as {backbone_dtype} on {device}"
    )
    conditioner = FrozenClipTextConditioner(
        checkpoint,
        max_length=int(text_config["max_length"]),
        device=device,
        dtype=backbone_dtype,
    )
    embeddings: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    batch_size = int(text_config["encode_batch_size"])
    for start in tqdm(
        range(0, len(flattened), batch_size),
        desc="encode captions",
        unit="batch",
        dynamic_ncols=True,
    ):
        batch = flattened[start : start + batch_size]
        batch_embeddings, batch_attention = conditioner.encode(
            [item[2] for item in batch]
        )
        embeddings.append(batch_embeddings.to(device="cpu", dtype=torch.float16))
        attention_masks.append(batch_attention.to(device="cpu", dtype=torch.bool))
    del conditioner
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tensors = {
        "embeddings": torch.cat(embeddings, dim=0),
        "attention_mask": torch.cat(attention_masks, dim=0),
        "caption_ids": torch.tensor(caption_ids, dtype=torch.int64),
        "image_ids": torch.tensor([item[1] for item in flattened], dtype=torch.int64),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save_text_cache(data_path, tensors)
    atomic_write_json(identity_path, identity)
    manifest = {
        "schema_version": 1,
        "cache_key": cache_key,
        "caption_count": len(flattened),
        "shape": list(tensors["embeddings"].shape),
        "dtype": canonical_dtype_name(tensors["embeddings"].dtype),
        "byte_size": data_path.stat().st_size,
        "sha256": sha256_file(data_path),
    }
    atomic_write_json(manifest_path, manifest)
    return CaptionEmbeddingStore.load(data_path), identity, data_path
