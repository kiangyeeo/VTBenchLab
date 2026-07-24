from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vtm_lcg.adapters import create_adapter
from vtm_lcg.cache.dataset import CocoRecord, load_coco_karpathy_records
from vtm_lcg.config import (
    canonical_dtype_name,
    load_config,
    resolve_project_path,
    torch_dtype_from_name,
)
from vtm_lcg.utils import (
    atomic_write_json,
    code_fingerprint,
    git_provenance,
    sha256_file,
    sha256_json,
)

from .views import DeterministicPairedViewTransform, align_flipped_patch_tokens


DEFAULT_VIEW_CONFIG = {
    "horizontal_flip": True,
    "brightness_delta": 0.1,
    "contrast_delta": 0.1,
    "saturation_delta": 0.1,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CrossViewShardDescriptor:
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
    def from_dict(cls, payload: dict[str, Any]) -> "CrossViewShardDescriptor":
        return cls(
            index=int(payload["index"]),
            filename=str(payload["filename"]),
            image_ids=[int(value) for value in payload["image_ids"]],
            shape=[int(value) for value in payload["shape"]],
            dtype=str(payload["dtype"]),
            byte_size=int(payload["byte_size"]),
            sha256=str(payload["sha256"]),
        )


class PairedCocoImageDataset(Dataset):
    def __init__(
        self,
        records: Sequence[CocoRecord],
        indices: Sequence[int],
        transform: DeterministicPairedViewTransform,
    ) -> None:
        self.records = records
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        record_index = self.indices[item]
        record = self.records[record_index]
        with Image.open(record.image_path) as image:
            view_a, view_b = self.transform(
                image.convert("RGB"),
                record.image_id,
            )
        return view_a, view_b, record.image_id, record_index


def write_cross_view_shard(
    path: Path,
    *,
    shard_index: int,
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    image_ids: torch.Tensor,
) -> CrossViewShardDescriptor:
    if view_a.device.type != "cpu" or view_b.device.type != "cpu":
        raise ValueError("cross-view tensors must be on CPU before writing")
    if view_a.shape != view_b.shape or view_a.ndim != 3:
        raise ValueError("view_a and view_b must have matching shape [B,N,D]")
    if image_ids.device.type != "cpu" or image_ids.dtype is not torch.int64:
        raise ValueError("image_ids must be CPU int64")
    if image_ids.ndim != 1 or image_ids.shape[0] != view_a.shape[0]:
        raise ValueError("image_ids must match the cross-view batch")
    if view_a.dtype is not torch.float16 or view_b.dtype is not torch.float16:
        raise ValueError("cross-view cache tensors must be float16")
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
            {
                "view_a": view_a.contiguous(),
                "view_b": view_b.contiguous(),
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
    return CrossViewShardDescriptor(
        index=shard_index,
        filename=path.name,
        image_ids=[int(value) for value in image_ids.tolist()],
        shape=[int(value) for value in view_a.shape],
        dtype="float16",
        byte_size=path.stat().st_size,
        sha256=sha256_file(path),
    )


def validate_cross_view_shard(
    path: Path,
    descriptor: CrossViewShardDescriptor,
    *,
    verify_checksum: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing cross-view shard: {path}")
    if path.stat().st_size != descriptor.byte_size:
        raise ValueError(f"Cross-view shard byte-size mismatch: {path}")
    if verify_checksum and sha256_file(path) != descriptor.sha256:
        raise ValueError(f"Cross-view shard checksum mismatch: {path}")
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != {"view_a", "view_b", "image_ids"}:
        raise ValueError(f"Unexpected tensors in {path}: {sorted(tensors)}")
    view_a = tensors["view_a"]
    view_b = tensors["view_b"]
    image_ids = tensors["image_ids"]
    if list(view_a.shape) != descriptor.shape or view_b.shape != view_a.shape:
        raise ValueError(f"Cross-view shard shape mismatch: {path}")
    if view_a.dtype is not torch.float16 or view_b.dtype is not torch.float16:
        raise ValueError(f"Cross-view shard dtype mismatch: {path}")
    if image_ids.dtype is not torch.int64:
        raise ValueError(f"Cross-view image ids must be int64: {path}")
    if [int(value) for value in image_ids.tolist()] != descriptor.image_ids:
        raise ValueError(f"Cross-view image ids do not match manifest: {path}")
    if not bool(torch.isfinite(view_a).all() and torch.isfinite(view_b).all()):
        raise ValueError(f"Cross-view shard contains non-finite values: {path}")
    return view_a, view_b, image_ids


def _expected_ids(
    records: Sequence[CocoRecord],
    shard_size: int,
) -> dict[int, list[int]]:
    return {
        start // shard_size: [
            record.image_id for record in records[start : start + shard_size]
        ]
        for start in range(0, len(records), shard_size)
    }


def _build_identity(
    *,
    adapter,
    dataset_metadata: dict[str, Any],
    runtime: dict[str, Any],
    view_config: dict[str, Any],
    project_root: Path,
) -> tuple[dict[str, Any], str]:
    provenance = git_provenance(project_root.parent, project_root)
    identity = {
        "schema_version": 1,
        "protocol": "cross_view_aligned_cache_v1",
        "tokenizer": {
            **dict(adapter.metadata),
            "checkpoint_sha256": sha256_file(adapter.checkpoint_file),
        },
        "dataset": dataset_metadata,
        "views": view_config,
        "backbone_dtype": canonical_dtype_name(runtime["backbone_dtype"]),
        "feature_dtype": "float16",
        "source_commit": provenance["source_commit"],
        "source_dirty": provenance["source_dirty"],
        "extractor_code_sha256": code_fingerprint(
            project_root / "vtm_lcg" / "cvrvtm"
        ),
    }
    cache_key = sha256_json(identity)
    identity["cache_key"] = cache_key
    return identity, cache_key


def _initialize_cache(
    cache_dir: Path,
    *,
    identity: dict[str, Any],
    records: Sequence[CocoRecord],
    dataset_metadata: dict[str, Any],
    shard_size: int,
    token_count: int,
    hidden_dim: int,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "shards").mkdir(parents=True, exist_ok=True)
    identity_path = cache_dir / "identity.json"
    records_path = cache_dir / "records.json"
    manifest_path = cache_dir / "manifest.json"
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError(f"Cross-view cache identity collision: {cache_dir}")
    else:
        atomic_write_json(identity_path, identity)
    records_payload = {
        "schema_version": 1,
        "dataset": dataset_metadata,
        "records": [record.to_dict() for record in records],
    }
    if records_path.is_file():
        if json.loads(records_path.read_text(encoding="utf-8")) != records_payload:
            raise RuntimeError(f"Cross-view record manifest changed: {cache_dir}")
    else:
        atomic_write_json(records_path, records_payload)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": 1,
            "protocol": "cross_view_aligned_cache_v1",
            "cache_key": identity["cache_key"],
            "complete": False,
            "record_count": len(records),
            "shard_size": shard_size,
            "expected_shards": (len(records) + shard_size - 1) // shard_size,
            "token_count": token_count,
            "hidden_dim": hidden_dim,
            "feature_dtype": "float16",
            "shards": [],
            "extraction": {"elapsed_seconds": 0.0, "computed_images": 0},
        }
        atomic_write_json(manifest_path, manifest)
    expected = {
        "cache_key": identity["cache_key"],
        "record_count": len(records),
        "shard_size": shard_size,
        "token_count": token_count,
        "hidden_dim": hidden_dim,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Cross-view manifest mismatch: {mismatches}")
    return manifest


def _update_manifest(
    cache_dir: Path,
    manifest: dict[str, Any],
    descriptors: dict[int, CrossViewShardDescriptor],
) -> None:
    manifest["complete"] = len(descriptors) == int(manifest["expected_shards"])
    manifest["shards"] = [
        descriptors[index].to_dict() for index in sorted(descriptors)
    ]
    atomic_write_json(cache_dir / "manifest.json", manifest)


def _find_valid_shards(
    cache_dir: Path,
    manifest: dict[str, Any],
    expected_ids: dict[int, list[int]],
) -> dict[int, CrossViewShardDescriptor]:
    valid: dict[int, CrossViewShardDescriptor] = {}
    token_count = int(manifest["token_count"])
    hidden_dim = int(manifest["hidden_dim"])
    for payload in manifest.get("shards", []):
        try:
            descriptor = CrossViewShardDescriptor.from_dict(payload)
            intended = expected_ids[descriptor.index]
            if descriptor.image_ids != intended:
                raise ValueError("ordered image ids changed")
            if descriptor.shape != [len(intended), token_count, hidden_dim]:
                raise ValueError("cross-view tensor shape changed")
            validate_cross_view_shard(
                cache_dir / "shards" / descriptor.filename,
                descriptor,
                verify_checksum=True,
            )
            valid[descriptor.index] = descriptor
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            print(f"cross-view cache: shard will be recomputed: {payload!r}: {error}")
    return valid


def _extract_missing(
    *,
    adapter,
    records: list[CocoRecord],
    cache_dir: Path,
    manifest: dict[str, Any],
    descriptors: dict[int, CrossViewShardDescriptor],
    expected_ids: dict[int, list[int]],
    runtime: dict[str, Any],
    view_config: dict[str, Any],
    device: torch.device,
) -> dict[int, CrossViewShardDescriptor]:
    missing = sorted(set(expected_ids) - set(descriptors))
    if not missing:
        print(f"{adapter.tokenizer_id}: valid cross-view shards reused")
        return descriptors
    shard_size = int(runtime["shard_size"])
    indices = [
        index
        for shard_index in missing
        for index in range(
            shard_index * shard_size,
            min((shard_index + 1) * shard_size, len(records)),
        )
    ]
    transform = DeterministicPairedViewTransform(
        adapter.preprocess_config,
        view_config,
    )
    dataset = PairedCocoImageDataset(records, indices, transform)
    num_workers = int(runtime["num_workers"])
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(runtime["batch_size"]),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(**loader_kwargs)
    backbone_dtype = torch_dtype_from_name(runtime["backbone_dtype"])
    if device.type == "cpu":
        backbone_dtype = torch.float32
    print(
        f"{adapter.tokenizer_id}: loading {adapter.checkpoint_file.name} "
        f"for paired views as {backbone_dtype} on {device}"
    )
    adapter.load(device, backbone_dtype)
    a_buffers: dict[int, list[torch.Tensor]] = {}
    b_buffers: dict[int, list[torch.Tensor]] = {}
    id_buffers: dict[int, list[torch.Tensor]] = {}
    started = time.perf_counter()
    computed = 0
    progress = tqdm(
        total=len(indices),
        desc=f"extract paired {adapter.tokenizer_id}",
        unit="image",
        dynamic_ncols=True,
    )
    with torch.inference_mode():
        for images_a, images_b, image_ids, record_indices in loader:
            values_a = adapter.encode(images_a).values
            values_b = adapter.encode(images_b).values
            values_b = align_flipped_patch_tokens(
                values_b,
                grid_shape=adapter.grid_shape,
                horizontal_flip=bool(view_config["horizontal_flip"]),
            )
            values_a = values_a.detach().to("cpu", dtype=torch.float16)
            values_b = values_b.detach().to("cpu", dtype=torch.float16)
            image_ids = image_ids.to("cpu", dtype=torch.int64)
            record_indices = record_indices.to("cpu", dtype=torch.int64)
            batch_shards = torch.div(
                record_indices,
                shard_size,
                rounding_mode="floor",
            )
            for shard_tensor in torch.unique_consecutive(batch_shards):
                shard_index = int(shard_tensor.item())
                selection = batch_shards == shard_index
                a_buffers.setdefault(shard_index, []).append(values_a[selection])
                b_buffers.setdefault(shard_index, []).append(values_b[selection])
                id_buffers.setdefault(shard_index, []).append(image_ids[selection])
                buffered = sum(item.shape[0] for item in id_buffers[shard_index])
                intended = expected_ids[shard_index]
                if buffered == len(intended):
                    shard_a = torch.cat(a_buffers.pop(shard_index), dim=0)
                    shard_b = torch.cat(b_buffers.pop(shard_index), dim=0)
                    shard_ids = torch.cat(id_buffers.pop(shard_index), dim=0)
                    if [int(value) for value in shard_ids.tolist()] != intended:
                        raise RuntimeError(
                            f"Cross-view DataLoader order mismatch: {shard_index}"
                        )
                    descriptor = write_cross_view_shard(
                        cache_dir / "shards" / f"{shard_index:05d}.safetensors",
                        shard_index=shard_index,
                        view_a=shard_a,
                        view_b=shard_b,
                        image_ids=shard_ids,
                    )
                    descriptors[shard_index] = descriptor
                    _update_manifest(cache_dir, manifest, descriptors)
                elif buffered > len(intended):
                    raise RuntimeError(
                        f"Cross-view shard {shard_index} received too many rows"
                    )
            batch_count = int(images_a.shape[0])
            computed += batch_count
            progress.update(batch_count)
    progress.close()
    if a_buffers or b_buffers or id_buffers:
        raise RuntimeError("Incomplete cross-view shard buffers remained")
    elapsed = time.perf_counter() - started
    extraction = manifest["extraction"]
    extraction["elapsed_seconds"] = float(extraction["elapsed_seconds"]) + elapsed
    extraction["computed_images"] = int(extraction["computed_images"]) + computed
    extraction["last_run_elapsed_seconds"] = elapsed
    extraction["last_run_computed_images"] = computed
    extraction["last_run_images_per_second"] = computed / max(elapsed, 1.0e-12)
    _update_manifest(cache_dir, manifest, descriptors)
    return descriptors


def _compute_stats(
    cache_dir: Path,
    descriptors: Sequence[CrossViewShardDescriptor],
    *,
    epsilon: float,
) -> dict[str, Any]:
    channel_sum: torch.Tensor | None = None
    channel_square_sum: torch.Tensor | None = None
    observation_count = 0
    for descriptor in tqdm(
        descriptors,
        desc="cross-view stats",
        unit="shard",
        dynamic_ncols=True,
        leave=False,
    ):
        view_a, view_b, _ = validate_cross_view_shard(
            cache_dir / "shards" / descriptor.filename,
            descriptor,
            verify_checksum=False,
        )
        values = torch.cat((view_a, view_b), dim=0).to(torch.float64)
        current_sum = values.sum(dim=(0, 1))
        current_square_sum = values.square().sum(dim=(0, 1))
        channel_sum = (
            current_sum if channel_sum is None else channel_sum + current_sum
        )
        channel_square_sum = (
            current_square_sum
            if channel_square_sum is None
            else channel_square_sum + current_square_sum
        )
        observation_count += int(values.shape[0] * values.shape[1])
    if channel_sum is None or channel_square_sum is None:
        raise RuntimeError("Cannot compute stats for an empty cross-view cache")
    mean = channel_sum / observation_count
    variance = (channel_square_sum / observation_count - mean.square()).clamp_min(0)
    std = (variance + epsilon).sqrt()
    finite = bool(torch.isfinite(mean).all() and torch.isfinite(std).all())
    acceptance = {
        "finite": finite,
        "all_channel_std_above_1e-6": bool((std > 1.0e-6).all()),
    }
    if not all(acceptance.values()):
        raise RuntimeError(f"Cross-view cache statistics failed: {acceptance}")
    return {
        "schema_version": 1,
        "observation_count": observation_count,
        "channel_mean": mean.tolist(),
        "channel_std": std.tolist(),
        "channel_std_min": float(std.min().item()),
        "channel_std_max": float(std.max().item()),
        "epsilon": epsilon,
        "acceptance": acceptance,
    }


def extract_tokenizer(
    *,
    tokenizer_config: dict[str, Any],
    config: dict[str, Any],
    project_root: Path,
    records: list[CocoRecord],
    dataset_metadata: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    runtime = config["runtime"]
    view_config = {**DEFAULT_VIEW_CONFIG, **dict(config.get("cross_view", {}))}
    adapter = create_adapter(tokenizer_config, config["preprocess"])
    identity, cache_key = _build_identity(
        adapter=adapter,
        dataset_metadata=dataset_metadata,
        runtime=runtime,
        view_config=view_config,
        project_root=project_root,
    )
    artifact_root = Path(runtime["artifact_root"])
    cache_dir = artifact_root / "cache" / adapter.tokenizer_id / cache_key
    token_count = adapter.grid_shape[0] * adapter.grid_shape[1]
    hidden_dim = adapter.expected_hidden_dim
    manifest = _initialize_cache(
        cache_dir,
        identity=identity,
        records=records,
        dataset_metadata=dataset_metadata,
        shard_size=int(runtime["shard_size"]),
        token_count=token_count,
        hidden_dim=hidden_dim,
    )
    expected_ids = _expected_ids(records, int(runtime["shard_size"]))
    descriptors = _find_valid_shards(cache_dir, manifest, expected_ids)
    _update_manifest(cache_dir, manifest, descriptors)
    descriptors = _extract_missing(
        adapter=adapter,
        records=records,
        cache_dir=cache_dir,
        manifest=manifest,
        descriptors=descriptors,
        expected_ids=expected_ids,
        runtime=runtime,
        view_config=view_config,
        device=device,
    )
    ordered = [descriptors[index] for index in sorted(descriptors)]
    if len(ordered) != len(expected_ids):
        raise RuntimeError(f"Incomplete cross-view cache: {adapter.tokenizer_id}")
    stats_path = cache_dir / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats.get("cache_key") != cache_key:
            stats = {}
    else:
        stats = {}
    if not stats:
        stats = _compute_stats(
            cache_dir,
            ordered,
            epsilon=float(runtime["stats_epsilon"]),
        )
        stats["cache_key"] = cache_key
        stats["tokenizer_id"] = adapter.tokenizer_id
        atomic_write_json(stats_path, stats)
    del adapter
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "tokenizer_id": tokenizer_config["id"],
        "cache_key": cache_key,
        "cache_dir": str(cache_dir),
        "dataset_fingerprint": dataset_metadata["dataset_fingerprint"],
        "record_count": len(records),
        "token_count": token_count,
        "hidden_dim": hidden_dim,
        "surface": tokenizer_config["surface"],
        "checkpoint_sha256": identity["tokenizer"]["checkpoint_sha256"],
        "views": view_config,
        "stats": {
            "channel_std_min": stats["channel_std_min"],
            "channel_std_max": stats["channel_std_max"],
            "cache_bytes": sum(item.byte_size for item in ordered),
            "acceptance": stats["acceptance"],
            "extraction": manifest["extraction"],
        },
    }


def write_summary(artifact_root: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        raise ValueError("Cannot write an empty cross-view summary")
    if len({item["dataset_fingerprint"] for item in results}) != 1:
        raise RuntimeError("Cross-view caches use different datasets")
    payload = {
        "schema_version": 1,
        "protocol": "cross_view_aligned_cache_v1",
        "generated_at": utc_now(),
        "dataset_fingerprint": results[0]["dataset_fingerprint"],
        "record_count": results[0]["record_count"],
        "all_acceptance_checks_passed": all(
            all(item["stats"]["acceptance"].values()) for item in results
        ),
        "tokenizers": results,
    }
    atomic_write_json(artifact_root / "summary.json", payload)
    lines = [
        "# CV-RVTM Cross-view Cache Summary",
        "",
        f"- Records: `{payload['record_count']}`",
        f"- Dataset fingerprint: `{payload['dataset_fingerprint']}`",
        "",
        "| Tokenizer | Shape/view | Min std | Cache GiB |",
        "|---|---:|---:|---:|",
    ]
    for item in results:
        stats = item["stats"]
        lines.append(
            f"| {item['tokenizer_id']} | "
            f"{item['token_count']}×{item['hidden_dim']} | "
            f"{stats['channel_std_min']:.6g} | "
            f"{stats['cache_bytes'] / 2**30:.3f} |"
        )
    lines.append("")
    artifact_root.mkdir(parents=True, exist_ok=True)
    temporary = artifact_root / "summary.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, artifact_root / "summary.md")


def _resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    return device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract aligned deterministic paired-view caches for CV-RVTM"
    )
    parser.add_argument("--config", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--tokenizer", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    config, project_root = load_config(args.config)
    runtime = config["runtime"]
    if args.limit is not None:
        config["dataset"]["limit"] = args.limit
    for argument_name in ("device", "batch_size", "shard_size", "num_workers"):
        value = getattr(args, argument_name)
        if value is not None:
            runtime[argument_name] = value
    if args.artifact_root is not None:
        runtime["artifact_root"] = str(
            resolve_project_path(project_root, args.artifact_root)
        )
    runtime["feature_dtype"] = "float16"
    tokenizers = {item["id"]: item for item in config["tokenizers"]}
    if args.all:
        selected = list(tokenizers)
    else:
        unknown = sorted(set(args.tokenizer) - set(tokenizers))
        if unknown:
            raise ValueError(f"Unknown tokenizer ids: {unknown}")
        selected = list(args.tokenizer)
    records, dataset_metadata = load_coco_karpathy_records(
        Path(config["dataset"]["annotations"]),
        Path(config["dataset"]["image_root"]),
        limit=config["dataset"].get("limit"),
    )
    device = _resolve_device(str(runtime["device"]))
    for tokenizer_id in selected:
        adapter = create_adapter(tokenizers[tokenizer_id], config["preprocess"])
        if not adapter.checkpoint_file.is_file():
            raise FileNotFoundError(adapter.checkpoint_file)
        if adapter.grid_shape != (16, 16) or adapter.expected_hidden_dim != 1024:
            raise ValueError(
                f"CV-RVTM v1 requires 16x16x1024 tokens: {tokenizer_id}"
            )
    print(
        f"CV-RVTM cache: records={len(records)} tokenizers={selected} device={device}"
    )
    if args.preflight_only:
        return 0
    results = [
        extract_tokenizer(
            tokenizer_config=tokenizers[tokenizer_id],
            config=config,
            project_root=project_root,
            records=records,
            dataset_metadata=dataset_metadata,
            device=device,
        )
        for tokenizer_id in selected
    ]
    write_summary(Path(runtime["artifact_root"]), results)
    print(f"CV-RVTM cache summary: {Path(runtime['artifact_root']) / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
