from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vtm_lcg.adapters import create_adapter
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

from .dataset import CocoImageTensorDataset, CocoRecord, load_coco_karpathy_records
from .io import ShardDescriptor, validate_shard, write_shard_atomic
from .stats import compute_cache_stats, validate_stats_acceptance


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_device(name: str) -> torch.device:
    normalized = name.lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Run with GPU access or pass --device cpu "
            "for a small integration check."
        )
    return device


def preflight(
    config: dict[str, Any],
    selected_tokenizers: list[dict[str, Any]],
    records: list[CocoRecord],
    device: torch.device,
) -> None:
    runtime = config["runtime"]
    batch_size = int(runtime["batch_size"])
    shard_size = int(runtime["shard_size"])
    if batch_size <= 0 or shard_size <= 0:
        raise ValueError("batch_size and shard_size must be positive")
    if shard_size < batch_size:
        raise ValueError("shard_size must be greater than or equal to batch_size")
    if canonical_dtype_name(runtime["feature_dtype"]) != "float16":
        raise ValueError("Phase 0 cache protocol requires feature_dtype=float16")
    if not records:
        raise ValueError("The selected COCO dataset is empty")

    for tokenizer_config in selected_tokenizers:
        adapter = create_adapter(tokenizer_config, config["preprocess"])
        if not adapter.checkpoint_file.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint for {adapter.tokenizer_id}: {adapter.checkpoint_file}"
            )

    print(
        f"preflight: records={len(records)} batch_size={batch_size} "
        f"shard_size={shard_size} device={device}"
    )
    if device.type == "cuda":
        device_index = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        properties = torch.cuda.get_device_properties(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        print(
            f"preflight: gpu={properties.name} "
            f"free={free_bytes / 2**30:.1f}GiB total={total_bytes / 2**30:.1f}GiB"
        )


def build_cache_identity(
    *,
    adapter,
    dataset_metadata: dict[str, Any],
    runtime: dict[str, Any],
    project_root: Path,
) -> tuple[dict[str, Any], str]:
    workspace_root = project_root.parent
    checkpoint_sha256 = sha256_file(adapter.checkpoint_file)
    provenance = git_provenance(workspace_root, project_root)
    identity = {
        "schema_version": 1,
        "tokenizer": {
            **dict(adapter.metadata),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "dataset": dataset_metadata,
        "backbone_dtype": canonical_dtype_name(runtime["backbone_dtype"]),
        "feature_dtype": canonical_dtype_name(runtime["feature_dtype"]),
        "source_commit": provenance["source_commit"],
        "source_dirty": provenance["source_dirty"],
        "extractor_code_sha256": code_fingerprint(project_root / "vtm_lcg"),
    }
    cache_key = sha256_json(identity)
    identity["cache_key"] = cache_key
    return identity, cache_key


def expected_shard_image_ids(
    records: list[CocoRecord],
    shard_size: int,
) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for start in range(0, len(records), shard_size):
        shard_index = start // shard_size
        result[shard_index] = [
            record.image_id for record in records[start : start + shard_size]
        ]
    return result


def initialize_cache_files(
    cache_dir: Path,
    *,
    identity: dict[str, Any],
    records: list[CocoRecord],
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
        if read_json(identity_path) != identity:
            raise RuntimeError(f"Cache identity collision at {cache_dir}")
    else:
        atomic_write_json(identity_path, identity)

    records_payload = {
        "schema_version": 1,
        "dataset": dataset_metadata,
        "records": [record.to_dict() for record in records],
    }
    if records_path.is_file():
        if read_json(records_path) != records_payload:
            raise RuntimeError(f"Cache record manifest changed at {cache_dir}")
    else:
        atomic_write_json(records_path, records_payload)

    expected_shards = (len(records) + shard_size - 1) // shard_size
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("cache_key") != identity["cache_key"]:
            raise RuntimeError(f"Manifest cache key mismatch at {cache_dir}")
    else:
        now = utc_now()
        manifest = {
            "schema_version": 1,
            "cache_key": identity["cache_key"],
            "created_at": now,
            "updated_at": now,
            "complete": False,
            "record_count": len(records),
            "shard_size": shard_size,
            "expected_shards": expected_shards,
            "token_count": token_count,
            "hidden_dim": hidden_dim,
            "feature_dtype": identity["feature_dtype"],
            "shards": [],
            "extraction": {
                "elapsed_seconds": 0.0,
                "computed_images": 0,
            },
        }
        atomic_write_json(manifest_path, manifest)

    expected_fields = {
        "record_count": len(records),
        "shard_size": shard_size,
        "expected_shards": expected_shards,
        "token_count": token_count,
        "hidden_dim": hidden_dim,
        "feature_dtype": identity["feature_dtype"],
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in expected_fields.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Existing manifest protocol mismatch: {mismatches}")
    return manifest


def find_valid_shards(
    cache_dir: Path,
    manifest: dict[str, Any],
    expected_ids: dict[int, list[int]],
    *,
    token_count: int,
    hidden_dim: int,
) -> dict[int, ShardDescriptor]:
    valid: dict[int, ShardDescriptor] = {}
    for raw_descriptor in manifest.get("shards", []):
        try:
            descriptor = ShardDescriptor.from_dict(raw_descriptor)
            intended_ids = expected_ids.get(descriptor.index)
            expected_shape = [len(intended_ids or []), token_count, hidden_dim]
            if intended_ids is None:
                raise ValueError("unexpected shard index")
            if descriptor.image_ids != intended_ids or descriptor.shape != expected_shape:
                raise ValueError("shard does not match current ordered dataset")
            validate_shard(
                cache_dir / "shards" / descriptor.filename,
                descriptor,
                verify_checksum=True,
            )
            valid[descriptor.index] = descriptor
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            print(f"cache: shard will be recomputed: {raw_descriptor!r}: {error}")
    return valid


def update_manifest(
    cache_dir: Path,
    manifest: dict[str, Any],
    descriptors: dict[int, ShardDescriptor],
    *,
    complete: bool,
) -> None:
    manifest["updated_at"] = utc_now()
    manifest["complete"] = complete
    manifest["shards"] = [
        descriptors[index].to_dict() for index in sorted(descriptors)
    ]
    atomic_write_json(cache_dir / "manifest.json", manifest)


def extract_missing_shards(
    *,
    adapter,
    records: list[CocoRecord],
    cache_dir: Path,
    manifest: dict[str, Any],
    descriptors: dict[int, ShardDescriptor],
    expected_ids: dict[int, list[int]],
    runtime: dict[str, Any],
    device: torch.device,
) -> dict[int, ShardDescriptor]:
    missing_shards = sorted(set(expected_ids) - set(descriptors))
    if not missing_shards:
        print(f"{adapter.tokenizer_id}: all shards are valid; feature extraction skipped")
        return descriptors

    batch_size = int(runtime["batch_size"])
    shard_size = int(runtime["shard_size"])
    num_workers = int(runtime["num_workers"])
    backbone_dtype = torch_dtype_from_name(runtime["backbone_dtype"])
    if device.type == "cpu" and backbone_dtype is not torch.float32:
        print("runtime: forcing float32 backbone dtype on CPU")
        backbone_dtype = torch.float32

    missing_indices: list[int] = []
    for shard_index in missing_shards:
        start = shard_index * shard_size
        stop = min(start + shard_size, len(records))
        missing_indices.extend(range(start, stop))
    dataset = CocoImageTensorDataset(records, missing_indices, adapter.preprocess)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)

    print(
        f"{adapter.tokenizer_id}: loading {adapter.checkpoint_file.name} "
        f"as {backbone_dtype} on {device}"
    )
    adapter.load(device, backbone_dtype)
    shard_value_buffers: dict[int, list[torch.Tensor]] = {}
    shard_id_buffers: dict[int, list[torch.Tensor]] = {}
    started_at = time.perf_counter()
    computed_images = 0

    progress = tqdm(
        total=len(missing_indices),
        desc=f"extract {adapter.tokenizer_id}",
        unit="image",
        dynamic_ncols=True,
    )
    with torch.inference_mode():
        for images, image_ids, record_indices in loader:
            token_batch = adapter.encode(images)
            values = token_batch.values.detach().to(
                device="cpu",
                dtype=torch.float16,
                non_blocking=False,
            )
            image_ids = image_ids.to(dtype=torch.int64, device="cpu")
            record_indices = record_indices.to(dtype=torch.int64, device="cpu")

            batch_shards = torch.div(record_indices, shard_size, rounding_mode="floor")
            for shard_tensor in torch.unique_consecutive(batch_shards):
                shard_index = int(shard_tensor.item())
                selection = batch_shards == shard_index
                shard_value_buffers.setdefault(shard_index, []).append(values[selection])
                shard_id_buffers.setdefault(shard_index, []).append(image_ids[selection])
                buffered_count = sum(
                    tensor.shape[0] for tensor in shard_id_buffers[shard_index]
                )
                intended_count = len(expected_ids[shard_index])
                if buffered_count == intended_count:
                    shard_values = torch.cat(shard_value_buffers.pop(shard_index), dim=0)
                    shard_image_ids = torch.cat(shard_id_buffers.pop(shard_index), dim=0)
                    intended_ids = expected_ids[shard_index]
                    if [int(value) for value in shard_image_ids.tolist()] != intended_ids:
                        raise RuntimeError(
                            f"DataLoader order mismatch in shard {shard_index}"
                        )
                    shard_path = (
                        cache_dir / "shards" / f"{shard_index:05d}.safetensors"
                    )
                    descriptor = write_shard_atomic(
                        shard_path,
                        shard_index=shard_index,
                        values=shard_values,
                        image_ids=shard_image_ids,
                    )
                    descriptors[shard_index] = descriptor
                    update_manifest(
                        cache_dir,
                        manifest,
                        descriptors,
                        complete=len(descriptors) == len(expected_ids),
                    )
                elif buffered_count > intended_count:
                    raise RuntimeError(f"Shard {shard_index} received too many samples")

            batch_count = int(images.shape[0])
            computed_images += batch_count
            progress.update(batch_count)
    progress.close()

    if shard_value_buffers or shard_id_buffers:
        raise RuntimeError("Incomplete shard buffers remained after extraction")
    elapsed = time.perf_counter() - started_at
    extraction = manifest.setdefault("extraction", {})
    extraction["elapsed_seconds"] = float(
        extraction.get("elapsed_seconds", 0.0)
    ) + elapsed
    extraction["computed_images"] = int(
        extraction.get("computed_images", 0)
    ) + computed_images
    extraction["last_run_elapsed_seconds"] = elapsed
    extraction["last_run_computed_images"] = computed_images
    extraction["last_run_images_per_second"] = computed_images / max(elapsed, 1.0e-12)
    update_manifest(
        cache_dir,
        manifest,
        descriptors,
        complete=len(descriptors) == len(expected_ids),
    )
    return descriptors


def load_or_compute_stats(
    *,
    cache_dir: Path,
    cache_key: str,
    tokenizer_id: str,
    descriptors: list[ShardDescriptor],
    manifest: dict[str, Any],
    epsilon: float,
) -> dict[str, Any]:
    stats_path = cache_dir / "stats.json"
    if manifest.get("complete") and stats_path.is_file():
        stats = read_json(stats_path)
        if stats.get("cache_key") == cache_key:
            validate_stats_acceptance(stats)
            print(f"{tokenizer_id}: valid cached statistics reused")
            return stats

    print(f"{tokenizer_id}: computing streaming statistics and normalized readback")
    stats = compute_cache_stats(cache_dir, descriptors, epsilon=epsilon)
    stats.update(
        {
            "cache_key": cache_key,
            "tokenizer_id": tokenizer_id,
            "cache_bytes": sum(descriptor.byte_size for descriptor in descriptors),
            "computed_at": utc_now(),
            "extraction": dict(manifest.get("extraction", {})),
        }
    )
    validate_stats_acceptance(stats)
    atomic_write_json(stats_path, stats)
    return stats


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
    adapter = create_adapter(tokenizer_config, config["preprocess"])
    print(f"{adapter.tokenizer_id}: hashing checkpoint for cache identity")
    identity, cache_key = build_cache_identity(
        adapter=adapter,
        dataset_metadata=dataset_metadata,
        runtime=runtime,
        project_root=project_root,
    )
    artifact_root = Path(runtime["artifact_root"])
    cache_dir = artifact_root / "cache" / adapter.tokenizer_id / cache_key
    token_count = adapter.grid_shape[0] * adapter.grid_shape[1]
    hidden_dim = adapter.expected_hidden_dim
    shard_size = int(runtime["shard_size"])
    manifest = initialize_cache_files(
        cache_dir,
        identity=identity,
        records=records,
        dataset_metadata=dataset_metadata,
        shard_size=shard_size,
        token_count=token_count,
        hidden_dim=hidden_dim,
    )
    expected_ids = expected_shard_image_ids(records, shard_size)
    descriptors = find_valid_shards(
        cache_dir,
        manifest,
        expected_ids,
        token_count=token_count,
        hidden_dim=hidden_dim,
    )
    update_manifest(
        cache_dir,
        manifest,
        descriptors,
        complete=len(descriptors) == len(expected_ids),
    )
    descriptors = extract_missing_shards(
        adapter=adapter,
        records=records,
        cache_dir=cache_dir,
        manifest=manifest,
        descriptors=descriptors,
        expected_ids=expected_ids,
        runtime=runtime,
        device=device,
    )
    ordered_descriptors = [descriptors[index] for index in sorted(descriptors)]
    if len(ordered_descriptors) != len(expected_ids):
        raise RuntimeError(f"{adapter.tokenizer_id}: cache is incomplete")
    manifest = read_json(cache_dir / "manifest.json")
    stats = load_or_compute_stats(
        cache_dir=cache_dir,
        cache_key=cache_key,
        tokenizer_id=adapter.tokenizer_id,
        descriptors=ordered_descriptors,
        manifest=manifest,
        epsilon=float(runtime["stats_epsilon"]),
    )
    result = {
        "tokenizer_id": adapter.tokenizer_id,
        "cache_key": cache_key,
        "cache_dir": str(cache_dir),
        "dataset_fingerprint": dataset_metadata["dataset_fingerprint"],
        "record_count": len(records),
        "token_count": token_count,
        "hidden_dim": hidden_dim,
        "surface": adapter.surface,
        "checkpoint_sha256": identity["tokenizer"]["checkpoint_sha256"],
        "stats": {
            key: stats[key]
            for key in (
                "nan_count",
                "inf_count",
                "channel_std_min",
                "channel_std_max",
                "mean_token_variance",
                "normalized_max_abs_channel_mean",
                "normalized_max_abs_channel_std_error",
                "cache_bytes",
                "acceptance",
                "extraction",
            )
        },
    }
    return result


def write_summary(artifact_root: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        raise ValueError("Cannot write an empty Phase 0 summary")
    fingerprints = {result["dataset_fingerprint"] for result in results}
    record_counts = {result["record_count"] for result in results}
    if len(fingerprints) != 1 or len(record_counts) != 1:
        raise RuntimeError("Tokenizer caches do not use the same ordered dataset")
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "dataset_fingerprint": results[0]["dataset_fingerprint"],
        "record_count": results[0]["record_count"],
        "all_acceptance_checks_passed": all(
            all(result["stats"]["acceptance"].values()) for result in results
        ),
        "tokenizers": results,
    }
    atomic_write_json(artifact_root / "summary.json", payload)

    lines = [
        "# VTM-LCG Phase 0 Summary",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Ordered COCO records: `{payload['record_count']}`",
        f"- Dataset fingerprint: `{payload['dataset_fingerprint']}`",
        f"- All acceptance checks passed: `{payload['all_acceptance_checks_passed']}`",
        "",
        "| Tokenizer | Shape per image | NaN/Inf | Min channel std | Token variance | "
        "Norm mean error | Norm std error | Cache GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        stats = result["stats"]
        lines.append(
            f"| {result['tokenizer_id']} | {result['token_count']}×{result['hidden_dim']} | "
            f"{stats['nan_count']}/{stats['inf_count']} | "
            f"{stats['channel_std_min']:.6g} | {stats['mean_token_variance']:.6g} | "
            f"{stats['normalized_max_abs_channel_mean']:.3g} | "
            f"{stats['normalized_max_abs_channel_std_error']:.3g} | "
            f"{stats['cache_bytes'] / 2**30:.3f} |"
        )
    lines.append("")
    summary_markdown = artifact_root / "summary.md"
    summary_markdown.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_markdown.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, summary_markdown)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and validate the VTM-LCG Phase 0 visual-token cache"
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
    parser.add_argument("--backbone-dtype", default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    config, project_root = load_config(args.config)
    runtime = config["runtime"]
    dataset_config = config["dataset"]

    if args.limit is not None:
        dataset_config["limit"] = args.limit
    for argument_name, config_name in (
        ("device", "device"),
        ("batch_size", "batch_size"),
        ("shard_size", "shard_size"),
        ("num_workers", "num_workers"),
        ("backbone_dtype", "backbone_dtype"),
    ):
        value = getattr(args, argument_name)
        if value is not None:
            runtime[config_name] = value
    if args.artifact_root is not None:
        runtime["artifact_root"] = str(
            resolve_project_path(project_root, args.artifact_root)
        )

    tokenizers_by_id = {
        tokenizer["id"]: tokenizer for tokenizer in config["tokenizers"]
    }
    if args.all:
        selected_tokenizers = list(config["tokenizers"])
    else:
        unknown = sorted(set(args.tokenizer) - set(tokenizers_by_id))
        if unknown:
            raise ValueError(
                f"Unknown tokenizer ids {unknown}; choices={sorted(tokenizers_by_id)}"
            )
        selected_tokenizers = [tokenizers_by_id[name] for name in args.tokenizer]

    records, dataset_metadata = load_coco_karpathy_records(
        Path(dataset_config["annotations"]),
        Path(dataset_config["image_root"]),
        limit=int(dataset_config["limit"]),
    )
    device = resolve_device(str(runtime["device"]))
    preflight(config, selected_tokenizers, records, device)
    if args.preflight_only:
        return 0

    artifact_root = Path(runtime["artifact_root"])
    results: list[dict[str, Any]] = []
    for tokenizer_config in selected_tokenizers:
        result = extract_tokenizer(
            tokenizer_config=tokenizer_config,
            config=config,
            project_root=project_root,
            records=records,
            dataset_metadata=dataset_metadata,
            device=device,
        )
        results.append(result)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_summary(artifact_root, results)
    print(f"phase0: summary={artifact_root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except torch.cuda.OutOfMemoryError:
        print(
            "CUDA out of memory. Re-run with --batch-size 16 (or lower); "
            "valid completed shards will be reused.",
            file=sys.stderr,
        )
        raise
