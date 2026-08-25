#!/usr/bin/env python
"""Deterministic ImageNet linear probe with one-time frozen-feature caching."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


WORKSPACE = Path(__file__).resolve().parents[2]
BASE_SCRIPT_DIR = WORKSPACE / "scripts" / "linear_probe_tokenizers"
if str(BASE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_SCRIPT_DIR))

import linear_probe as base


LOGGER = logging.getLogger("dinov2")
PROTOCOL_VERSION = "tokenizer_linear_probe_deterministic_cached_v1"
CACHE_VERSION = "deterministic_frozen_features_v1"
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE / "outputs" / "vae_linear_probing_dinov2_single_noaug_cached_paperlr"
)
DEFAULT_CACHE_ROOT = DEFAULT_OUTPUT_ROOT / "_feature_cache"


class IndexedDataset(Dataset):
    """Attach stable dataset indices to samples while building a cache."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        return index, image, target


class CachedIndexDataset(Dataset):
    """The loader samples indices; the collator gathers complete feature batches."""

    def __init__(self, size: int):
        self.size = int(size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return int(index)


class CachedBatchCollator:
    def __init__(self, features_path: Path, labels_path: Path):
        self.features_path = str(features_path)
        self.labels_path = str(labels_path)
        self._features = None
        self._labels = None

    def _open(self):
        if self._features is None:
            self._features = np.load(self.features_path, mmap_mode="r")
            self._labels = np.load(self.labels_path, mmap_mode="r")

    def __call__(self, indices):
        self._open()
        batch_indices = np.asarray(indices, dtype=np.int64)
        # Advanced indexing creates one writable, contiguous array per batch.
        # That avoids both per-sample copies and PyTorch's read-only NumPy warning.
        features = np.ascontiguousarray(self._features[batch_indices])
        labels = np.ascontiguousarray(self._labels[batch_indices])
        return torch.from_numpy(features), torch.from_numpy(labels)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_features"] = None
        state["_labels"] = None
        return state


class CachedFeatureTransfer(nn.Module):
    """Move cached FP32 features to the active GPU for heads and metrics."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.to(self.device, non_blocking=True)


def _build_parser():
    parser = base._build_parser()
    parser.description = (
        "DINOv2-style ImageNet linear probing with deterministic preprocessing "
        "and one-time frozen-feature caching"
    )
    parser.set_defaults(output_root=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--cache-root",
        default=str(DEFAULT_CACHE_ROOT),
        help="Persistent root for deterministic train/validation feature caches.",
    )
    parser.add_argument(
        "--cache-batch-size",
        type=int,
        default=base.BATCH_SIZE,
        help="Image loader batch size while extracting each split once.",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=3,
        help=(
            "Stop at this epoch while retaining the original 10-epoch cosine "
            "schedule. Re-run with a larger value to resume the same probe."
        ),
    )
    return parser


def _json_fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_identity(path_string: str) -> dict:
    path = Path(path_string).expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cache_identity(args, bundle, data_root: Path, extra_root: Path) -> dict:
    return {
        "version": CACHE_VERSION,
        "model": args.model,
        "representation": bundle.representation,
        "backbone_precision": bundle.backbone_precision,
        "deterministic_transform": (
            "bundle.eval_transform for both ImageNet TRAIN and VAL; "
            + bundle.transform_description
        ),
        "data_root": str(data_root),
        "extra_root": str(extra_root),
        "checkpoints": [_checkpoint_identity(path) for path in bundle.checkpoint_paths],
        "dtype": "float32",
    }


def _load_valid_cache(cache_dir: Path, identity: dict):
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("identity") != identity:
        return None
    for split in ("train", "val"):
        split_metadata = metadata["splits"][split]
        features_path = cache_dir / f"{split}_features.npy"
        labels_path = cache_dir / f"{split}_labels.npy"
        if not features_path.is_file() or not labels_path.is_file():
            return None
        features = np.load(features_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        if list(features.shape) != split_metadata["feature_shape"]:
            return None
        if list(labels.shape) != split_metadata["label_shape"]:
            return None
        if features.dtype != np.float32 or labels.dtype != np.int64:
            return None
    return metadata


@torch.no_grad()
def _extract_split(
    *,
    split: str,
    dataset,
    feature_model: base.FrozenFeatureModel,
    feature_dim: int,
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
) -> dict:
    feature_tmp = cache_dir / f".{split}_features.npy.tmp"
    label_tmp = cache_dir / f".{split}_labels.npy.tmp"
    features_path = cache_dir / f"{split}_features.npy"
    labels_path = cache_dir / f"{split}_labels.npy"
    feature_array = np.lib.format.open_memmap(
        feature_tmp,
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), feature_dim),
    )
    label_array = np.lib.format.open_memmap(
        label_tmp,
        mode="w+",
        dtype=np.int64,
        shape=(len(dataset),),
    )
    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    LOGGER.info(
        "Extracting deterministic %s features once: samples=%d batches=%d",
        split,
        len(dataset),
        len(loader),
    )
    for batch_index, (indices, images, labels) in enumerate(loader):
        batch_features = feature_model(images).detach().cpu().numpy()
        if batch_features.dtype != np.float32:
            raise RuntimeError(f"Expected FP32 cached features, got {batch_features.dtype}")
        if batch_features.shape != (len(indices), feature_dim):
            raise RuntimeError(
                f"Unexpected {split} feature shape {batch_features.shape}; "
                f"expected {(len(indices), feature_dim)}"
            )
        if not np.isfinite(batch_features).all():
            raise RuntimeError(f"Non-finite values in {split} feature batch {batch_index}")
        numpy_indices = indices.numpy()
        feature_array[numpy_indices] = batch_features
        label_array[numpy_indices] = labels.numpy()
        if batch_index % 100 == 0:
            LOGGER.info(
                "Cached %s batch %d/%d",
                split,
                batch_index + 1,
                len(loader),
            )

    expected_labels = np.asarray(dataset.get_targets(), dtype=np.int64)
    if not np.array_equal(label_array, expected_labels):
        raise RuntimeError(f"Cached {split} labels do not match dataset targets")
    feature_array.flush()
    label_array.flush()
    del feature_array, label_array, loader
    os.replace(feature_tmp, features_path)
    os.replace(label_tmp, labels_path)
    return {
        "feature_shape": [len(dataset), feature_dim],
        "label_shape": [len(dataset)],
    }


def _prepare_cache(args, bundle, feature_model, train_dataset, val_dataset, identity, cache_dir):
    metadata = _load_valid_cache(cache_dir, identity)
    if metadata is not None:
        LOGGER.info("Reusing complete deterministic feature cache: %s", cache_dir)
        return metadata

    sample = train_dataset[0][0].unsqueeze(0)
    feature_dim = base._validate_feature_model(feature_model, sample)
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_metadata = {}
    split_metadata["train"] = _extract_split(
        split="train",
        dataset=train_dataset,
        feature_model=feature_model,
        feature_dim=feature_dim,
        cache_dir=cache_dir,
        batch_size=args.cache_batch_size,
        num_workers=args.num_workers,
    )
    split_metadata["val"] = _extract_split(
        split="val",
        dataset=val_dataset,
        feature_model=feature_model,
        feature_dim=feature_dim,
        cache_dir=cache_dir,
        batch_size=args.cache_batch_size,
        num_workers=args.num_workers,
    )
    metadata = {
        "identity": identity,
        "fingerprint": _json_fingerprint(identity),
        "splits": split_metadata,
    }
    base._atomic_json_dump(cache_dir / "metadata.json", metadata)
    LOGGER.info("Completed deterministic feature cache: %s", cache_dir)
    return metadata


def _make_protocol(args, bundle, effective_lrs, cache_fingerprint: str) -> dict:
    protocol = base._make_protocol(args, bundle, effective_lrs)
    protocol.pop("fingerprint", None)
    protocol.update(
        {
            "version": PROTOCOL_VERSION,
            "transform": (
                "train=bundle.eval_transform (deterministic, no augmentation); "
                "val=bundle.eval_transform; original bundle: "
                + bundle.transform_description
            ),
            "data_augmentation": False,
            "feature_extraction": "once per split, then persistent FP32 cache reuse",
            "feature_cache_dtype": "float32",
            "feature_cache_fingerprint": cache_fingerprint,
            "training_cutoff_semantics": (
                "--stop-after-epoch is an execution cutoff; optimizer and cosine "
                "scheduler retain the full 10-epoch/12500-update horizon"
            ),
        }
    )
    protocol["fingerprint"] = base._protocol_fingerprint(protocol)
    return protocol


def _make_cached_loader(
    *,
    features_path: Path,
    labels_path: Path,
    size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    sampler_advance: int = 0,
):
    return base.make_data_loader(
        dataset=CachedIndexDataset(size),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=train,
        seed=base.SEED,
        sampler_type=(base.SamplerType.SHARDED_INFINITE if train else base.SamplerType.DISTRIBUTED),
        sampler_advance=sampler_advance,
        drop_last=train,
        persistent_workers=num_workers > 0,
        collate_fn=CachedBatchCollator(features_path, labels_path),
    )


def _read_latest_result(output_dir: Path):
    path = output_dir / "results_eval_linear.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This protocol requires one CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, found {torch.cuda.device_count()}; "
            "set CUDA_VISIBLE_DEVICES to one device."
        )
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.feature_microbatch_size <= 0 or args.cache_batch_size <= 0:
        raise ValueError("Feature microbatch and cache batch sizes must be positive")
    if not 1 <= args.stop_after_epoch <= base.EPOCHS:
        raise ValueError(f"--stop-after-epoch must be in [1, {base.EPOCHS}]")

    base.PROTOCOL_VERSION = PROTOCOL_VERSION
    base.distributed.enable(overwrite=True)
    if base.distributed.get_global_size() != 1:
        raise RuntimeError(f"Expected world_size=1, got {base.distributed.get_global_size()}")
    base._seed_everything(base.SEED)
    device = torch.device("cuda", torch.cuda.current_device())

    output_dir = Path(
        args.output_dir or Path(args.output_root) / base.OUTPUT_NAMES[args.model]
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base.setup_logging(output=str(output_dir), level=logging.INFO)

    data_root = Path(args.data_root).expanduser().resolve()
    extra_root = Path(args.extra_root).expanduser().resolve() if args.extra_root else data_root / "extra"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing ImageNet root: {data_root}")
    if not extra_root.is_dir():
        raise FileNotFoundError(f"Missing ImageNet extra directory: {extra_root}")

    train_dataset_str = f"ImageNet:split=TRAIN:root={data_root}:extra={extra_root}"
    val_dataset_str = f"ImageNet:split=VAL:root={data_root}:extra={extra_root}"
    bundle = base.load_feature_bundle(args.model, args, device)
    base._configure_cuda_math()
    feature_model = base.FrozenFeatureModel(
        bundle,
        device=device,
        microbatch_size=args.feature_microbatch_size,
    ).to(device).eval()
    # The key change: TRAIN uses the same deterministic preprocessing as VAL.
    train_dataset = base.make_dataset(dataset_str=train_dataset_str, transform=bundle.eval_transform)
    val_dataset = base.make_dataset(dataset_str=val_dataset_str, transform=bundle.eval_transform)
    train_targets = np.asarray(train_dataset.get_targets(), dtype=np.int64)
    if len(np.unique(train_targets)) != base.NUM_CLASSES:
        raise RuntimeError(
            f"Expected {base.NUM_CLASSES} ImageNet classes, found {len(np.unique(train_targets))}"
        )

    identity = _cache_identity(args, bundle, data_root, extra_root)
    cache_fingerprint = _json_fingerprint(identity)
    cache_dir = (
        Path(args.cache_root).expanduser().resolve()
        / base.OUTPUT_NAMES[args.model]
        / cache_fingerprint[:16]
    )
    metadata = _prepare_cache(
        args,
        bundle,
        feature_model,
        train_dataset,
        val_dataset,
        identity,
        cache_dir,
    )
    feature_dim = int(metadata["splits"]["train"]["feature_shape"][1])
    if metadata["splits"]["val"]["feature_shape"][1] != feature_dim:
        raise RuntimeError("Train and validation cache feature dimensions differ")

    # Release the tokenizer before allocating the LR grid; it is never used again.
    del feature_model, train_dataset, val_dataset
    bundle.encoder = nn.Identity()
    torch.cuda.empty_cache()

    head_grid, parameter_groups, effective_lrs = base._build_heads(feature_dim, device)
    protocol = _make_protocol(args, bundle, effective_lrs, metadata["fingerprint"])
    base._write_or_validate_protocol(output_dir, protocol)
    LOGGER.info("Protocol: %s", json.dumps(protocol, sort_keys=True))
    LOGGER.info("Cached feature dimension=%d; heads=%d", feature_dim, len(head_grid.heads))

    optimizer = torch.optim.SGD(parameter_groups, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, base.MAX_UPDATES, eta_min=0.0
    )
    checkpointer = base.Checkpointer(
        head_grid,
        str(output_dir),
        optimizer=optimizer,
        scheduler=scheduler,
    )
    checkpoint = checkpointer.resume_or_load("", resume=not args.no_resume)
    start_update = int(checkpoint.get("iteration", -1)) + 1
    stop_update = args.stop_after_epoch * base.EPOCH_LENGTH
    if start_update < 0 or start_update > base.MAX_UPDATES:
        raise RuntimeError(f"Invalid checkpoint update: {start_update}")
    if start_update > stop_update:
        raise RuntimeError(
            f"Checkpoint is already at update {start_update}, beyond requested cutoff "
            f"{stop_update}; use a larger --stop-after-epoch or a new --output-dir."
        )

    train_size = int(metadata["splits"]["train"]["feature_shape"][0])
    val_size = int(metadata["splits"]["val"]["feature_shape"][0])
    train_loader = _make_cached_loader(
        features_path=cache_dir / "train_features.npy",
        labels_path=cache_dir / "train_labels.npy",
        size=train_size,
        batch_size=base.BATCH_SIZE,
        num_workers=args.num_workers,
        train=True,
        sampler_advance=start_update * base.BATCH_SIZE,
    )
    val_loader = _make_cached_loader(
        features_path=cache_dir / "val_features.npy",
        labels_path=cache_dir / "val_labels.npy",
        size=val_size,
        batch_size=base.EVAL_BATCH_SIZE,
        num_workers=args.num_workers,
        train=False,
    )
    cached_feature_model = CachedFeatureTransfer(device).eval()
    metrics_path = output_dir / "metrics_history.jsonl"

    LOGGER.info(
        "Starting cached %s from update %d/%d; requested cutoff epoch=%d (%d updates)",
        args.model,
        start_update,
        base.MAX_UPDATES,
        args.stop_after_epoch,
        stop_update,
    )
    if (
        start_update > 0
        and start_update % base.EPOCH_LENGTH == 0
        and not base._metrics_history_has_iteration(metrics_path, start_update)
    ):
        base._evaluate_heads(
            cached_feature_model,
            head_grid,
            val_loader,
            start_update,
            output_dir,
        )

    if start_update < stop_update:
        metric_logger = base.MetricLogger(delimiter="  ")
        remaining_batches = itertools.islice(train_loader, stop_update - start_update)
        update = start_update
        for features, labels in metric_logger.log_every(
            remaining_batches,
            10,
            "Cached training",
            stop_update,
            start_update,
        ):
            labels = labels.to(device, non_blocking=True)
            features = cached_feature_model(features)
            logits = head_grid(features)
            losses = [nn.functional.cross_entropy(output, labels) for output in logits.values()]
            loss = torch.stack(losses).sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            completed_updates = update + 1
            if update % 10 == 0:
                metric_logger.update(
                    loss=float(loss.item()),
                    lr=float(optimizer.param_groups[0]["lr"]),
                )
            if completed_updates % base.EPOCH_LENGTH == 0:
                if completed_updates < base.MAX_UPDATES:
                    checkpointer.save("running_checkpoint_linear_eval", iteration=update)
                if not base._metrics_history_has_iteration(metrics_path, completed_updates):
                    base._evaluate_heads(
                        cached_feature_model,
                        head_grid,
                        val_loader,
                        completed_updates,
                        output_dir,
                    )
            update += 1

    if stop_update == base.MAX_UPDATES:
        checkpointer.save("model_final", iteration=base.MAX_UPDATES - 1)
    result = _read_latest_result(output_dir)
    if result is None or result.get("iteration") != stop_update:
        raise RuntimeError(f"Missing validation result at cutoff update {stop_update}")
    LOGGER.info(
        "Stopped at epoch %d/%d: %s",
        args.stop_after_epoch,
        base.EPOCHS,
        result["best_classifier"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
