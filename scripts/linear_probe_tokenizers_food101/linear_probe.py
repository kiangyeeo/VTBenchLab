#!/usr/bin/env python
"""Balanced Food-101 probing for the Tokenizer_set_up.md model panel."""

import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
from torch.utils.data import Dataset


WORKSPACE = Path(__file__).resolve().parents[2]
DRIVER_PATH = (
    WORKSPACE / "scripts" / "linear_probe_tokenizers" / "dataset_probe_driver.py"
)
MANIFEST_PATH = Path(__file__).resolve().parent / "tokenizers_from_setup.tsv"
OUTPUT_ROOT = WORKSPACE / "outputs" / "food101_linear_probing_dinov2_single_surface"
PROTOCOL_VERSION = "tokenizer_linear_probe_food101_balanced_epoch_barrier_v2"
DATASET_NAME = "food101"
DATASET_DISPLAY_NAME = "Food-101-balanced"
NUM_CLASSES = 101
OFFICIAL_TRAIN_PER_CLASS = 750
VALIDATION_PER_CLASS = 100
TRAIN_PER_CLASS = OFFICIAL_TRAIN_PER_CLASS - VALIDATION_PER_CLASS
TEST_PER_CLASS = 250
SPLIT_SEED = 0


def _load_driver():
    spec = spec_from_file_location("tokenizer_dataset_probe_driver_food101", DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the dataset-probe driver from {DRIVER_PATH}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tokenizer_manifest() -> tuple[tuple[str, str], ...]:
    rows = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) != 2:
                raise RuntimeError(
                    f"Expected two tab-separated columns in {MANIFEST_PATH}:{line_number}"
                )
            rows.append((columns[0], columns[1]))
    if len(rows) != 45:
        raise RuntimeError(f"Expected 45 Tokenizer_set_up.md rows, found {len(rows)}")
    if len({setup_id for setup_id, _ in rows}) != len(rows):
        raise RuntimeError(f"Duplicate Tokenizer_set_up.md id in {MANIFEST_PATH}")
    if len({model_id for _, model_id in rows}) != len(rows):
        raise RuntimeError(f"Duplicate probe model id in {MANIFEST_PATH}")
    return tuple(rows)


class IndexedDataset(Dataset):
    """A deterministic indexed view that retains the probe driver's dataset API."""

    def __init__(self, dataset, indices: np.ndarray, *, split: str):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.split = split
        source_targets = np.asarray(dataset.get_targets(), dtype=np.int64)
        self._targets = source_targets[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[int(self.indices[index])]

    def get_targets(self) -> np.ndarray:
        return self._targets


driver = _load_driver()
TOKENIZER_MANIFEST = _load_tokenizer_manifest()
TOKENIZER_MANIFEST_SHA256 = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
MODEL_TO_SETUP = {model_id: setup_id for setup_id, model_id in TOKENIZER_MANIFEST}
MODEL_CHOICES = tuple(model_id for _, model_id in TOKENIZER_MANIFEST)
if not set(MODEL_CHOICES).issubset(driver.base.MODEL_NAMES):
    missing = sorted(set(MODEL_CHOICES) - set(driver.base.MODEL_NAMES))
    raise RuntimeError(f"Tokenizer_set_up.md models unsupported by the probe: {missing}")

# Frozen-feature chunking only: the optimization batch remains 1024. These
# conservative defaults make the mixed-size 45-model panel runnable without
# assuming that every visible GPU can forward 1024 images through a giant ViT.
FEATURE_MICROBATCH_SIZES = {
    "clip_openai__l14": 256,
    "clip_meta__l14": 256,
    "mc1_b32_224_400m": 512,
    "mc1_b16_224_400m": 512,
    "mc1_l14_224_400m": 256,
    "mc1_b32_224_2.5b": 512,
    "mc1_b16_224_2.5b": 512,
    "mc1_l14_224_2.5b": 256,
    "mc1_h14_224_2.5b": 128,
    "mc1_g14_224_2.5b": 64,
    "mc1_h14_224_v1.2": 128,
    "mc2_h14_378": 32,
    "mc2_g14_224": 64,
    "mc2_g14_378": 32,
    "mc2_s16_224": 1024,
    "mc2_s16_384": 512,
    "mc2_s16_224_mt5": 1024,
    "mc2_m16_224": 1024,
    "mc2_m16_384": 512,
    "mc2_m16_224_mt5": 1024,
    "mc2_b32_224": 1024,
    "mc2_b32_384": 512,
    "mc2_b32_224_mt5": 1024,
    "mc2_b16_224": 512,
    "mc2_b16_384": 256,
    "mc2_l14_224": 256,
    "siglip2_b32_256": 1024,
    "siglip2_b16_224": 512,
    "siglip2_b16_256": 512,
    "siglip2_b16_384": 256,
    "siglip2_b16_512": 128,
    "siglip2_l16_256": 256,
    "siglip2_l16_384": 128,
    "siglip2_l16_512": 64,
    "siglip2_sm14_224": 128,
    "siglip2_sm14_384": 64,
    "siglip2_sm16_256": 128,
    "siglip2_sm16_384": 64,
    "siglip2_sm16_512": 32,
    "siglip2_g16_256": 32,
    "siglip2_g16_384": 16,
    "unitok": 256,
    "vilau": 256,
    "toklip_s": 256,
    "toklip_l": 256,
}
if set(FEATURE_MICROBATCH_SIZES) != set(MODEL_CHOICES):
    raise RuntimeError("Feature-microbatch defaults must cover the exact 45-model panel")

_PARTITION_METADATA = None


def _stratified_indices(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SPLIT_SEED)
    train_indices = []
    validation_indices = []
    for class_id in range(NUM_CLASSES):
        class_indices = np.flatnonzero(targets == class_id)
        if len(class_indices) != OFFICIAL_TRAIN_PER_CLASS:
            raise RuntimeError(
                f"Expected {OFFICIAL_TRAIN_PER_CLASS} official train images for Food-101 "
                f"class {class_id}, found {len(class_indices)}"
            )
        shuffled = rng.permutation(class_indices)
        validation_indices.extend(shuffled[:VALIDATION_PER_CLASS])
        train_indices.extend(shuffled[VALIDATION_PER_CLASS:])
    return (
        np.sort(np.asarray(train_indices, dtype=np.int64)),
        np.sort(np.asarray(validation_indices, dtype=np.int64)),
    )


def _indices_sha256(indices: np.ndarray) -> str:
    stable_bytes = np.asarray(indices, dtype="<i8").tobytes(order="C")
    return hashlib.sha256(stable_bytes).hexdigest()


def _build_food101_datasets(args, bundle):
    global _PARTITION_METADATA

    data_root = Path(args.data_root).expanduser().resolve()
    dataset_root = data_root / DATASET_NAME
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing Food-101 dataset directory: {dataset_root}")

    official_train_str = f"HFDataset:name={DATASET_NAME}:split=TRAIN:root={data_root}"
    official_test_str = f"HFDataset:name={DATASET_NAME}:split=VAL:root={data_root}"
    train_source = driver.base.make_dataset(
        dataset_str=official_train_str,
        transform=bundle.train_transform,
    )
    validation_source = driver.base.make_dataset(
        dataset_str=official_train_str,
        transform=bundle.eval_transform,
    )
    test_source = driver.base.make_dataset(
        dataset_str=official_test_str,
        transform=bundle.eval_transform,
    )
    if train_source.split != "train" or validation_source.split != "train":
        raise RuntimeError("Food-101 internal train/validation sources must both resolve train")
    if test_source.split != "validation":
        raise RuntimeError(
            "Food-101 official test must resolve the local Hugging Face validation split"
        )

    train_source_fingerprint = getattr(train_source._dataset, "_fingerprint", None)
    validation_source_fingerprint = getattr(
        validation_source._dataset,
        "_fingerprint",
        None,
    )
    official_test_fingerprint = getattr(test_source._dataset, "_fingerprint", None)
    if not train_source_fingerprint or not official_test_fingerprint:
        raise RuntimeError("Food-101 Hugging Face source fingerprints are unavailable")
    if train_source_fingerprint != validation_source_fingerprint:
        raise RuntimeError("Food-101 train/eval source fingerprints differ")

    train_targets = np.asarray(train_source.get_targets(), dtype=np.int64)
    validation_source_targets = np.asarray(
        validation_source.get_targets(),
        dtype=np.int64,
    )
    test_targets = np.asarray(test_source.get_targets(), dtype=np.int64)
    if not np.array_equal(train_targets, validation_source_targets):
        raise RuntimeError("Food-101 train/eval-transform source target orders differ")
    if len(train_targets) != NUM_CLASSES * OFFICIAL_TRAIN_PER_CLASS:
        raise RuntimeError(f"Unexpected Food-101 official train size: {len(train_targets)}")
    test_counts = np.bincount(test_targets, minlength=NUM_CLASSES)
    if len(test_counts) != NUM_CLASSES or not np.all(test_counts == TEST_PER_CLASS):
        raise RuntimeError("Food-101 official test is not 250 images for each of 101 classes")

    train_indices, validation_indices = _stratified_indices(train_targets)
    if np.intersect1d(train_indices, validation_indices).size:
        raise RuntimeError("Food-101 internal train and validation indices overlap")
    if len(train_indices) + len(validation_indices) != len(train_targets):
        raise RuntimeError("Food-101 internal split does not cover the official train set")

    _PARTITION_METADATA = {
        "split_seed": SPLIT_SEED,
        "split_algorithm": (
            "for class ids 0..100, apply one numpy default_rng(seed) permutation; "
            "take the first 100 indices for validation and the remaining 650 for train; "
            "sort both resulting index arrays"
        ),
        "train_indices_sha256": _indices_sha256(train_indices),
        "validation_indices_sha256": _indices_sha256(validation_indices),
        "official_train_hf_fingerprint": train_source_fingerprint,
        "official_test_hf_fingerprint": official_test_fingerprint,
    }
    test_indices = np.arange(len(test_source), dtype=np.int64)
    return (
        IndexedDataset(train_source, train_indices, split="train"),
        IndexedDataset(validation_source, validation_indices, split="validation"),
        IndexedDataset(test_source, test_indices, split="test"),
    )


def _make_food101_protocol(
    args,
    bundle,
    effective_lrs: list[float],
    *,
    train_size: int,
    val_size: int,
    test_size: int,
) -> dict:
    if _PARTITION_METADATA is None:
        raise RuntimeError("Food-101 partition metadata was not initialized")
    protocol = driver.base._make_protocol(args, bundle, effective_lrs)
    protocol.pop("fingerprint", None)
    samples_per_epoch = driver.base.EPOCH_LENGTH * driver.base.BATCH_SIZE
    protocol.update(
        {
            "version": PROTOCOL_VERSION,
            "dataset": "Food-101",
            "dataset_backend": "local Hugging Face save_to_disk",
            "official_source_splits": {
                "train": "local train",
                "test": "local validation (the Hugging Face name for official test)",
            },
            "official_annotation_note": (
                "Food-101 train contains the benchmark's known noisy labels; official test "
                "annotations were cleaned by the dataset authors"
            ),
            "train_split": "official train minus fixed held-out validation",
            "validation_split": "fixed class-stratified subset of official train",
            "test_split": "official Food-101 test",
            "official_train_samples": NUM_CLASSES * OFFICIAL_TRAIN_PER_CLASS,
            "train_samples": train_size,
            "validation_samples": val_size,
            "test_samples": test_size,
            "train_per_class": TRAIN_PER_CLASS,
            "validation_per_class": VALIDATION_PER_CLASS,
            "test_per_class": TEST_PER_CLASS,
            "class_balance": "exactly equal counts for all 101 classes in every split",
            **_PARTITION_METADATA,
            "samples_drawn_per_epoch": samples_per_epoch,
            "epoch_coverage_ratio": samples_per_epoch / train_size,
            "epoch_semantics": (
                "ceil(train_samples/global_batch_size) consecutive batches from the "
                "shuffled infinite sampler; approximately one full train pass"
            ),
            "validation_head_selection": "best validation micro top-1 over 13 LR heads",
            "test_policy": (
                "evaluate only after training using the head selected on final validation; "
                "never use official test for model or LR selection"
            ),
            "execution_schedule": (
                "global tokenizer barrier after every epoch: all 45 configurations finish "
                "checkpoint plus validation for epoch N before any starts epoch N+1"
            ),
            "execution_cutoff_semantics": (
                "stop_after_epoch controls only one process invocation and is excluded from "
                "the protocol fingerprint; the optimizer and cosine horizon remain fixed at "
                f"the configured {args.epochs} epochs"
            ),
            "feature_microbatch_default_source": (
                "conservative per-model map in "
                "scripts/linear_probe_tokenizers_food101/linear_probe.py"
            ),
            "resume_augmentation_note": (
                "the planned process boundary after every epoch restores head, optimizer, "
                "scheduler, and sampler position, but restarts DataLoader worker augmentation "
                "RNG; this epoch-barrier trajectory intentionally differs from one uninterrupted "
                f"{args.epochs}-epoch process. An unplanned mid-epoch resume is not "
                "bitwise identical"
            ),
            "reported_metrics": {
                "top-1": "micro accuracy; used for LR-head selection",
                "top-5": "micro top-5 accuracy",
                "macro_top-1": "unweighted mean per-class top-1 accuracy",
                "macro_top-5": "unweighted mean per-class top-5 accuracy",
            },
            "tokenizer_manifest": str(MANIFEST_PATH.relative_to(WORKSPACE)),
            "tokenizer_manifest_sha256": TOKENIZER_MANIFEST_SHA256,
            "tokenizer_setup_id": MODEL_TO_SETUP[args.model],
            "tokenizer_setup_config_count": len(TOKENIZER_MANIFEST),
            "tokenizer_setup_independent_model_count": 44,
            "known_duplicate_configuration": {
                "clip_meta__l14": "same checkpoint and feature surface as mc1_l14_224_2.5b"
            },
        }
    )
    protocol["fingerprint"] = driver.base._protocol_fingerprint(protocol)
    return protocol


driver.OUTPUT_ROOT = OUTPUT_ROOT
driver.PROTOCOL_VERSION = PROTOCOL_VERSION
driver.DATASET_NAME = DATASET_NAME
driver.DATASET_DISPLAY_NAME = DATASET_DISPLAY_NAME
driver.NUM_CLASSES = NUM_CLASSES
driver.MODEL_CHOICES = MODEL_CHOICES
driver.SAFE_FEATURE_MICROBATCH_SIZES = FEATURE_MICROBATCH_SIZES
driver.REQUIRE_CHECKPOINT_FINGERPRINT = True
driver.REQUIRE_EPOCH_CUTOFF = True
driver.EXPECTED_SPLIT_SIZES = {
    "train": NUM_CLASSES * TRAIN_PER_CLASS,
    "validation": NUM_CLASSES * VALIDATION_PER_CLASS,
    "test": NUM_CLASSES * TEST_PER_CLASS,
}
driver._build_datasets = _build_food101_datasets
driver._make_protocol = _make_food101_protocol


if __name__ == "__main__":
    sys.exit(driver.main())
