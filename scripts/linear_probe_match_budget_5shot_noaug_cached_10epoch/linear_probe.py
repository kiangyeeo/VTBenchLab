#!/usr/bin/env python
"""Deterministic cached ImageNet-1K 5-shot probing for the 70-model panel."""

from __future__ import annotations

import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
from torch.utils.data import Dataset


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
CACHED_PATH = WORKSPACE / "scripts/linear_probe_tokenizers_cached/linear_probe_cached.py"
FIVE_SHOT_PATH = WORKSPACE / "scripts/linear_probe_match_budget_5shot/linear_probe.py"
MANIFEST_PATH = SCRIPT_DIR / "tokenizers.tsv"
OUTPUT_ROOT = WORKSPACE / "outputs/vae_linear_probing_match_budget_5shot_noaug_cached_10epoch"
CACHE_ROOT = OUTPUT_ROOT / "_feature_cache"

PROTOCOL_VERSION = "tokenizer_linear_probe_imagenet_5shot_noaug_cached_10epoch_allbn_v1"
NUM_CLASSES = 1_000
SHOTS_PER_CLASS = 5
TRAIN_SAMPLES = NUM_CLASSES * SHOTS_PER_CLASS
GLOBAL_BATCH_SIZE = 1_000
EPOCHS = 10
EPOCH_LENGTH = TRAIN_SAMPLES // GLOBAL_BATCH_SIZE
MAX_UPDATES = EPOCHS * EPOCH_LENGTH


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_manifest() -> tuple[tuple[int, str, str, str], ...]:
    rows = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) != 4:
                raise RuntimeError(
                    f"Expected four TSV columns at {MANIFEST_PATH}:{line_number}"
                )
            rank, label, model, head = columns
            rows.append((int(rank), label, model, head))
    if len(rows) != 70 or [row[0] for row in rows] != list(range(1, 71)):
        raise RuntimeError("Tokenizer manifest must contain ranks 1..70 exactly once")
    models = [row[2] for row in rows]
    if len(set(models)) != 70:
        raise RuntimeError("Tokenizer manifest contains duplicate model ids")
    return tuple(rows)


cached = _load_module("tokenizer_noaug_cached_base", CACHED_PATH)
base = cached.base
five_shot = _load_module("tokenizer_five_shot_reference", FIVE_SHOT_PATH)

MANIFEST = _load_manifest()
MODEL_CHOICES = tuple(row[2] for row in MANIFEST)
MODEL_METADATA = {
    model: {"rank": rank, "vision_encoder": label, "head": head}
    for rank, label, model, head in MANIFEST
}
FEATURE_MICROBATCH_SIZES = dict(five_shot.driver.FEATURE_MICROBATCH_SIZES)
if set(FEATURE_MICROBATCH_SIZES) != set(MODEL_CHOICES):
    raise RuntimeError("Feature-microbatch defaults must cover all 70 models")
unsupported = sorted(set(MODEL_CHOICES) - set(base.MODEL_NAMES))
if unsupported:
    raise RuntimeError(f"Cached probing base does not support: {unsupported}")

expected_bn = set(MODEL_CHOICES)
selected_bn = {model for _rank, _label, model, head in MANIFEST if head == "bn"}
if selected_bn != expected_bn:
    raise RuntimeError(
        f"BN routing mismatch: selected={sorted(selected_bn)}, expected={sorted(expected_bn)}"
    )

# Keep this run isolated and give every non-BN entry its ordinary output name.
for _model in base.WEBSSL_DINO_SPECS:
    base.OUTPUT_NAMES[_model] = _model

base.BATCH_SIZE = GLOBAL_BATCH_SIZE
base.EPOCHS = EPOCHS
base.EPOCH_LENGTH = EPOCH_LENGTH
base.MAX_UPDATES = MAX_UPDATES
base.EVAL_PERIOD_UPDATES = EPOCH_LENGTH
cached.PROTOCOL_VERSION = PROTOCOL_VERSION
cached.DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT
cached.DEFAULT_CACHE_ROOT = CACHE_ROOT


class SupportDataset(Dataset):
    """Indexed support view retaining the ImageNet ``get_targets`` interface."""

    def __init__(self, dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        targets = np.asarray(dataset.get_targets(), dtype=np.int64)
        self._targets = targets[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[int(self.indices[index])]

    def get_targets(self) -> np.ndarray:
        return self._targets


def make_balanced_support_indices(targets: np.ndarray, seed: int) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64)
    expected_classes = np.arange(NUM_CLASSES, dtype=np.int64)
    if not np.array_equal(np.unique(targets), expected_classes):
        raise RuntimeError("Expected ImageNet labels 0..999")
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in expected_classes:
        candidates = np.flatnonzero(targets == class_id)
        if len(candidates) < SHOTS_PER_CLASS:
            raise RuntimeError(f"Class {class_id} has fewer than five examples")
        selected.extend(rng.choice(candidates, SHOTS_PER_CLASS, replace=False))
    indices = np.sort(np.asarray(selected, dtype=np.int64))
    counts = np.bincount(targets[indices], minlength=NUM_CLASSES)
    if len(indices) != TRAIN_SAMPLES or not np.all(counts == SHOTS_PER_CLASS):
        raise RuntimeError("Constructed support set is not exactly class-balanced")
    return indices


_original_build_parser = cached._build_parser
_original_make_dataset = base.make_dataset
_original_cache_identity = cached._cache_identity
_original_make_protocol = cached._make_protocol
_plain_linear_head = base.LinearHead
_active_args = None
_support_metadata = None


def _build_parser():
    parser = _original_build_parser()
    parser.allow_abbrev = False
    parser.description = __doc__
    model_action = next(action for action in parser._actions if action.dest == "model")
    model_action.choices = MODEL_CHOICES
    parser.set_defaults(
        output_root=str(OUTPUT_ROOT),
        cache_root=str(CACHE_ROOT),
        feature_microbatch_size=None,
        cache_batch_size=None,
        stop_after_epoch=EPOCHS,
    )
    parser.add_argument("--seed", type=int, default=0, help="Training/head seed.")
    parser.add_argument(
        "--support-seed",
        type=int,
        default=0,
        help="Seed for the shared class-balanced five-shot support set.",
    )
    return parser


def _parse_args():
    global _active_args
    args = _build_parser().parse_args()
    if args.seed < 0 or args.support_seed < 0:
        raise ValueError("--seed and --support-seed must be non-negative")
    if args.stop_after_epoch != EPOCHS:
        raise ValueError("This protocol is fixed to exactly 10 cached-head epochs")
    if args.feature_microbatch_size is None:
        args.feature_microbatch_size = FEATURE_MICROBATCH_SIZES[args.model]
    if args.cache_batch_size is None:
        args.cache_batch_size = min(256, args.feature_microbatch_size)
    base.SEED = int(args.seed)
    _active_args = args
    return args


def _make_dataset(*, dataset_str: str, transform):
    global _support_metadata
    dataset = _original_make_dataset(dataset_str=dataset_str, transform=transform)
    if "split=TRAIN" not in dataset_str:
        return dataset
    if _active_args is None:
        raise RuntimeError("Arguments must be parsed before constructing the dataset")
    targets = np.asarray(dataset.get_targets(), dtype=np.int64)
    indices = make_balanced_support_indices(targets, _active_args.support_seed)
    stable_indices = np.asarray(indices, dtype="<i8")
    _support_metadata = {
        "official_train_samples": int(len(dataset)),
        "support_samples": int(len(indices)),
        "support_seed": int(_active_args.support_seed),
        "support_indices_sha256": hashlib.sha256(
            stable_indices.tobytes(order="C")
        ).hexdigest(),
    }
    return SupportDataset(dataset, indices)


def _uses_bn(model: str) -> bool:
    return MODEL_METADATA[model]["head"] == "bn"


def _linear_head_class(model: str):
    return cached.BatchNormalizedLinearHead if _uses_bn(model) else _plain_linear_head


def _cache_identity(args, bundle, data_root: Path, extra_root: Path) -> dict:
    if _support_metadata is None:
        raise RuntimeError("Support metadata is unavailable")
    identity = _original_cache_identity(args, bundle, data_root, extra_root)
    identity.update(
        {
            "version": "imagenet_5shot_deterministic_frozen_features_v1",
            "train_split": "fixed class-balanced five-shot support subset",
            "support_seed": _support_metadata["support_seed"],
            "support_indices_sha256": _support_metadata["support_indices_sha256"],
            "support_samples": TRAIN_SAMPLES,
        }
    )
    return identity


def _make_protocol(args, bundle, effective_lrs, cache_fingerprint: str) -> dict:
    if _support_metadata is None:
        raise RuntimeError("Support metadata is unavailable")
    protocol = _original_make_protocol(args, bundle, effective_lrs, cache_fingerprint)
    protocol.pop("fingerprint", None)
    metadata = MODEL_METADATA[args.model]
    use_bn = _uses_bn(args.model)
    protocol.update(
        {
            "version": PROTOCOL_VERSION,
            "model_rank": metadata["rank"],
            "vision_encoder": metadata["vision_encoder"],
            "train_split": "fixed class-balanced subset of official train",
            "train_sampling": "exactly 5 distinct images per class without replacement",
            "shots_per_class": SHOTS_PER_CLASS,
            "train_samples": TRAIN_SAMPLES,
            "samples_drawn_per_epoch": TRAIN_SAMPLES,
            "support_seed": args.support_seed,
            "support_indices_sha256": _support_metadata["support_indices_sha256"],
            "official_train_samples": _support_metadata["official_train_samples"],
            "validation_split": "official ImageNet-1k validation (50,000 images)",
            "validation_samples": 50_000,
            "validation_feature_extraction_microbatch_size": (
                args.feature_microbatch_size
            ),
            "cache_loader_batch_size": args.cache_batch_size,
            "epochs": EPOCHS,
            "epoch_length_updates": EPOCH_LENGTH,
            "max_updates": MAX_UPDATES,
            "eval_period_updates": EPOCH_LENGTH,
            "training_feature_presentations": TRAIN_SAMPLES * EPOCHS,
            "budget_scope": (
                "5,000 frozen-tokenizer training forwards cached once; the cached "
                "features are presented to the probe head for 10 epochs; official "
                "validation encoder forwards are reported separately"
            ),
            "probe_head": "BatchNorm1d(affine=False)->Linear" if use_bn else "Linear",
            "probe_head_source": (
                "scripts/linear_probe_tokenizers_cached/linear_probe_cached.py"
                if use_bn
                else "scripts/linear_probe_tokenizers/linear_probe.py"
            ),
            "feature_normalization": use_bn,
            "training_cutoff_semantics": "fixed 10-epoch/50-update cosine schedule",
        }
    )
    if not use_bn:
        for key in tuple(protocol):
            if key.startswith("batch_norm_") or key in {
                "feature_normalization_type",
                "feature_normalization_placement",
                "pixio_readout_protocol",
            }:
                protocol.pop(key, None)
    protocol["fingerprint"] = base._protocol_fingerprint(protocol)
    return protocol


cached._build_parser = _build_parser
cached._parse_args = _parse_args
cached._linear_head_class = _linear_head_class
cached._cache_identity = _cache_identity
cached._make_protocol = _make_protocol
base.make_dataset = _make_dataset


def main() -> int:
    return cached.main()


if __name__ == "__main__":
    sys.exit(main())
