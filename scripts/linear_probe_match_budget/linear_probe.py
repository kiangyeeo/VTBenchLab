#!/usr/bin/env python
"""One-epoch, class-balanced ImageNet-1K 4-shot tokenizer probing.

This is a budget-matched adapter around ``scripts/linear_probe_tokenizers``.
The 4,000-image support set is shared by all 42 tokenizers. Pixio and Web-SSL
MAE use the exact non-affine BatchNorm head implementation from
``scripts/linear_probe_tokenizers_bn``; every other model uses the no-BN head.
"""

from __future__ import annotations

import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
from torch.utils.data import Dataset


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SCRIPT_DIR / "tokenizers.tsv"
BASE_DIR = WORKSPACE / "scripts/linear_probe_tokenizers"
BASE_PATH = BASE_DIR / "linear_probe.py"
BN_PATH = WORKSPACE / "scripts/linear_probe_tokenizers_bn/linear_probe.py"
OUTPUT_ROOT = WORKSPACE / "outputs/vae_linear_probing_match_budget_4shot"

PROTOCOL_VERSION = "tokenizer_linear_probe_imagenet_4shot_1epoch_v1"
NUM_CLASSES = 1_000
SHOTS_PER_CLASS = 4
TRAIN_SAMPLES = NUM_CLASSES * SHOTS_PER_CLASS
GLOBAL_BATCH_SIZE = 1_000
EPOCHS = 1
EPOCH_LENGTH = TRAIN_SAMPLES // GLOBAL_BATCH_SIZE

if TRAIN_SAMPLES % GLOBAL_BATCH_SIZE:
    raise RuntimeError("The fixed 4-shot support set must divide the global batch size")

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_module("tokenizer_match_budget_base", BASE_PATH)
bn_source = _load_module("tokenizer_match_budget_bn_source", BN_PATH)


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
    if len(rows) != 42:
        raise RuntimeError(f"Expected 42 tokenizer rows, found {len(rows)}")
    if [rank for rank, *_rest in rows] != list(range(1, 43)):
        raise RuntimeError("Tokenizer ranks must be exactly 1..42")
    models = [model for _rank, _label, model, _head in rows]
    if len(set(models)) != len(models):
        raise RuntimeError("Tokenizer manifest contains duplicate model ids")
    unsupported = sorted(set(models) - set(base.MODEL_NAMES))
    if unsupported:
        raise RuntimeError(f"Base probing code does not support: {unsupported}")
    if any(head not in {"linear", "bn"} for *_prefix, head in rows):
        raise RuntimeError("Tokenizer head must be 'linear' or 'bn'")
    expected_bn = {*base.PIXIO_SPECS, *base.WEBSSL_MAE_SPECS}
    selected_bn = {model for _rank, _label, model, head in rows if head == "bn"}
    expected_bn &= set(models)
    if selected_bn != expected_bn:
        raise RuntimeError(
            f"BN routing mismatch: selected={sorted(selected_bn)}, "
            f"expected={sorted(expected_bn)}"
        )
    return tuple(rows)


MANIFEST = _load_manifest()
MODEL_CHOICES = tuple(model for _rank, _label, model, _head in MANIFEST)
MODEL_METADATA = {
    model: {"rank": rank, "vision_encoder": label, "head": head}
    for rank, label, model, head in MANIFEST
}

# Conservative frozen-encoder chunk sizes. They affect memory and throughput,
# but not the 1,000-example optimization batch or the FLOPs budget.
FEATURE_MICROBATCH_SIZES = {
    "siglip2_sm14_384": 64,
    "siglip2_g16_384": 16,
    "siglip2_l16_384": 128,
    "siglip2_sm16_512": 32,
    "siglip2_sm16_384": 64,
    "siglip2_g16_256": 32,
    "mc2_g14_378": 32,
    "siglip2_sm14_224": 128,
    "pe_core_g14_448": 16,
    "siglip2_sm16_256": 128,
    "siglip2_l16_256": 256,
    "mc1_g14_224_2.5b": 64,
    "mc2_g14_224": 64,
    "siglip2_b16_512": 128,
    "mc1_h14_224_v1.2": 128,
    "clip_openai__l14": 256,
    "mc1_h14_224_2.5b": 128,
    "mc1_l14_224_2.5b": 256,
    "siglip2_b16_256": 512,
    "siglip2_b16_224": 512,
    "mc2_l14_224": 256,
    "pe_lang_l14_448": 32,
    "pe_core_b16_224": 512,
    "mc1_b16_224_400m": 512,
    "mc1_b16_224_2.5b": 512,
    "pixio_vitl16": 128,
    "dinov2_giant": 64,
    "dinov2_large": 256,
    "pixio_vitb16": 256,
    "eupe_vit_b": 512,
    "eupe_vit_s": 1000,
    "mc2_s16_224": 1000,
    "dinov2_base": 512,
    "dinov2_small": 1000,
    "pixio_vith16": 64,
    "ijepa": 128,
    "dinov1_vitb16": 512,
    "webssl_mae3b_full2b_224": 16,
    "eupe_vit_t": 1000,
    "dinov1_vits8": 512,
    "dinov1_vitb8": 256,
    "dinov1_vits16": 1000,
}
if set(FEATURE_MICROBATCH_SIZES) != set(MODEL_CHOICES):
    raise RuntimeError("Feature-microbatch defaults must cover exactly the 42 models")


class IndexedDataset(Dataset):
    """An indexed dataset view retaining DINOv2's ``get_targets`` API."""

    def __init__(self, dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        source_targets = np.asarray(dataset.get_targets(), dtype=np.int64)
        self._targets = source_targets[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[int(self.indices[index])]

    def get_targets(self) -> np.ndarray:
        return self._targets


def make_balanced_support_indices(
    targets: np.ndarray,
    *,
    shots_per_class: int = SHOTS_PER_CLASS,
    seed: int = 0,
) -> np.ndarray:
    """Choose exactly K distinct examples from every ImageNet class."""
    targets = np.asarray(targets, dtype=np.int64)
    classes = np.unique(targets)
    expected_classes = np.arange(NUM_CLASSES, dtype=np.int64)
    if not np.array_equal(classes, expected_classes):
        raise RuntimeError(
            f"Expected ImageNet labels 0..999, found {len(classes)} classes"
        )
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in expected_classes:
        candidates = np.flatnonzero(targets == class_id)
        if len(candidates) < shots_per_class:
            raise RuntimeError(
                f"Class {class_id} has only {len(candidates)} candidates for "
                f"{shots_per_class}-shot sampling"
            )
        selected.extend(rng.choice(candidates, shots_per_class, replace=False))
    indices = np.sort(np.asarray(selected, dtype=np.int64))
    selected_targets = targets[indices]
    counts = np.bincount(selected_targets, minlength=NUM_CLASSES)
    if len(indices) != NUM_CLASSES * shots_per_class or not np.all(
        counts == shots_per_class
    ):
        raise RuntimeError("Constructed support set is not exactly class-balanced")
    return indices


_original_build_parser = base._build_parser
_original_make_dataset = base.make_dataset
_original_make_protocol = base._make_protocol
_original_build_heads = base._build_heads
_active_args = None
_support_metadata = None


def _build_parser():
    parser = _original_build_parser()
    parser.allow_abbrev = False
    parser.description = __doc__
    model_action = next(action for action in parser._actions if action.dest == "model")
    model_action.choices = MODEL_CHOICES
    feature_action = next(
        action for action in parser._actions if action.dest == "feature_microbatch_size"
    )
    feature_action.help = (
        "Frozen-backbone forward chunk size; this changes memory/throughput only. "
        "The optimization batch remains 1,000."
    )
    cutoff_action = next(
        action for action in parser._actions if action.dest == "stop_after_epoch"
    )
    cutoff_action.help = "Fixed to 1 by this one-epoch budget-matched protocol."
    parser.set_defaults(
        output_root=str(OUTPUT_ROOT),
        feature_microbatch_size=None,
        stop_after_epoch=1,
    )
    parser.add_argument("--seed", type=int, default=0, help="Training/head seed.")
    parser.add_argument(
        "--support-seed",
        type=int,
        default=0,
        help="Seed for the shared class-balanced 4-shot support subset.",
    )
    return parser


def _parse_args():
    global _active_args
    args = _build_parser().parse_args()
    if args.seed < 0 or args.support_seed < 0:
        raise ValueError("--seed and --support-seed must be non-negative")
    if args.stop_after_epoch != 1:
        raise ValueError("This fixed-budget protocol supports exactly one epoch")
    if args.feature_microbatch_size is None:
        args.feature_microbatch_size = FEATURE_MICROBATCH_SIZES[args.model]
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
    indices = make_balanced_support_indices(
        targets,
        shots_per_class=SHOTS_PER_CLASS,
        seed=_active_args.support_seed,
    )
    stable_indices = np.asarray(indices, dtype="<i8")
    _support_metadata = {
        "official_train_samples": int(len(dataset)),
        "support_samples": int(len(indices)),
        "support_seed": int(_active_args.support_seed),
        "support_indices_sha256": hashlib.sha256(
            stable_indices.tobytes(order="C")
        ).hexdigest(),
    }
    return IndexedDataset(dataset, indices)


def _uses_bn(model: str) -> bool:
    return MODEL_METADATA[model]["head"] == "bn"


def _build_heads(in_dim, device, *, use_batch_norm=False):
    if _active_args is None:
        raise RuntimeError("Arguments must be parsed before building heads")
    return _original_build_heads(
        in_dim,
        device,
        use_batch_norm=_uses_bn(_active_args.model),
    )


def _make_protocol(args, bundle, effective_lrs):
    if _support_metadata is None:
        raise RuntimeError("Support metadata is unavailable")
    protocol = _original_make_protocol(args, bundle, effective_lrs)
    protocol.pop("fingerprint", None)
    metadata = MODEL_METADATA[args.model]
    use_bn = _uses_bn(args.model)
    protocol.update(
        {
            "version": PROTOCOL_VERSION,
            "model_rank": metadata["rank"],
            "vision_encoder": metadata["vision_encoder"],
            "dataset": "ImageNet-1k",
            "train_split": "fixed class-balanced subset of official train",
            "train_sampling": "exactly 4 distinct images per class without replacement",
            "shots_per_class": SHOTS_PER_CLASS,
            "train_samples": TRAIN_SAMPLES,
            "samples_drawn_per_epoch": TRAIN_SAMPLES,
            "support_seed": args.support_seed,
            "support_indices_sha256": _support_metadata["support_indices_sha256"],
            "official_train_samples": _support_metadata["official_train_samples"],
            "validation_split": "official ImageNet-1k validation (50,000 images)",
            "validation_samples": 50_000,
            "epochs": EPOCHS,
            "epoch_length_updates": EPOCH_LENGTH,
            "max_updates": EPOCH_LENGTH,
            "eval_period_updates": EPOCH_LENGTH,
            "budget_scope": (
                "4,000 frozen-tokenizer training forwards; official validation "
                "forwards are reported separately"
            ),
            "probe_head": "BatchNorm1d(affine=False)->Linear" if use_bn else "Linear",
            "probe_head_source": (
                "scripts/linear_probe_tokenizers_bn/linear_probe.py"
                if use_bn
                else "scripts/linear_probe_tokenizers/linear_probe.py"
            ),
            "feature_normalization": use_bn,
        }
    )
    if use_bn:
        protocol.update(
            {
                "feature_normalization_type": "BatchNorm1d",
                "feature_normalization_placement": (
                    "immediately before each linear classifier"
                ),
                "batch_norm_affine": False,
                "batch_norm_fixed_scale": 1.0,
                "batch_norm_fixed_shift": 0.0,
                "batch_norm_eps": 1e-6,
                "batch_norm_momentum": 0.1,
                "batch_norm_track_running_stats": True,
                "batch_norm_training_batch_size": GLOBAL_BATCH_SIZE,
                "batch_norm_applied_after_feature_microbatch_concatenation": True,
            }
        )
    protocol["fingerprint"] = base._protocol_fingerprint(protocol)
    return protocol


# Fix the optimization surface before the baseline main function constructs its
# parser, scheduler, loaders, heads, and protocol.
base.PROTOCOL_VERSION = PROTOCOL_VERSION
base.BATCH_SIZE = GLOBAL_BATCH_SIZE
base.EPOCHS = EPOCHS
base.EPOCH_LENGTH = EPOCH_LENGTH
base.MAX_UPDATES = EPOCH_LENGTH
base.EVAL_PERIOD_UPDATES = EPOCH_LENGTH
base.BatchNormalizedLinearHead = bn_source.BatchNormalizedLinearHead
base._build_parser = _build_parser
base._parse_args = _parse_args
base.make_dataset = _make_dataset
base._build_heads = _build_heads
base._make_protocol = _make_protocol


def main() -> int:
    return base.main()


if __name__ == "__main__":
    sys.exit(main())
