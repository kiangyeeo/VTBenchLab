#!/usr/bin/env python
"""PASCAL VOC 2007 multi-label linear probing for visual tokenizers.

This implements the fixed-feature protocol from Kornblith et al., CVPR 2019:

* freeze the feature extractor and resize the entire image to its native input;
* train 20 independent L2-regularized binary logistic regressions with L-BFGS;
* select one dataset-level regularization value on the official validation set;
* refit on train+val and report official VOC2007 11-point mAP on test.

The 45-point regularization grid, warm-start path, lack of augmentation, and
whole-image preprocessing follow Appendix A.4 of the supplementary material.
VOC ``difficult`` labels (0) are ignored, matching ``VOCevalcls.m``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable, Iterable

import numpy as np
from PIL import Image, ImageFile
from sklearn.linear_model import LogisticRegression
import sklearn
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from linear_probe_tokenizers import (  # noqa: E402
    FEATURE_SURFACES,
    MODEL_NAMES,
    WORKSPACE,
    _environment_metadata,
    _sha256_array,
    _sha256_json,
    checkpoint_manifest,
    load_feature_encoder,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PROTOCOL = "voc2007-kornblith-lbfgs-v1"
DEFAULT_DATA_ROOT = WORKSPACE / "data/voc2007/VOCdevkit/VOC2007"
DEFAULT_OUTPUT_ROOT = WORKSPACE / "outputs/voc2007_multilabel_linear_kornblith_v1"
DEFAULT_BATCH_SIZE = 100
DEFAULT_NUM_WORKERS = 8
DEFAULT_MAX_ITER = 1_000
DEFAULT_TOL = 1e-4
DEFAULT_SEED = 0

VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
SPLIT_SIZES = {"train": 2_501, "val": 2_510, "test": 4_952}

# The paper specifies 45 log-spaced L2 values from 1e-6 to 1e5.  Search from
# strong to weak regularization so every class can use a stable warm start.
LAMBDA_EXPONENTS = tuple(float(value) for value in np.linspace(5.0, -6.0, 45))
LAMBDA_VALUES = tuple(10.0**value for value in LAMBDA_EXPONENTS)

PM1_MEAN = (0.5, 0.5, 0.5)
PM1_STD = (0.5, 0.5, 0.5)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class TransformSpec:
    size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: str = "bicubic"
    antialias: bool = True

    def as_dict(self) -> dict:
        return {
            "mode": "whole-image-resize-no-crop",
            "size": list(self.size),
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": self.interpolation,
            "antialias": self.antialias,
        }


MODEL_TRANSFORM_SPECS = {
    "unitok": TransformSpec((256, 256), PM1_MEAN, PM1_STD),
    "toklips": TransformSpec((256, 256), PM1_MEAN, PM1_STD),
    "toklipl": TransformSpec((384, 384), PM1_MEAN, PM1_STD),
    "vilau": TransformSpec((256, 256), PM1_MEAN, PM1_STD),
    "metaclip": TransformSpec((224, 224), CLIP_MEAN, CLIP_STD),
}


def regularization_grid() -> np.ndarray:
    """Return the paper's 45 lambda values in warm-start search order."""
    return np.asarray(LAMBDA_VALUES, dtype=np.float64)


def lambda_to_c(lambda_value: float) -> float:
    if not np.isfinite(lambda_value) or lambda_value <= 0:
        raise ValueError("lambda must be finite and positive")
    return 1.0 / float(lambda_value)


def select_best_lambda(lambdas: Iterable[float], scores: Iterable[float]) -> int:
    """Return the best candidate index; exact ties prefer stronger L2."""
    lambda_array = np.asarray(list(lambdas), dtype=np.float64)
    score_array = np.asarray(list(scores), dtype=np.float64)
    if lambda_array.ndim != 1 or score_array.shape != lambda_array.shape or len(lambda_array) == 0:
        raise ValueError("lambdas and scores must be non-empty one-dimensional arrays of equal length")
    if not np.isfinite(lambda_array).all() or not np.isfinite(score_array).all():
        raise ValueError("lambdas and scores must be finite")
    # np.lexsort uses the final key as primary: highest score, then largest
    # lambda (strongest regularization) wins.
    return int(np.lexsort((lambda_array, score_array))[-1])


def _read_class_split(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing VOC classification split file: {path}")
    image_ids: list[str] = []
    labels: list[int] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Malformed VOC split line {path}:{line_number}: {line.rstrip()!r}")
            image_id, raw_label = fields
            try:
                label = int(raw_label)
            except ValueError as error:
                raise ValueError(f"Invalid VOC label at {path}:{line_number}: {raw_label!r}") from error
            if label not in (-1, 0, 1):
                raise ValueError(f"VOC labels must be -1, 0, or 1; got {label} at {path}:{line_number}")
            image_ids.append(image_id)
            labels.append(label)
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"Duplicate image IDs in VOC split file: {path}")
    return image_ids, np.asarray(labels, dtype=np.int8)


def load_voc_split(
    data_root: Path,
    split: str,
    *,
    validate_size: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Load the official class-major text labels as an [images, 20] matrix."""
    if split not in SPLIT_SIZES:
        raise ValueError(f"Unsupported VOC split {split!r}; expected one of {sorted(SPLIT_SIZES)}")
    reference_ids: list[str] | None = None
    columns = []
    for class_name in VOC_CLASSES:
        path = data_root / "ImageSets/Main" / f"{class_name}_{split}.txt"
        image_ids, labels = _read_class_split(path)
        if reference_ids is None:
            reference_ids = image_ids
        elif image_ids != reference_ids:
            mismatch = next(
                (
                    index
                    for index, (actual, expected) in enumerate(zip(image_ids, reference_ids))
                    if actual != expected
                ),
                min(len(image_ids), len(reference_ids)),
            )
            raise ValueError(
                f"VOC image order differs for class={class_name} split={split} at index={mismatch}"
            )
        columns.append(labels)
    assert reference_ids is not None
    label_matrix = np.stack(columns, axis=1)
    expected_shape = (len(reference_ids), len(VOC_CLASSES))
    if label_matrix.shape != expected_shape:
        raise RuntimeError(f"Unexpected VOC label shape {label_matrix.shape}; expected {expected_shape}")
    if validate_size and len(reference_ids) != SPLIT_SIZES[split]:
        raise ValueError(
            f"Unexpected VOC2007 {split} size: {len(reference_ids)}; expected {SPLIT_SIZES[split]}"
        )
    return reference_ids, label_matrix


def split_manifest(split: str, image_ids: list[str], labels: np.ndarray) -> dict:
    ids_digest = hashlib.sha256()
    for image_id in image_ids:
        ids_digest.update(image_id.encode("ascii"))
        ids_digest.update(b"\n")
    return {
        "name": split,
        "count": len(image_ids),
        "image_ids_sha256": ids_digest.hexdigest(),
        "labels_sha256": _sha256_array(labels),
        "positive_counts": {
            class_name: int(np.sum(labels[:, index] > 0)) for index, class_name in enumerate(VOC_CLASSES)
        },
        "difficult_counts": {
            class_name: int(np.sum(labels[:, index] == 0)) for index, class_name in enumerate(VOC_CLASSES)
        },
    }


class VOC2007ClassificationDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        transform: Callable | None = None,
        *,
        validate_size: bool = True,
        validate_image_files: bool = True,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.image_ids, self.labels = load_voc_split(self.data_root, split, validate_size=validate_size)
        self.image_paths = [self.data_root / "JPEGImages" / f"{image_id}.jpg" for image_id in self.image_ids]
        if validate_image_files:
            missing = [path for path in self.image_paths if not path.is_file()]
            if missing:
                preview = ", ".join(str(path) for path in missing[:5])
                raise FileNotFoundError(f"Missing {len(missing)} VOC JPEG images; first entries: {preview}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        with Image.open(self.image_paths[index]) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, torch.from_numpy(self.labels[index].copy()), self.image_ids[index]


def make_whole_image_transform(spec: TransformSpec) -> Callable:
    interpolation = {
        "bicubic": InterpolationMode.BICUBIC,
        "bilinear": InterpolationMode.BILINEAR,
        "nearest": InterpolationMode.NEAREST,
    }.get(spec.interpolation)
    if interpolation is None:
        raise ValueError(f"Unsupported interpolation: {spec.interpolation}")
    return transforms.Compose(
        [
            transforms.Resize(spec.size, interpolation=interpolation, antialias=spec.antialias),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ]
    )


def _pair(value) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    values = tuple(int(item) for item in value)
    if len(values) != 2:
        raise ValueError(f"Expected a scalar or pair, got {value!r}")
    return values


def _validate_loader_transform(model_name: str, fallback_transform: Callable, requested: TransformSpec):
    """Guard the static whole-image spec against a changed model-native config."""
    operations = getattr(fallback_transform, "transforms", ())
    crop = next((op for op in operations if isinstance(op, transforms.CenterCrop)), None)
    resize = next((op for op in operations if isinstance(op, transforms.Resize)), None)
    normalize = next((op for op in operations if isinstance(op, transforms.Normalize)), None)
    size_operation = crop if crop is not None else resize
    if size_operation is None:
        raise ValueError(f"{model_name} native transform contains no resize/crop size")
    if _pair(size_operation.size) != requested.size:
        raise ValueError(
            f"{model_name} native input size changed: "
            f"loader={_pair(size_operation.size)} requested={requested.size}"
        )
    if normalize is None:
        raise ValueError(f"{model_name} native transform contains no normalization")
    actual_mean = tuple(float(value) for value in normalize.mean)
    actual_std = tuple(float(value) for value in normalize.std)
    if not np.allclose(actual_mean, requested.mean, rtol=0, atol=1e-7) or not np.allclose(
        actual_std, requested.std, rtol=0, atol=1e-7
    ):
        raise ValueError(
            f"{model_name} native normalization changed: mean={actual_mean} std={actual_std}"
        )


def _feature_cache_complete(
    cache_dir: Path,
    expected_count: int,
    expected_fingerprint: dict,
    expected_image_ids: list[str],
    *,
    overwrite: bool = False,
) -> bool:
    metadata_path = cache_dir / "metadata.json"
    feature_path = cache_dir / "features.npy"
    label_path = cache_dir / "labels.npy"
    image_ids_path = cache_dir / "image_ids.txt"
    if not all(path.is_file() for path in (metadata_path, feature_path, label_path, image_ids_path)):
        return False
    try:
        with metadata_path.open() as handle:
            metadata = json.load(handle)
        actual_ids = image_ids_path.read_text().splitlines()
        feature_array = np.load(feature_path, mmap_mode="r")
        label_array = np.load(label_path, mmap_mode="r")
        valid = (
            metadata.get("count") == expected_count
            and metadata.get("fingerprint") == expected_fingerprint
            and feature_array.ndim == 2
            and feature_array.shape[0] == expected_count
            and label_array.shape == (expected_count, len(VOC_CLASSES))
            and actual_ids == expected_image_ids
        )
    except (json.JSONDecodeError, OSError, ValueError):
        valid = False
    if valid and not overwrite:
        return True
    if overwrite:
        return False
    raise RuntimeError(
        f"Feature cache exists but does not match the requested VOC protocol: {cache_dir}. "
        "Use --overwrite-features to regenerate it explicitly."
    )


@torch.inference_mode()
def extract_features(
    encoder: torch.nn.Module,
    dataset: VOC2007ClassificationDataset,
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    autocast_factory: Callable,
    fingerprint: dict,
    *,
    overwrite: bool = False,
):
    expected_count = len(dataset)
    if _feature_cache_complete(
        cache_dir,
        expected_count,
        fingerprint,
        dataset.image_ids,
        overwrite=overwrite,
    ):
        print(f"Using feature cache: {cache_dir}")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("features.npy", "labels.npy", "image_ids.txt", "metadata.json"):
        stale_path = cache_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    feature_memmap = None
    label_memmap = np.lib.format.open_memmap(
        cache_dir / "labels.npy",
        mode="w+",
        dtype=np.int8,
        shape=(expected_count, len(VOC_CLASSES)),
    )
    written_ids: list[str] = []
    offset = 0
    started = time.time()
    for images, labels, image_ids in tqdm(loader, desc=f"features:{dataset.split}", mininterval=5.0):
        images = images.to(device, non_blocking=True)
        with autocast_factory():
            features = encoder(images)
        if features.ndim != 2:
            raise ValueError(f"Encoder must return [batch, dim], got {tuple(features.shape)}")
        feature_array = features.float().cpu().numpy()
        if not np.isfinite(feature_array).all():
            raise ValueError(f"{dataset.split} feature extractor produced NaN or Inf")
        if feature_memmap is None:
            feature_memmap = np.lib.format.open_memmap(
                cache_dir / "features.npy",
                mode="w+",
                dtype=np.float32,
                shape=(expected_count, feature_array.shape[1]),
            )
        end = offset + len(feature_array)
        feature_memmap[offset:end] = feature_array
        label_memmap[offset:end] = labels.numpy().astype(np.int8, copy=False)
        written_ids.extend(image_ids)
        offset = end

    if feature_memmap is None or offset != expected_count or written_ids != dataset.image_ids:
        raise RuntimeError(
            f"Incomplete {dataset.split} feature cache: wrote={offset} expected={expected_count} "
            f"ids_match={written_ids == dataset.image_ids}"
        )
    feature_memmap.flush()
    label_memmap.flush()
    (cache_dir / "image_ids.txt").write_text("\n".join(written_ids) + "\n")
    metadata = {
        "count": expected_count,
        "feature_dim": int(feature_memmap.shape[1]),
        "feature_dtype": "float32",
        "label_dtype": "int8",
        "elapsed_seconds": time.time() - started,
        "fingerprint": fingerprint,
        "fingerprint_sha256": _sha256_json(fingerprint),
    }
    temporary_path = cache_dir / "metadata.json.tmp"
    with temporary_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
    os.replace(temporary_path, cache_dir / "metadata.json")


def voc2007_11_point_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute the VOC2007 interpolated AP for one class as a fraction."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("labels and scores must be equal-length one-dimensional arrays")
    if not np.isin(labels, (-1, 0, 1)).all():
        raise ValueError("VOC labels must contain only -1, 0, and 1")
    if not np.isfinite(scores).all():
        raise ValueError("VOC prediction scores must be finite")
    valid = labels != 0
    valid_labels = labels[valid]
    valid_scores = scores[valid]
    positive_count = int(np.sum(valid_labels > 0))
    if positive_count == 0:
        raise ValueError("VOC AP is undefined for a class without positive examples")
    order = np.argsort(-valid_scores, kind="mergesort")
    true_positive = np.cumsum(valid_labels[order] > 0)
    false_positive = np.cumsum(valid_labels[order] < 0)
    recall = true_positive / positive_count
    precision = true_positive / (true_positive + false_positive)
    average_precision = 0.0
    for threshold in np.linspace(0.0, 1.0, 11):
        eligible = precision[recall >= threshold]
        average_precision += (float(np.max(eligible)) if len(eligible) else 0.0) / 11.0
    return average_precision


def evaluate_multilabel(labels: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    expected_shape = (labels.shape[0], len(VOC_CLASSES)) if labels.ndim == 2 else None
    if labels.shape != expected_shape or scores.shape != labels.shape:
        raise ValueError(
            f"Expected labels and scores shaped [N,{len(VOC_CLASSES)}], got {labels.shape} and {scores.shape}"
        )
    per_class = np.asarray(
        [voc2007_11_point_ap(labels[:, index], scores[:, index]) for index in range(len(VOC_CLASSES))],
        dtype=np.float64,
    )
    return float(np.mean(per_class)), per_class


def _class_search_cache_complete(
    path: Path,
    fingerprint_sha256: str,
    val_count: int,
    *,
    overwrite: bool,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            valid = (
                str(payload["fingerprint_sha256"].item()) == fingerprint_sha256
                and str(payload["class_name"].item()) == path.stem
                and payload["lambdas"].shape == (len(LAMBDA_VALUES),)
                and np.allclose(payload["lambdas"], regularization_grid(), rtol=1e-12, atol=0)
                and payload["val_scores"].shape == (len(LAMBDA_VALUES), val_count)
                and np.isfinite(payload["val_scores"]).all()
                and payload["n_iter"].shape == (len(LAMBDA_VALUES),)
                and payload["converged"].shape == (len(LAMBDA_VALUES),)
                and payload["elapsed_seconds"].shape == (len(LAMBDA_VALUES),)
            )
    except (KeyError, OSError, ValueError):
        valid = False
    if valid and not overwrite:
        return True
    if overwrite:
        return False
    raise RuntimeError(
        f"Search cache exists but its protocol does not match: {path}. "
        "Use --overwrite-probe to replace it explicitly."
    )


def _fit_class_regularization_path(
    args,
    class_index: int,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    cache_path: Path,
    fingerprint_sha256: str,
):
    class_name = VOC_CLASSES[class_index]
    if _class_search_cache_complete(
        cache_path,
        fingerprint_sha256,
        len(val_features),
        overwrite=args.overwrite_probe,
    ):
        print(f"Using completed regularization path: model={args.model} class={class_name}")
        return

    valid = train_labels[:, class_index] != 0
    labels = (train_labels[valid, class_index] > 0).astype(np.int64)
    features = np.asarray(train_features[valid])
    if not np.array_equal(np.unique(labels), np.asarray([0, 1])):
        raise ValueError(f"Training split for {class_name} must contain positive and negative examples")

    classifier = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        fit_intercept=True,
        class_weight=None,
        warm_start=True,
        random_state=args.seed,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=args.logreg_verbose,
    )
    lambdas = regularization_grid()
    validation_scores = np.empty((len(lambdas), len(val_features)), dtype=np.float64)
    iterations = np.empty(len(lambdas), dtype=np.int32)
    converged = np.empty(len(lambdas), dtype=np.bool_)
    elapsed = np.empty(len(lambdas), dtype=np.float64)
    for candidate_index, lambda_value in enumerate(lambdas):
        started = time.time()
        classifier.set_params(C=lambda_to_c(float(lambda_value)))
        classifier.fit(features, labels)
        validation_scores[candidate_index] = classifier.decision_function(val_features)
        iterations[candidate_index] = int(np.max(classifier.n_iter_))
        converged[candidate_index] = iterations[candidate_index] < args.max_iter
        elapsed[candidate_index] = time.time() - started
        print(
            f"Path model={args.model} class={class_name} candidate={candidate_index + 1}/{len(lambdas)} "
            f"lambda={lambda_value:.12g} C={lambda_to_c(float(lambda_value)):.12g} "
            f"n_iter={iterations[candidate_index]} converged={bool(converged[candidate_index])}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez_compressed(
        temporary_path,
        fingerprint_sha256=np.asarray(fingerprint_sha256),
        class_name=np.asarray(class_name),
        lambdas=lambdas,
        val_scores=validation_scores,
        n_iter=iterations,
        converged=converged,
        elapsed_seconds=elapsed,
    )
    os.replace(temporary_path, cache_path)


def _load_search_paths(
    search_dir: Path,
    fingerprint_sha256: str,
    val_labels: np.ndarray,
) -> tuple[list[dict], int]:
    lambdas = regularization_grid()
    all_scores = np.empty((len(lambdas), len(val_labels), len(VOC_CLASSES)), dtype=np.float64)
    all_iterations = np.empty((len(lambdas), len(VOC_CLASSES)), dtype=np.int32)
    all_converged = np.empty((len(lambdas), len(VOC_CLASSES)), dtype=np.bool_)
    all_elapsed = np.empty((len(lambdas), len(VOC_CLASSES)), dtype=np.float64)
    for class_index, class_name in enumerate(VOC_CLASSES):
        path = search_dir / f"{class_name}.npz"
        if not _class_search_cache_complete(path, fingerprint_sha256, len(val_labels), overwrite=False):
            raise FileNotFoundError(f"Missing completed search path: {path}")
        with np.load(path, allow_pickle=False) as payload:
            all_scores[:, :, class_index] = payload["val_scores"]
            all_iterations[:, class_index] = payload["n_iter"]
            all_converged[:, class_index] = payload["converged"]
            all_elapsed[:, class_index] = payload["elapsed_seconds"]

    history = []
    validation_maps = []
    for candidate_index, lambda_value in enumerate(lambdas):
        mean_ap, per_class = evaluate_multilabel(val_labels, all_scores[candidate_index])
        validation_maps.append(100.0 * mean_ap)
        history.append(
            {
                "index": candidate_index,
                "lambda": float(lambda_value),
                "C": lambda_to_c(float(lambda_value)),
                "validation_mAP_11point": 100.0 * mean_ap,
                "validation_AP_11point": {
                    class_name: 100.0 * float(per_class[index])
                    for index, class_name in enumerate(VOC_CLASSES)
                },
                "n_iter_max": int(np.max(all_iterations[candidate_index])),
                "nonconverged_classes": [
                    class_name
                    for index, class_name in enumerate(VOC_CLASSES)
                    if not all_converged[candidate_index, index]
                ],
                "elapsed_seconds": float(np.sum(all_elapsed[candidate_index])),
            }
        )
    selected_index = select_best_lambda(lambdas, validation_maps)
    return history, selected_index


def _fit_final_classifiers(
    args,
    selected_lambda: float,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trainval_features = np.concatenate((np.asarray(train_features), np.asarray(val_features)), axis=0)
    trainval_labels = np.concatenate((np.asarray(train_labels), np.asarray(val_labels)), axis=0)
    feature_dim = trainval_features.shape[1]
    test_scores = np.empty((len(test_features), len(VOC_CLASSES)), dtype=np.float64)
    coefficients = np.empty((len(VOC_CLASSES), feature_dim), dtype=np.float64)
    intercepts = np.empty(len(VOC_CLASSES), dtype=np.float64)
    iterations = np.empty(len(VOC_CLASSES), dtype=np.int32)
    for class_index, class_name in enumerate(VOC_CLASSES):
        valid = trainval_labels[:, class_index] != 0
        labels = (trainval_labels[valid, class_index] > 0).astype(np.int64)
        if not np.array_equal(np.unique(labels), np.asarray([0, 1])):
            raise ValueError(f"Combined train+val split for {class_name} lacks a binary class")
        classifier = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            fit_intercept=True,
            class_weight=None,
            warm_start=False,
            random_state=args.seed,
            C=lambda_to_c(selected_lambda),
            max_iter=args.max_iter,
            tol=args.tol,
            verbose=args.logreg_verbose,
        )
        classifier.fit(trainval_features[valid], labels)
        test_scores[:, class_index] = classifier.decision_function(test_features)
        coefficients[class_index] = classifier.coef_[0]
        intercepts[class_index] = classifier.intercept_[0]
        iterations[class_index] = int(np.max(classifier.n_iter_))
        print(
            f"Final model={args.model} class={class_name} lambda={selected_lambda:.12g} "
            f"n_iter={iterations[class_index]} converged={iterations[class_index] < args.max_iter}"
        )
    return test_scores, coefficients, intercepts, iterations


def _write_voc_result_files(result_dir: Path, image_ids: list[str], scores: np.ndarray):
    result_dir.mkdir(parents=True, exist_ok=True)
    for class_index, class_name in enumerate(VOC_CLASSES):
        destination = result_dir / f"comp1_cls_test_{class_name}.txt"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w") as handle:
            for image_id, score in zip(image_ids, scores[:, class_index]):
                handle.write(f"{image_id} {score:.12g}\n")
        os.replace(temporary, destination)


def _load_cached_arrays(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    features = np.load(cache_dir / "features.npy", mmap_mode="r")
    labels = np.load(cache_dir / "labels.npy", mmap_mode="r")
    image_ids = (cache_dir / "image_ids.txt").read_text().splitlines()
    with (cache_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    return features, labels, image_ids, metadata


def run_probe(
    args,
    model_root: Path,
    protocol_config: dict,
    split_manifests: dict[str, dict],
    checkpoint_metadata: dict,
):
    result_path = model_root / "results.json"
    protocol_sha256 = _sha256_json(protocol_config)
    if result_path.is_file() and not args.overwrite_probe:
        with result_path.open() as handle:
            existing = json.load(handle)
        if existing.get("protocol_config_sha256") != protocol_sha256:
            raise RuntimeError(
                f"Existing result uses a different protocol: {result_path}. "
                "Use --overwrite-probe to replace it explicitly."
            )
        print(f"Using completed VOC2007 result: {result_path}")
        return

    train_features, train_labels, train_ids, train_cache_metadata = _load_cached_arrays(
        model_root / "features_train"
    )
    val_features, val_labels, val_ids, val_cache_metadata = _load_cached_arrays(model_root / "features_val")
    test_features, test_labels, test_ids, test_cache_metadata = _load_cached_arrays(model_root / "features_test")
    if train_features.shape[1] != val_features.shape[1] or train_features.shape[1] != test_features.shape[1]:
        raise ValueError("VOC feature dimensions differ across train/val/test caches")
    if len(train_ids) != SPLIT_SIZES["train"] or len(val_ids) != SPLIT_SIZES["val"] or len(test_ids) != SPLIT_SIZES["test"]:
        raise ValueError("Cached VOC image ID counts do not match official split sizes")

    search_fingerprint = {
        "protocol_config_sha256": protocol_sha256,
        "train_feature_fingerprint_sha256": train_cache_metadata["fingerprint_sha256"],
        "val_feature_fingerprint_sha256": val_cache_metadata["fingerprint_sha256"],
        "lambdas": list(LAMBDA_VALUES),
        "max_iter": args.max_iter,
        "tol": args.tol,
        "seed": args.seed,
        "scikit_learn": sklearn.__version__,
    }
    search_fingerprint_sha256 = _sha256_json(search_fingerprint)
    search_dir = model_root / "search"
    for class_index in range(len(VOC_CLASSES)):
        _fit_class_regularization_path(
            args,
            class_index,
            train_features,
            train_labels,
            val_features,
            search_dir / f"{VOC_CLASSES[class_index]}.npz",
            search_fingerprint_sha256,
        )

    search_history, selected_index = _load_search_paths(
        search_dir,
        search_fingerprint_sha256,
        np.asarray(val_labels),
    )
    selected_item = search_history[selected_index]
    selected_lambda = float(selected_item["lambda"])
    print(
        f"Selected model={args.model} lambda={selected_lambda:.12g} C={lambda_to_c(selected_lambda):.12g} "
        f"validation_mAP={selected_item['validation_mAP_11point']:.6f}"
    )

    final_started = time.time()
    test_scores, coefficients, intercepts, final_iterations = _fit_final_classifiers(
        args,
        selected_lambda,
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
    )
    test_mean_ap, test_per_class = evaluate_multilabel(np.asarray(test_labels), test_scores)
    final_elapsed = time.time() - final_started

    head_path = model_root / "linear_head.npz"
    temporary_head_path = head_path.with_name(head_path.name + ".tmp.npz")
    np.savez_compressed(
        temporary_head_path,
        protocol_config_sha256=np.asarray(protocol_sha256),
        class_names=np.asarray(VOC_CLASSES),
        coefficients=coefficients,
        intercepts=intercepts,
        selected_lambda=np.asarray(selected_lambda),
        selected_C=np.asarray(lambda_to_c(selected_lambda)),
        n_iter=final_iterations,
    )
    os.replace(temporary_head_path, head_path)
    _write_voc_result_files(model_root / "voc_results", test_ids, test_scores)

    final_converged = final_iterations < args.max_iter
    result = {
        "protocol": PROTOCOL,
        "protocol_description": (
            "Kornblith et al. fixed whole-image features, 45-point L2 L-BFGS selection, "
            "VOC2007 official 11-point mAP"
        ),
        "protocol_references": {
            "main_paper": (
                "https://openaccess.thecvf.com/content_CVPR_2019/papers/"
                "Kornblith_Do_Better_ImageNet_Models_Transfer_Better_CVPR_2019_paper.pdf"
            ),
            "supplementary_material": (
                "https://openaccess.thecvf.com/content_CVPR_2019/supplemental/"
                "Kornblith_Do_Better_ImageNet_CVPR_2019_supplemental.pdf"
            ),
            "voc2007_devkit": "https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/htmldoc/",
            "openai_clip_linear_probe": "https://github.com/openai/CLIP#linear-probe-evaluation",
            "local_feature_loader": "CLIP/linear_probe_tokenizers.py",
            "local_dinov2_reference": "dinov2/dinov2/eval/linear.py",
        },
        "protocol_config": protocol_config,
        "protocol_config_sha256": protocol_sha256,
        "model": args.model,
        "dataset": "PASCAL VOC 2007 classification",
        "class_names": list(VOC_CLASSES),
        "splits": split_manifests,
        "feature_dim": int(train_features.shape[1]),
        "feature_surface": FEATURE_SURFACES[args.model],
        "feature_normalized": False,
        "preprocessing": MODEL_TRANSFORM_SPECS[args.model].as_dict(),
        "feature_extraction_batch_size": args.batch_size,
        "checkpoint_manifest": checkpoint_metadata,
        "feature_caches": {
            "train": train_cache_metadata,
            "val": val_cache_metadata,
            "test": test_cache_metadata,
        },
        "classifier": "20 independent sklearn.linear_model.LogisticRegression(solver='lbfgs')",
        "classifier_config": {
            "penalty": "l2",
            "fit_intercept": True,
            "class_weight": None,
            "warm_start_during_search": True,
            "random_state": args.seed,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "difficult_label_policy": "ignore label 0 independently for each class",
            "regularization_parameter": "lambda",
            "sklearn_mapping": "C = 1 / lambda",
            "lambda_search_order": "strong-to-weak",
            "lambda_values": list(LAMBDA_VALUES),
            "selection_metric": "validation 11-point mAP",
            "tie_break": "largest lambda (strongest regularization)",
        },
        "regularization_search": search_history,
        "selection": {
            "selected_index": selected_index,
            "selected_lambda": selected_lambda,
            "selected_C": lambda_to_c(selected_lambda),
            "validation_mAP_11point": selected_item["validation_mAP_11point"],
            "validation_AP_11point": selected_item["validation_AP_11point"],
            "nonconverged_classes": selected_item["nonconverged_classes"],
        },
        "final_evaluation": {
            "fit_split": "train+val",
            "fit_images": len(train_ids) + len(val_ids),
            "evaluation_split": "test",
            "evaluation_images": len(test_ids),
            "mAP_11point": 100.0 * test_mean_ap,
            "AP_11point": {
                class_name: 100.0 * float(test_per_class[index])
                for index, class_name in enumerate(VOC_CLASSES)
            },
            "n_iter": {
                class_name: int(final_iterations[index]) for index, class_name in enumerate(VOC_CLASSES)
            },
            "converged": bool(np.all(final_converged)),
            "nonconverged_classes": [
                class_name
                for index, class_name in enumerate(VOC_CLASSES)
                if not final_converged[index]
            ],
            "elapsed_seconds": final_elapsed,
        },
        "artifacts": {
            "linear_head": str(head_path),
            "official_voc_result_dir": str(model_root / "voc_results"),
        },
        "environment": _environment_metadata(),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_result_path = result_path.with_suffix(".json.tmp")
    with temporary_result_path.open("w") as handle:
        json.dump(result, handle, indent=2)
    os.replace(temporary_result_path, result_path)
    print(json.dumps(result["final_evaluation"], indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="GPU feature-extraction batch size; the L-BFGS classifier has no minibatch size.",
    )
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--logreg-verbose", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--overwrite-probe", action="store_true")
    parser.add_argument("--overwrite-features", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_iter < 1:
        parser.error("--batch-size and --max-iter must be positive")
    if args.num_workers < 0 or args.logreg_verbose < 0:
        parser.error("--num-workers and --logreg-verbose must be nonnegative")
    if args.tol <= 0 or not np.isfinite(args.tol):
        parser.error("--tol must be finite and positive")
    if args.smoke_test and args.probe_only:
        parser.error("--smoke-test and --probe-only are mutually exclusive")
    if args.probe_only and args.overwrite_features:
        parser.error("--probe-only cannot be combined with --overwrite-features")
    return args


def _make_feature_fingerprint(
    args,
    split: str,
    manifest: dict,
    checkpoint_metadata: dict,
) -> dict:
    return {
        "protocol": PROTOCOL,
        "model": args.model,
        "feature_surface": FEATURE_SURFACES[args.model],
        "feature_normalized": False,
        "transform": MODEL_TRANSFORM_SPECS[args.model].as_dict(),
        "checkpoint_manifest_sha256": checkpoint_metadata["sha256"],
        "split": split,
        "split_manifest": manifest,
        "extraction_batch_size": args.batch_size,
        "environment": _environment_metadata(),
    }


def main(argv=None):
    args = parse_args(argv)
    if not args.data_root.is_dir():
        raise FileNotFoundError(
            f"VOC2007 root does not exist: {args.data_root}. Extract the official trainval/test archives first."
        )

    base_datasets = {
        split: VOC2007ClassificationDataset(args.data_root, split, transform=None)
        for split in ("train", "val", "test")
    }
    split_manifests = {
        split: split_manifest(split, dataset.image_ids, dataset.labels)
        for split, dataset in base_datasets.items()
    }
    split_id_sets = {split: set(dataset.image_ids) for split, dataset in base_datasets.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_id_sets[left].intersection(split_id_sets[right])
        if overlap:
            raise ValueError(
                f"VOC2007 {left}/{right} splits overlap by {len(overlap)} image IDs; "
                f"first={sorted(overlap)[0]}"
            )
    checkpoint_metadata = checkpoint_manifest(args.model)
    model_root = args.output_root / args.model
    cache_dirs = {split: model_root / f"features_{split}" for split in base_datasets}
    fingerprints = {
        split: _make_feature_fingerprint(args, split, split_manifests[split], checkpoint_metadata)
        for split in base_datasets
    }

    if args.probe_only:
        for split, dataset in base_datasets.items():
            if not _feature_cache_complete(
                cache_dirs[split],
                len(dataset),
                fingerprints[split],
                dataset.image_ids,
            ):
                raise FileNotFoundError(f"Required {split} feature cache is missing: {cache_dirs[split]}")
    else:
        cache_states = {
            split: _feature_cache_complete(
                cache_dirs[split],
                len(dataset),
                fingerprints[split],
                dataset.image_ids,
                overwrite=args.overwrite_features,
            )
            for split, dataset in base_datasets.items()
        }
        if args.smoke_test or not all(cache_states.values()):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for tokenizer feature extraction")
            device = torch.device("cuda")
            encoder, fallback_transform, autocast_factory = load_feature_encoder(args.model, device)
            transform_spec = MODEL_TRANSFORM_SPECS[args.model]
            _validate_loader_transform(args.model, fallback_transform, transform_spec)
            transform = make_whole_image_transform(transform_spec)
            if args.smoke_test:
                smoke_dataset = VOC2007ClassificationDataset(args.data_root, "test", transform=transform)
                images = torch.stack([smoke_dataset[index][0] for index in range(2)]).to(device)
                with torch.inference_mode(), autocast_factory():
                    features = encoder(images)
                print(
                    f"Smoke test {args.model}: images={tuple(images.shape)} features={tuple(features.shape)} "
                    f"finite={bool(torch.isfinite(features).all())}"
                )
                return
            for split, cache_complete in cache_states.items():
                if cache_complete and not args.overwrite_features:
                    continue
                dataset = VOC2007ClassificationDataset(args.data_root, split, transform=transform)
                extract_features(
                    encoder,
                    dataset,
                    cache_dirs[split],
                    args.batch_size,
                    args.num_workers,
                    device,
                    autocast_factory,
                    fingerprints[split],
                    overwrite=args.overwrite_features,
                )
            del encoder
            gc.collect()
            torch.cuda.empty_cache()

    protocol_config = {
        "protocol": PROTOCOL,
        "model": args.model,
        "class_names": list(VOC_CLASSES),
        "split_manifests": split_manifests,
        "checkpoint_manifest_sha256": checkpoint_metadata["sha256"],
        "feature_surface": FEATURE_SURFACES[args.model],
        "feature_normalized": False,
        "transform": MODEL_TRANSFORM_SPECS[args.model].as_dict(),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "lambda_values": list(LAMBDA_VALUES),
        "lambda_search_order": "strong-to-weak",
        "sklearn_C_mapping": "1/lambda",
        "selection_metric": "validation 11-point mAP",
        "tie_break": "strongest regularization",
        "final_fit_split": "train+val",
        "final_evaluation_split": "test",
        "environment": _environment_metadata(),
    }
    run_probe(args, model_root, protocol_config, split_manifests, checkpoint_metadata)


if __name__ == "__main__":
    main()
