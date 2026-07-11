#!/usr/bin/env python
"""Reproducible ImageNet-1K few-shot linear probing for tokenizer features.

The default ``clip-paper-v1`` protocol follows the linear-probe description in
Appendix A.3 of the CLIP paper and fills in its unpublished ImageNet split with
the deterministic 10% training split used by DINOv2:

* freeze the feature extractor and use deterministic model-native evaluation
  preprocessing;
* reserve a fixed random 10% of ImageNet train for regularization selection;
* draw exactly k examples per class from the disjoint 90% support pool;
* fit an L-BFGS logistic-regression head for at most 1,000 iterations;
* select C with CLIP's parametric search on the held-out selection split; and
* report the selected support-only classifier on the official 50k validation
  split.  The unpublished-label 100k ILSVRC test split is not used.

The legacy ``clip-readme-fixed`` protocol keeps the old fixed-C behavior for
comparison.  Feature caches and result files are resumable only when their
complete protocol fingerprints match.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Callable, Iterable

import numpy as np
from PIL import ImageFile
from sklearn.linear_model import LogisticRegression
import sklearn
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
IMAGE_SCRIPTS = WORKSPACE / "TokBench/tokenzier_vae_scripts/image_scripts"
MODEL_ZOO = WORKSPACE / "TokBench/tokenizer_modelzoo"

DEFAULT_DATA_ROOT = WORKSPACE / "data/imagenet1k"
DEFAULT_OUTPUT_ROOT = WORKSPACE / "outputs/imagenet_kshot_linear_clip_paper_v1"
DEFAULT_SHOTS = (1, 2, 4, 8, 16)
DEFAULT_C = 0.316
DEFAULT_SELECTION_FRACTION = 0.1
DEFAULT_SELECTION_SEED = 0
DEFAULT_C_EXPONENTS = (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0)
DEFAULT_C_RESOLUTION = 0.125
NUM_CLASSES = 1_000
IMAGENET_TRAIN_SIZE = 1_281_167
IMAGENET_VAL_SIZE = 50_000
PROTOCOLS = ("clip-paper-v1", "clip-readme-fixed")

MODEL_NAMES = ("unitok", "toklips", "toklipl", "vilau", "metaclip")
FEATURE_SURFACES = {
    "unitok": "mean-pooled quantized tokens after fc_norm, before UniTok projection",
    "toklips": "TokLIP-S semantic visual representation before the output head",
    "toklipl": "TokLIP-L semantic visual representation before the output head",
    "vilau": "mean-pooled penultimate SigLIP tokens used as the VILA-U tokenizer input",
    "metaclip": "final MetaCLIP class token after norm, before the 768-to-512 projection",
}
TRANSFORM_DESCRIPTIONS = {
    "unitok": "Resize(256,bicubic)+CenterCrop(256)+RGB+Normalize(0.5,0.5)",
    "toklips": "Resize(256,bicubic)+CenterCrop(256)+RGB+Normalize(0.5,0.5)",
    "toklipl": "Resize(384,bicubic)+CenterCrop(384)+RGB+Normalize(0.5,0.5)",
    "vilau": "model-native-size bicubic resize+center-crop+RGB+Normalize(0.5,0.5)",
    "metaclip": "timm model-native deterministic evaluation transform",
}


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _environment_metadata() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "timm": timm.__version__,
    }


def _dataset_order_sha256(dataset: datasets.ImageFolder) -> str:
    digest = hashlib.sha256()
    root = Path(dataset.root)
    for sample_path, label in dataset.samples:
        relative = Path(sample_path).relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(label).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _checkpoint_paths(model_name: str) -> list[Path]:
    if model_name == "unitok":
        return [MODEL_ZOO / "unitok_20250227/unitok_tokenizer.pth"]
    if model_name == "toklips":
        return [MODEL_ZOO / "TokLIP/TokLIP_S_256.pt", MODEL_ZOO / "TokLIP/vq_ds16_t2i.pt"]
    if model_name == "toklipl":
        return [MODEL_ZOO / "TokLIP/TokLIP_L_384.pt", MODEL_ZOO / "TokLIP/vq_ds16_t2i.pt"]
    if model_name == "vilau":
        return [MODEL_ZOO / "VILA-U/vila-u-7b-256", MODEL_ZOO / "VILA-U/siglip-large-patch16-256"]
    if model_name == "metaclip":
        return [MODEL_ZOO / "MetaCLIP/vit_base_patch16_clip_224.metaclip_2pt5b/model.safetensors"]
    raise ValueError(f"Unsupported model: {model_name}")


def checkpoint_manifest(model_name: str) -> dict:
    """Return a cheap, stable path/size manifest and its SHA256 fingerprint."""
    entries = []
    for configured_path in _checkpoint_paths(model_name):
        if not configured_path.exists():
            raise FileNotFoundError(f"Missing checkpoint path for {model_name}: {configured_path}")
        if configured_path.is_file():
            entries.append(
                {
                    "configured_path": str(configured_path),
                    "relative_path": configured_path.name,
                    "size": configured_path.stat().st_size,
                }
            )
            continue
        for file_path in sorted(path for path in configured_path.rglob("*") if path.is_file()):
            entries.append(
                {
                    "configured_path": str(configured_path),
                    "relative_path": file_path.relative_to(configured_path).as_posix(),
                    "size": file_path.stat().st_size,
                }
            )
    return {
        "kind": "sha256(path,size manifest)",
        "sha256": _sha256_json(entries),
        "entries": entries,
    }


def _pm1_eval_transform(size: int) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class UniTokFeatureEncoder(nn.Module):
    """Quantized UniTok representation immediately before its CLIP projection."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.model.encoder(images).float()
        tokens = self.model.quant_proj(tokens)
        indices = self.model.quantizer.f_to_idx(tokens)
        tokens = self.model.quantizer.idx_to_f(indices)
        tokens = self.model.post_quant_proj(tokens)
        return self.model.fc_norm(tokens.mean(dim=1)).float()


class TokLIPFeatureEncoder(nn.Module):
    """TokLIP semantic representation immediately before its output head."""

    def __init__(self, trunk: nn.Module, encode_tokens: Callable):
        super().__init__()
        self.trunk = trunk
        self.encode_tokens = encode_tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_tokens(self.trunk, images)
        return self.trunk.forward_head(tokens, pre_logits=True).float()


def _encode_vilau_penultimate(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    vision_model = model.siglip_model.vision_model
    hidden_states = vision_model.embeddings(images)
    target_idx = len(vision_model.encoder.layers) - 2
    for index, encoder_layer in enumerate(vision_model.encoder.layers):
        layer_outputs = encoder_layer(hidden_states, None, output_attentions=None)
        hidden_states = layer_outputs[0]
        if index == target_idx:
            return hidden_states.mean(dim=1).float()
    raise RuntimeError("Failed to extract the VILA-U penultimate visual tokens")


class VilaUFeatureEncoder(nn.Module):
    """Mean-pooled VILA-U tokenizer input at the penultimate SigLIP layer."""

    def __init__(self, model: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return _encode_vilau_penultimate(self.model, images.to(self.dtype))


class MetaCLIPFeatureEncoder(nn.Module):
    """MetaCLIP penultimate feature, before its 768 -> 512 projection."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(images)
        return self.model.forward_head(tokens, pre_logits=True).float()


def _load_unitok(device: torch.device):
    sys.path.insert(0, str(IMAGE_SCRIPTS))
    from unitok_vae_rec import load_unitok

    model, _ = load_unitok(
        SimpleNamespace(
            unitok_path=str(IMAGE_SCRIPTS / "UniTok"),
            ckpt_path=str(MODEL_ZOO / "unitok_20250227/unitok_tokenizer.pth"),
        )
    )
    encoder = UniTokFeatureEncoder(model).to(device).eval().requires_grad_(False)
    return encoder, _pm1_eval_transform(256), nullcontext


def _load_toklip(name: str, device: torch.device):
    sys.path.insert(0, str(IMAGE_SCRIPTS))
    from toklip_rec_common import encode_toklip_semantic_tokens, load_toklip_semantic_model

    variant = "s" if name == "toklips" else "l"
    model_name = f"toklip_{variant}"
    size = 256 if variant == "s" else 384
    checkpoint = MODEL_ZOO / "TokLIP" / ("TokLIP_S_256.pt" if variant == "s" else "TokLIP_L_384.pt")
    trunk = load_toklip_semantic_model(
        SimpleNamespace(
            toklip_path=str(IMAGE_SCRIPTS / "TokLIP"),
            toklip_ckpt_path=str(checkpoint),
            vq_ckpt_path=str(MODEL_ZOO / "TokLIP/vq_ds16_t2i.pt"),
            model_name=model_name,
            toklip_model_config=None,
        ),
        device=str(device),
    )
    encoder = TokLIPFeatureEncoder(trunk, encode_toklip_semantic_tokens)
    encoder = encoder.to(device).eval().requires_grad_(False)
    return encoder, _pm1_eval_transform(size), nullcontext


def _load_vilau(device: torch.device):
    sys.path.insert(0, str(IMAGE_SCRIPTS))
    from vilau_rec import load_vilau_tokenizer

    tokenizer = load_vilau_tokenizer(
        SimpleNamespace(
            vilau_path=str(IMAGE_SCRIPTS / "vila-u"),
            model_path=str(MODEL_ZOO / "VILA-U/vila-u-7b-256"),
            siglip_config_path=str(MODEL_ZOO / "VILA-U/siglip-large-patch16-256"),
            dtype="bfloat16",
        ),
        str(device),
    )
    encoder = VilaUFeatureEncoder(tokenizer.model, tokenizer.dtype)
    encoder = encoder.to(device).eval().requires_grad_(False)
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return encoder, _pm1_eval_transform(tokenizer.image_size), autocast


def _load_metaclip(device: torch.device):
    from timm.data import create_transform, resolve_model_data_config
    from timm.models import load_checkpoint

    checkpoint = MODEL_ZOO / "MetaCLIP/vit_base_patch16_clip_224.metaclip_2pt5b/model.safetensors"
    model = timm.create_model("vit_base_patch16_clip_224.metaclip_2pt5b", pretrained=False)
    load_checkpoint(model, str(checkpoint), strict=True)
    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    encoder = MetaCLIPFeatureEncoder(model).to(device).eval().requires_grad_(False)
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
    return encoder, transform, autocast


def load_feature_encoder(name: str, device: torch.device):
    if name == "unitok":
        return _load_unitok(device)
    if name in ("toklips", "toklipl"):
        return _load_toklip(name, device)
    if name == "vilau":
        return _load_vilau(device)
    if name == "metaclip":
        return _load_metaclip(device)
    raise ValueError(f"Unsupported model: {name}")


def make_train_selection_indices(num_samples: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if num_samples < 2:
        raise ValueError("num_samples must be at least two")
    if not 0.0 < fraction < 1.0:
        raise ValueError("selection fraction must be between zero and one")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator).numpy().astype(np.int64, copy=False)
    selection_count = int(num_samples * fraction)
    if selection_count < 1 or selection_count >= num_samples:
        raise ValueError("selection fraction produced an empty split")
    return permutation[:selection_count], permutation[selection_count:]


def load_or_create_train_selection(
    dataset: datasets.ImageFolder,
    split_path: Path,
    fraction: float,
    seed: int,
    dataset_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    expected_count = int(len(dataset) * fraction)
    if split_path.exists():
        with np.load(split_path, allow_pickle=False) as payload:
            selection = payload["selection_indices"].astype(np.int64, copy=False)
            support_pool = payload["support_pool_indices"].astype(np.int64, copy=False)
            stored_dataset_hash = str(payload["dataset_order_sha256"].item())
            stored_seed = int(payload["selection_seed"].item())
            stored_fraction = float(payload["selection_fraction"].item())
        if stored_dataset_hash != dataset_sha256 or stored_seed != seed or stored_fraction != fraction:
            raise ValueError(f"Existing selection split metadata does not match requested protocol: {split_path}")
    else:
        selection, support_pool = make_train_selection_indices(len(dataset), fraction, seed)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            split_path,
            selection_indices=selection,
            support_pool_indices=support_pool,
            selection_seed=np.asarray(seed),
            selection_fraction=np.asarray(fraction),
            dataset_order_sha256=np.asarray(dataset_sha256),
        )

    if len(selection) != expected_count or len(support_pool) != len(dataset) - expected_count:
        raise ValueError(f"Unexpected selection/support-pool sizes in {split_path}")
    all_indices = np.concatenate((selection, support_pool))
    if len(np.unique(all_indices)) != len(dataset) or all_indices.min() != 0 or all_indices.max() != len(dataset) - 1:
        raise ValueError(f"Selection and support pool do not form a partition: {split_path}")
    metadata = {
        "path": str(split_path),
        "selection_seed": seed,
        "selection_fraction": fraction,
        "selection_count": len(selection),
        "support_pool_count": len(support_pool),
        "selection_indices_sha256": _sha256_array(selection),
        "support_pool_indices_sha256": _sha256_array(support_pool),
        "dataset_order_sha256": dataset_sha256,
    }
    return selection, support_pool, metadata


def make_nested_support_indices(
    targets: np.ndarray,
    max_shot: int,
    seed: int,
    pool_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return deterministic [num_classes, max_shot] dataset indices."""
    targets = np.asarray(targets, dtype=np.int64)
    classes = np.unique(targets)
    if len(classes) != NUM_CLASSES or not np.array_equal(classes, np.arange(NUM_CLASSES)):
        raise ValueError(f"Expected ImageNet labels 0..999, found {len(classes)} classes")
    if pool_indices is None:
        pool_indices = np.arange(len(targets), dtype=np.int64)
    else:
        pool_indices = np.asarray(pool_indices, dtype=np.int64)
    pool_targets = targets[pool_indices]

    rng = np.random.default_rng(seed)
    support = np.empty((NUM_CLASSES, max_shot), dtype=np.int64)
    for class_index in range(NUM_CLASSES):
        candidates = pool_indices[pool_targets == class_index]
        if len(candidates) < max_shot:
            raise ValueError(f"Class {class_index} has only {len(candidates)} samples in support pool")
        support[class_index] = rng.choice(candidates, size=max_shot, replace=False)
    return support


def load_or_create_support(
    dataset: datasets.ImageFolder,
    split_path: Path,
    pool_indices: np.ndarray,
    max_shot: int,
    seed: int,
    dataset_sha256: str,
    pool_sha256: str,
) -> tuple[np.ndarray, dict]:
    if split_path.exists():
        with np.load(split_path, allow_pickle=False) as payload:
            support = payload["support_indices"].astype(np.int64, copy=False)
            stored_dataset_hash = str(payload["dataset_order_sha256"].item())
            stored_pool_hash = str(payload["support_pool_indices_sha256"].item())
            stored_seed = int(payload["support_seed"].item())
            stored_max_shot = int(payload["max_shot"].item())
        if (
            stored_dataset_hash != dataset_sha256
            or stored_pool_hash != pool_sha256
            or stored_seed != seed
            or stored_max_shot != max_shot
        ):
            raise ValueError(f"Existing support split metadata does not match requested protocol: {split_path}")
    else:
        support = make_nested_support_indices(np.asarray(dataset.targets), max_shot, seed, pool_indices)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            split_path,
            support_indices=support,
            support_seed=np.asarray(seed),
            max_shot=np.asarray(max_shot),
            support_pool_indices_sha256=np.asarray(pool_sha256),
            dataset_order_sha256=np.asarray(dataset_sha256),
        )

    if support.shape != (NUM_CLASSES, max_shot):
        raise ValueError(f"Existing split has unexpected shape {support.shape}: {split_path}")
    if not np.all(np.isin(support, pool_indices)):
        raise ValueError(f"Support split contains indices outside support pool: {split_path}")
    if len(np.unique(support)) != support.size:
        raise ValueError(f"Support split contains duplicate images: {split_path}")
    expected_labels = np.repeat(np.arange(NUM_CLASSES)[:, None], max_shot, axis=1)
    if not np.array_equal(np.asarray(dataset.targets)[support], expected_labels):
        raise ValueError(f"Support split is not class balanced: {split_path}")
    metadata = {
        "path": str(split_path),
        "support_seed": seed,
        "max_shot": max_shot,
        "support_indices_sha256": _sha256_array(support),
        "support_pool_indices_sha256": pool_sha256,
        "dataset_order_sha256": dataset_sha256,
    }
    return support, metadata


def _feature_cache_complete(
    cache_dir: Path,
    expected_count: int,
    expected_fingerprint: dict,
    overwrite: bool = False,
) -> bool:
    metadata_path = cache_dir / "metadata.json"
    feature_path = cache_dir / "features.npy"
    label_path = cache_dir / "labels.npy"
    paths = (metadata_path, feature_path, label_path)
    if not all(path.is_file() for path in paths):
        return False
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    actual_fingerprint = metadata.get("fingerprint")
    valid = (
        metadata.get("count") == expected_count
        and actual_fingerprint == expected_fingerprint
        and np.load(feature_path, mmap_mode="r").shape[0] == expected_count
        and np.load(label_path, mmap_mode="r").shape[0] == expected_count
    )
    if valid:
        return True
    if overwrite:
        return False
    raise RuntimeError(
        f"Feature cache exists but its protocol fingerprint does not match: {cache_dir}. "
        "Use --overwrite-features to regenerate it explicitly."
    )


@torch.inference_mode()
def extract_features(
    encoder: nn.Module,
    dataset,
    indices: Iterable[int] | None,
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    autocast_factory: Callable,
    fingerprint: dict,
    overwrite: bool = False,
):
    indices = None if indices is None else np.asarray(list(indices), dtype=np.int64)
    selected_dataset = dataset if indices is None else Subset(dataset, indices.tolist())
    expected_count = len(selected_dataset)
    if _feature_cache_complete(cache_dir, expected_count, fingerprint, overwrite=overwrite):
        print(f"Using feature cache: {cache_dir}")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    for stale in (cache_dir / "features.npy", cache_dir / "labels.npy", cache_dir / "metadata.json"):
        if stale.exists():
            stale.unlink()

    loader = DataLoader(
        selected_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    feature_memmap = None
    label_memmap = np.lib.format.open_memmap(cache_dir / "labels.npy", mode="w+", dtype=np.int64, shape=(expected_count,))
    offset = 0
    started = time.time()
    for images, labels in tqdm(loader, desc=f"features:{cache_dir.name}", mininterval=5.0):
        images = images.to(device, non_blocking=True)
        with autocast_factory():
            features = encoder(images)
        if features.ndim != 2:
            raise ValueError(f"Encoder must return [batch, dim], got {tuple(features.shape)}")
        features = features.float().cpu().numpy()
        if not np.isfinite(features).all():
            raise ValueError("Feature extractor produced NaN or Inf")
        if feature_memmap is None:
            feature_memmap = np.lib.format.open_memmap(
                cache_dir / "features.npy",
                mode="w+",
                dtype=np.float32,
                shape=(expected_count, features.shape[1]),
            )
        end = offset + len(features)
        feature_memmap[offset:end] = features
        label_memmap[offset:end] = labels.numpy()
        offset = end

    if offset != expected_count or feature_memmap is None:
        raise RuntimeError(f"Wrote {offset} features, expected {expected_count}")
    feature_memmap.flush()
    label_memmap.flush()
    metadata = {
        "count": expected_count,
        "feature_dim": int(feature_memmap.shape[1]),
        "dtype": "float32",
        "elapsed_seconds": time.time() - started,
        "fingerprint": fingerprint,
        "fingerprint_sha256": _sha256_json(fingerprint),
    }
    with (cache_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)


def _topk_correct(logits: np.ndarray, targets: np.ndarray, k: int) -> int:
    if k == 1:
        return int(np.sum(np.argmax(logits, axis=1) == targets))
    topk = np.argpartition(logits, -k, axis=1)[:, -k:]
    return int(np.sum(np.any(topk == targets[:, None], axis=1)))


def evaluate_classifier(
    classifier: LogisticRegression,
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 2_048,
    compute_top5: bool = True,
) -> tuple[float, float | None]:
    top1 = 0
    top5 = 0
    count = len(targets)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        logits = classifier.decision_function(features[start:end])
        top1 += _topk_correct(logits, targets[start:end], 1)
        if compute_top5:
            top5 += _topk_correct(logits, targets[start:end], 5)
    top1_percent = 100.0 * top1 / count
    top5_percent = 100.0 * top5 / count if compute_top5 else None
    return top1_percent, top5_percent


def parametric_c_search(
    evaluate_exponent: Callable[[float], float],
    initial_exponents: Iterable[float] = DEFAULT_C_EXPONENTS,
    resolution: float = DEFAULT_C_RESOLUTION,
) -> tuple[float, list[dict]]:
    """CLIP-style local binary search in log10(C), with deterministic ties."""
    initial = sorted(set(float(value) for value in initial_exponents))
    if len(initial) < 2 or resolution <= 0:
        raise ValueError("C search needs at least two initial exponents and positive resolution")
    lower_bound, upper_bound = initial[0], initial[-1]
    scores: dict[float, float] = {}
    history: list[dict] = []

    def evaluate(exponent: float):
        exponent = round(float(exponent), 12)
        if exponent in scores:
            return
        score = float(evaluate_exponent(exponent))
        if not np.isfinite(score):
            raise ValueError(f"Non-finite selection score for log10(C)={exponent}")
        scores[exponent] = score
        history.append({"log10_C": exponent, "C": 10.0**exponent, "selection_top1": score})

    for exponent in initial:
        evaluate(exponent)

    while True:
        # Maximize score; for exact ties choose smaller C (stronger sklearn L2).
        best_exponent = min(scores, key=lambda exponent: (-scores[exponent], exponent))
        ordered = sorted(scores)
        best_index = ordered.index(best_exponent)
        candidates = []
        if best_index > 0 and best_exponent - ordered[best_index - 1] > resolution + 1e-12:
            candidates.append((best_exponent + ordered[best_index - 1]) / 2.0)
        if best_index + 1 < len(ordered) and ordered[best_index + 1] - best_exponent > resolution + 1e-12:
            candidates.append((best_exponent + ordered[best_index + 1]) / 2.0)
        # A boundary peak is refined only toward the interior of the published range.
        candidates = [value for value in candidates if lower_bound <= value <= upper_bound]
        if not candidates:
            break
        for exponent in candidates:
            evaluate(exponent)

    best_exponent = min(scores, key=lambda exponent: (-scores[exponent], exponent))
    return 10.0**best_exponent, history


def _fit_logistic_regression(args, features: np.ndarray, labels: np.ndarray, c_value: float) -> LogisticRegression:
    classifier = LogisticRegression(
        random_state=args.seed,
        C=c_value,
        max_iter=args.max_iter,
        verbose=args.logreg_verbose,
        solver="lbfgs",
        tol=args.tol,
    )
    classifier.fit(features, labels)
    return classifier


def _load_completed_results(output_path: Path, protocol_config_sha256: str, overwrite_probe: bool) -> dict[int, dict]:
    if not output_path.exists() or overwrite_probe:
        return {}
    with output_path.open() as handle:
        previous = json.load(handle)
    if previous.get("protocol_config_sha256") != protocol_config_sha256:
        raise RuntimeError(
            f"Existing result protocol does not match requested configuration: {output_path}. "
            "Use --overwrite-probe to replace it explicitly."
        )
    return {int(item["shot"]): item for item in previous.get("results", [])}


def _write_probe_results(
    args,
    output_path: Path,
    feature_dim: int,
    completed: dict[int, dict],
    selection_metadata: dict | None,
    support_metadata: dict,
    checkpoint_metadata: dict,
    protocol_config: dict,
):
    results = [completed[shot] for shot in sorted(completed) if shot in args.shots]
    payload = {
        "protocol": args.protocol,
        "protocol_description": (
            "CLIP-paper-aligned reproduction with DINOv2-style fixed 10% train selection split"
            if args.protocol == "clip-paper-v1"
            else "OpenAI CLIP README fixed-C logistic-regression linear probe"
        ),
        "protocol_config": protocol_config,
        "protocol_config_sha256": _sha256_json(protocol_config),
        "model": args.model,
        "dataset": "ImageNet-1K",
        "support_seed": args.seed,
        "shots": args.shots,
        "selection_split": selection_metadata,
        "support_split": support_metadata,
        "final_evaluation_split": {
            "name": "official ImageNet validation",
            "count": IMAGENET_VAL_SIZE,
            "role": "final evaluation only",
            "dataset_order_sha256": protocol_config["val_dataset_order_sha256"],
        },
        "official_ilsvrc_test_100k": "unused: labels are not public",
        "feature_dim": feature_dim,
        "feature_surface": FEATURE_SURFACES[args.model],
        "feature_normalized": False,
        "preprocessing": TRANSFORM_DESCRIPTIONS[args.model],
        "feature_extraction_batch_size": args.batch_size,
        "checkpoint_manifest": checkpoint_metadata,
        "classifier": "sklearn.linear_model.LogisticRegression(solver='lbfgs')",
        "classifier_config": {
            "random_state": args.seed,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "fixed_C": args.c if args.protocol == "clip-readme-fixed" else None,
            "C_search_initial_log10": list(DEFAULT_C_EXPONENTS) if args.protocol == "clip-paper-v1" else None,
            "C_search_resolution_decades": DEFAULT_C_RESOLUTION if args.protocol == "clip-paper-v1" else None,
            "selection_metric": "top1" if args.protocol == "clip-paper-v1" else None,
            "tie_break": "smallest C" if args.protocol == "clip-paper-v1" else None,
        },
        "environment": _environment_metadata(),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")
    with temporary_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary_path, output_path)


def run_probe(
    args,
    support: np.ndarray,
    support_cache: Path,
    selection_cache: Path | None,
    val_cache: Path,
    output_path: Path,
    selection_metadata: dict | None,
    support_metadata: dict,
    checkpoint_metadata: dict,
    protocol_config: dict,
):
    support_features = np.load(support_cache / "features.npy", mmap_mode="r")
    support_labels = np.load(support_cache / "labels.npy", mmap_mode="r")
    val_features = np.load(val_cache / "features.npy", mmap_mode="r")
    val_labels = np.load(val_cache / "labels.npy", mmap_mode="r")
    selection_features = selection_labels = None
    if args.protocol == "clip-paper-v1":
        if selection_cache is None:
            raise RuntimeError("clip-paper-v1 requires a selection feature cache")
        selection_features = np.load(selection_cache / "features.npy", mmap_mode="r")
        selection_labels = np.load(selection_cache / "labels.npy", mmap_mode="r")
        if len(np.unique(selection_labels)) != NUM_CLASSES:
            raise ValueError("Selection split does not contain all ImageNet classes")

    max_shot = support.shape[1]
    expected_support_labels = np.repeat(np.arange(NUM_CLASSES), max_shot)
    if not np.array_equal(support_labels, expected_support_labels):
        raise ValueError("Cached support features do not match the class-major support split")
    if len(val_labels) != IMAGENET_VAL_SIZE:
        raise ValueError(f"Expected {IMAGENET_VAL_SIZE} ImageNet validation examples, got {len(val_labels)}")

    protocol_config_sha256 = _sha256_json(protocol_config)
    completed = _load_completed_results(output_path, protocol_config_sha256, args.overwrite_probe)

    for shot in args.shots:
        if shot in completed:
            print(f"Using completed {args.model} seed={args.seed} shot={shot} result")
            continue

        positions = (np.arange(NUM_CLASSES)[:, None] * max_shot + np.arange(shot)[None, :]).reshape(-1)
        features = np.asarray(support_features[positions])
        labels = np.asarray(support_labels[positions])
        search_history = []
        selection_top1 = None
        search_started = time.time()

        if args.protocol == "clip-paper-v1":
            assert selection_features is not None and selection_labels is not None

            def evaluate_exponent(exponent: float) -> float:
                c_value = 10.0**exponent
                started = time.time()
                print(
                    f"Selecting C: model={args.model} seed={args.seed} shot={shot} "
                    f"samples={len(labels)} C={c_value:.12g}"
                )
                classifier = _fit_logistic_regression(args, features, labels, c_value)
                top1, _ = evaluate_classifier(
                    classifier,
                    selection_features,
                    selection_labels,
                    compute_top5=False,
                )
                convergence = bool(np.max(classifier.n_iter_) < args.max_iter)
                candidate_details[round(exponent, 12)] = {
                    "n_iter_max": int(np.max(classifier.n_iter_)),
                    "converged": convergence,
                    "elapsed_seconds": time.time() - started,
                }
                return top1

            candidate_details: dict[float, dict] = {}
            selected_c, search_history = parametric_c_search(evaluate_exponent)
            for candidate in search_history:
                candidate.update(candidate_details[candidate["log10_C"]])
            selected_item = next(item for item in search_history if np.isclose(item["C"], selected_c, rtol=1e-12))
            selection_top1 = selected_item["selection_top1"]
        else:
            selected_c = args.c

        search_elapsed_seconds = time.time() - search_started
        final_started = time.time()
        print(
            f"Final fit: model={args.model} seed={args.seed} shot={shot} "
            f"samples={len(labels)} dim={features.shape[1]} C={selected_c:.12g}"
        )
        classifier = _fit_logistic_regression(args, features, labels, selected_c)
        top1, top5 = evaluate_classifier(classifier, val_features, val_labels)
        item = {
            "shot": shot,
            "train_samples": len(labels),
            "selection_samples": len(selection_labels) if selection_labels is not None else 0,
            "selected_C": selected_c,
            "selection_top1": selection_top1,
            "C_search": search_history,
            "top1": top1,
            "top5": top5,
            "max_iter": args.max_iter,
            "n_iter_max": int(np.max(classifier.n_iter_)),
            "converged": bool(np.max(classifier.n_iter_) < args.max_iter),
            "search_elapsed_seconds": search_elapsed_seconds,
            "final_fit_and_eval_elapsed_seconds": time.time() - final_started,
        }
        print(json.dumps(item, indent=2))
        completed[shot] = item
        _write_probe_results(
            args,
            output_path,
            int(support_features.shape[1]),
            completed,
            selection_metadata,
            support_metadata,
            checkpoint_metadata,
            protocol_config,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="clip-paper-v1")
    parser.add_argument("--shots", type=int, nargs="+", default=list(DEFAULT_SHOTS))
    parser.add_argument("--seed", type=int, default=0, help="Support-set seed.")
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--selection-fraction", type=float, default=DEFAULT_SELECTION_FRACTION)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Feature-extraction batch size; OpenAI CLIP README example uses 100.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--c", type=float, default=DEFAULT_C, help="Fixed C used only by clip-readme-fixed.")
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--logreg-verbose", type=int, default=1)
    parser.add_argument("--prepare-split-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--overwrite-probe", action="store_true")
    parser.add_argument("--overwrite-features", action="store_true")
    args = parser.parse_args(argv)

    args.shots = sorted(set(args.shots))
    if not args.shots or args.shots[0] < 1:
        parser.error("--shots must contain positive integers")
    if args.batch_size < 1 or args.num_workers < 0 or args.max_iter < 1 or args.tol <= 0:
        parser.error("batch size/max iterations must be positive, workers nonnegative, and tol positive")
    if not 0.0 < args.selection_fraction < 1.0:
        parser.error("--selection-fraction must be between zero and one")
    if args.c <= 0:
        parser.error("--c must be positive")
    # Keep subset runs compatible with the canonical nested 1/2/4/8/16 split.
    args.max_shot = max(max(args.shots), max(DEFAULT_SHOTS))
    fraction_string = format(args.selection_fraction, ".12g")
    args.selection_split_path = (
        args.output_root / "splits" / f"selection_seed{args.selection_seed}_fraction{fraction_string}.npz"
    )
    support_filename = (
        f"support_seed{args.seed}_max{args.max_shot}.npz"
        if args.protocol == "clip-paper-v1"
        else f"legacy_fulltrain_support_seed{args.seed}_max{args.max_shot}.npz"
    )
    args.support_split_path = args.output_root / "splits" / support_filename
    return args


def _make_cache_fingerprint(
    args,
    checkpoint_metadata: dict,
    split_role: str,
    indices_sha256: str,
) -> dict:
    return {
        "protocol": args.protocol,
        "model": args.model,
        "feature_surface": FEATURE_SURFACES[args.model],
        "feature_normalized": False,
        "transform": TRANSFORM_DESCRIPTIONS[args.model],
        "checkpoint_manifest_sha256": checkpoint_metadata["sha256"],
        "split_role": split_role,
        "indices_sha256": indices_sha256,
        "extraction_batch_size": args.batch_size,
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "timm_version": timm.__version__,
    }


def main(argv=None):
    args = parse_args(argv)
    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Expected ImageNet train/ and val/ under {args.data_root}")

    target_dataset = datasets.ImageFolder(train_dir)
    if len(target_dataset) != IMAGENET_TRAIN_SIZE or len(target_dataset.classes) != NUM_CLASSES:
        raise ValueError(f"Unexpected ImageNet train set: samples={len(target_dataset)}, classes={len(target_dataset.classes)}")
    val_target_dataset = datasets.ImageFolder(val_dir)
    if len(val_target_dataset) != IMAGENET_VAL_SIZE or len(val_target_dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"Unexpected ImageNet validation set: samples={len(val_target_dataset)}, "
            f"classes={len(val_target_dataset.classes)}"
        )
    if target_dataset.class_to_idx != val_target_dataset.class_to_idx:
        raise ValueError("ImageNet train and validation class mappings differ")
    dataset_sha256 = _dataset_order_sha256(target_dataset)
    val_dataset_sha256 = _dataset_order_sha256(val_target_dataset)

    selection_indices = None
    selection_metadata = None
    if args.protocol == "clip-paper-v1":
        selection_indices, support_pool, selection_metadata = load_or_create_train_selection(
            target_dataset,
            args.selection_split_path,
            args.selection_fraction,
            args.selection_seed,
            dataset_sha256,
        )
        pool_sha256 = selection_metadata["support_pool_indices_sha256"]
    else:
        support_pool = np.arange(len(target_dataset), dtype=np.int64)
        pool_sha256 = _sha256_array(support_pool)

    support, support_metadata = load_or_create_support(
        target_dataset,
        args.support_split_path,
        support_pool,
        args.max_shot,
        args.seed,
        dataset_sha256,
        pool_sha256,
    )
    print(f"Support split: {args.support_split_path} shape={support.shape}")
    if selection_metadata:
        print(
            f"Selection split: {args.selection_split_path} count={selection_metadata['selection_count']} "
            f"support_pool={selection_metadata['support_pool_count']}"
        )
    if args.prepare_split_only:
        return

    checkpoint_metadata = checkpoint_manifest(args.model)
    model_root = args.output_root / args.model
    seed_root = model_root / f"seed{args.seed}"
    support_cache = seed_root / f"features_train_max{args.max_shot}"
    selection_cache = model_root / "features_selection" if selection_indices is not None else None
    val_cache = model_root / "features_val"
    output_path = seed_root / "results.json"

    support_fingerprint = _make_cache_fingerprint(
        args, checkpoint_metadata, "support", support_metadata["support_indices_sha256"]
    )
    selection_fingerprint = None
    if selection_metadata is not None:
        selection_fingerprint = _make_cache_fingerprint(
            args, checkpoint_metadata, "selection", selection_metadata["selection_indices_sha256"]
        )
    val_fingerprint = _make_cache_fingerprint(
        args, checkpoint_metadata, "official_val", val_dataset_sha256
    )

    cache_requirements = [(support_cache, len(support.reshape(-1)), support_fingerprint)]
    if selection_cache is not None and selection_fingerprint is not None and selection_indices is not None:
        cache_requirements.append((selection_cache, len(selection_indices), selection_fingerprint))
    cache_requirements.append((val_cache, IMAGENET_VAL_SIZE, val_fingerprint))

    if args.probe_only:
        for cache_dir, count, fingerprint in cache_requirements:
            if not _feature_cache_complete(cache_dir, count, fingerprint):
                raise FileNotFoundError(f"Required feature cache is missing: {cache_dir}")
    else:
        caches_complete = all(
            _feature_cache_complete(cache_dir, count, fingerprint, overwrite=args.overwrite_features)
            for cache_dir, count, fingerprint in cache_requirements
        )
        if args.smoke_test or not caches_complete:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for tokenizer feature extraction")
            device = torch.device("cuda")
            encoder, transform, autocast_factory = load_feature_encoder(args.model, device)
            parameters = sum(parameter.numel() for parameter in encoder.parameters())
            print(f"Loaded {args.model}: parameters={parameters:,}")

            if args.smoke_test:
                dataset = datasets.ImageFolder(val_dir, transform=transform)
                images = torch.stack([dataset[index][0] for index in range(2)]).to(device)
                with torch.inference_mode(), autocast_factory():
                    features = encoder(images)
                print(
                    f"Smoke test {args.model}: images={tuple(images.shape)} features={tuple(features.shape)} "
                    f"finite={bool(torch.isfinite(features).all())}"
                )
                return

            train_dataset = datasets.ImageFolder(train_dir, transform=transform)
            val_dataset = datasets.ImageFolder(val_dir, transform=transform)
            extract_features(
                encoder,
                train_dataset,
                support.reshape(-1),
                support_cache,
                args.batch_size,
                args.num_workers,
                device,
                autocast_factory,
                support_fingerprint,
                args.overwrite_features,
            )
            if selection_cache is not None and selection_fingerprint is not None and selection_indices is not None:
                extract_features(
                    encoder,
                    train_dataset,
                    selection_indices,
                    selection_cache,
                    args.batch_size,
                    args.num_workers,
                    device,
                    autocast_factory,
                    selection_fingerprint,
                    args.overwrite_features,
                )
            extract_features(
                encoder,
                val_dataset,
                None,
                val_cache,
                args.batch_size,
                args.num_workers,
                device,
                autocast_factory,
                val_fingerprint,
                args.overwrite_features,
            )
            del encoder
            gc.collect()
            torch.cuda.empty_cache()

    protocol_config = {
        "protocol": args.protocol,
        "model": args.model,
        "support_seed": args.seed,
        "shots": args.shots,
        "selection_seed": args.selection_seed if selection_metadata else None,
        "selection_fraction": args.selection_fraction if selection_metadata else None,
        "selection_indices_sha256": selection_metadata["selection_indices_sha256"] if selection_metadata else None,
        "support_indices_sha256": support_metadata["support_indices_sha256"],
        "train_dataset_order_sha256": dataset_sha256,
        "val_dataset_order_sha256": val_dataset_sha256,
        "checkpoint_manifest_sha256": checkpoint_metadata["sha256"],
        "batch_size": args.batch_size,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "fixed_C": args.c if args.protocol == "clip-readme-fixed" else None,
        "c_search_initial_log10": list(DEFAULT_C_EXPONENTS) if args.protocol == "clip-paper-v1" else None,
        "c_search_resolution_decades": DEFAULT_C_RESOLUTION if args.protocol == "clip-paper-v1" else None,
        "environment": _environment_metadata(),
    }
    run_probe(
        args,
        support,
        support_cache,
        selection_cache,
        val_cache,
        output_path,
        selection_metadata,
        support_metadata,
        checkpoint_metadata,
        protocol_config,
    )


if __name__ == "__main__":
    main()
