#!/usr/bin/env python
"""CLIP-style ImageNet k-shot linear probing for tokenizer features.

OpenAI CLIP documents linear probing in README.md but does not ship a launch
script.  This file turns that example into a reproducible ImageNet-1K
benchmark for the tokenizers used in VTBenchLab:

* freeze the feature extractor;
* use deterministic resize + center-crop preprocessing;
* draw exactly k training images per ImageNet class;
* fit scikit-learn LogisticRegression with the OpenAI example defaults; and
* evaluate on all 50,000 ImageNet validation images.

The same nested support split is shared by every tokenizer.  For example, the
1-shot support is a strict subset of the 2-, 4-, 8-, and 16-shot supports.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
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
DEFAULT_OUTPUT_ROOT = WORKSPACE / "outputs/imagenet_kshot_linear_clip"
DEFAULT_SHOTS = (1, 2, 4, 8, 16)
DEFAULT_C = 0.316
NUM_CLASSES = 1_000

MODEL_NAMES = ("unitok", "toklips", "toklipl", "vilau", "metaclip")
FEATURE_SURFACES = {
    "unitok": "mean-pooled quantized tokens after fc_norm, before UniTok projection",
    "toklips": "TokLIP-S semantic visual representation before the output head",
    "toklipl": "TokLIP-L semantic visual representation before the output head",
    "vilau": "mean-pooled penultimate SigLIP tokens used as the VILA-U tokenizer input",
    "metaclip": "final MetaCLIP class token after norm, before the 768-to-512 projection",
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
        layer_outputs = encoder_layer(
            hidden_states,
            None,
            output_attentions=None,
        )
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


def make_nested_support_indices(targets: np.ndarray, max_shot: int, seed: int) -> np.ndarray:
    """Return [num_classes, max_shot] indices with deterministic class balance."""
    targets = np.asarray(targets, dtype=np.int64)
    classes = np.unique(targets)
    if len(classes) != NUM_CLASSES or not np.array_equal(classes, np.arange(NUM_CLASSES)):
        raise ValueError(f"Expected ImageNet labels 0..999, found {len(classes)} classes")

    rng = np.random.default_rng(seed)
    support = np.empty((NUM_CLASSES, max_shot), dtype=np.int64)
    for class_index in range(NUM_CLASSES):
        candidates = np.flatnonzero(targets == class_index)
        if len(candidates) < max_shot:
            raise ValueError(f"Class {class_index} has only {len(candidates)} samples")
        support[class_index] = rng.choice(candidates, size=max_shot, replace=False)
    return support


def load_or_create_support(dataset: datasets.ImageFolder, split_path: Path, max_shot: int, seed: int) -> np.ndarray:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    if split_path.exists():
        payload = np.load(split_path)
        support = payload["support_indices"]
        if support.shape != (NUM_CLASSES, max_shot):
            raise ValueError(f"Existing split has unexpected shape {support.shape}: {split_path}")
        return support

    support = make_nested_support_indices(np.asarray(dataset.targets), max_shot, seed)
    np.savez(
        split_path,
        support_indices=support,
        max_shot=np.asarray(max_shot),
        seed=np.asarray(seed),
    )
    return support


def _feature_cache_complete(cache_dir: Path, expected_count: int) -> bool:
    metadata_path = cache_dir / "metadata.json"
    feature_path = cache_dir / "features.npy"
    label_path = cache_dir / "labels.npy"
    if not metadata_path.is_file() or not feature_path.is_file() or not label_path.is_file():
        return False
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    return metadata.get("count") == expected_count and np.load(feature_path, mmap_mode="r").shape[0] == expected_count


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
):
    indices = None if indices is None else np.asarray(list(indices), dtype=np.int64)
    selected_dataset = dataset if indices is None else Subset(dataset, indices.tolist())
    expected_count = len(selected_dataset)
    if _feature_cache_complete(cache_dir, expected_count):
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

    if offset != expected_count:
        raise RuntimeError(f"Wrote {offset} features, expected {expected_count}")
    feature_memmap.flush()
    label_memmap.flush()
    metadata = {
        "count": expected_count,
        "feature_dim": int(feature_memmap.shape[1]),
        "dtype": "float32",
        "elapsed_seconds": time.time() - started,
    }
    with (cache_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)


def _topk_correct(logits: np.ndarray, targets: np.ndarray, k: int) -> int:
    if k == 1:
        return int(np.sum(np.argmax(logits, axis=1) == targets))
    topk = np.argpartition(logits, -k, axis=1)[:, -k:]
    return int(np.sum(np.any(topk == targets[:, None], axis=1)))


def evaluate_classifier(classifier: LogisticRegression, features: np.ndarray, targets: np.ndarray, batch_size: int = 2_048):
    top1 = 0
    top5 = 0
    count = len(targets)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        logits = classifier.decision_function(features[start:end])
        top1 += _topk_correct(logits, targets[start:end], 1)
        top5 += _topk_correct(logits, targets[start:end], 5)
    return 100.0 * top1 / count, 100.0 * top5 / count


def _write_probe_results(args, output_path: Path, feature_dim: int, completed: dict):
    results = [completed[k] for k in sorted(completed) if k in args.shots]
    payload = {
        "protocol": "OpenAI CLIP README logistic-regression linear probe",
        "model": args.model,
        "dataset": "ImageNet-1K",
        "seed": args.seed,
        "shots": args.shots,
        "support_split": str(args.split_path),
        "feature_dim": feature_dim,
        "feature_surface": FEATURE_SURFACES[args.model],
        "feature_normalized": False,
        "preprocessing": "deterministic model-native resize/center-crop",
        "classifier": "sklearn.linear_model.LogisticRegression(solver='lbfgs')",
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def run_probe(args, support: np.ndarray, train_cache: Path, val_cache: Path, output_path: Path):
    train_features = np.load(train_cache / "features.npy", mmap_mode="r")
    train_labels = np.load(train_cache / "labels.npy", mmap_mode="r")
    val_features = np.load(val_cache / "features.npy", mmap_mode="r")
    val_labels = np.load(val_cache / "labels.npy", mmap_mode="r")

    max_shot = support.shape[1]
    expected_train_labels = np.repeat(np.arange(NUM_CLASSES), max_shot)
    if not np.array_equal(train_labels, expected_train_labels):
        raise ValueError("Cached train features do not match the class-major support split")
    if len(val_labels) != 50_000:
        raise ValueError(f"Expected 50,000 ImageNet validation examples, got {len(val_labels)}")

    previous = {}
    if output_path.exists():
        with output_path.open() as handle:
            previous = json.load(handle)
    completed = {int(item["shot"]): item for item in previous.get("results", [])}

    results = []
    for shot in args.shots:
        if shot in completed and not args.overwrite_probe:
            print(f"Using completed {args.model} {shot}-shot result")
            results.append(completed[shot])
            continue

        positions = (np.arange(NUM_CLASSES)[:, None] * max_shot + np.arange(shot)[None, :]).reshape(-1)
        features = np.asarray(train_features[positions])
        labels = np.asarray(train_labels[positions])
        started = time.time()
        classifier = LogisticRegression(
            random_state=args.seed,
            C=args.c,
            max_iter=args.max_iter,
            verbose=args.logreg_verbose,
            n_jobs=-1,
            solver="lbfgs",
            tol=args.tol,
        )
        print(f"Fitting {args.model}: shot={shot}, samples={len(labels)}, dim={features.shape[1]}, C={args.c}")
        classifier.fit(features, labels)
        top1, top5 = evaluate_classifier(classifier, val_features, val_labels)
        item = {
            "shot": shot,
            "train_samples": len(labels),
            "top1": top1,
            "top5": top5,
            "C": args.c,
            "max_iter": args.max_iter,
            "n_iter_max": int(np.max(classifier.n_iter_)),
            "converged": bool(np.max(classifier.n_iter_) < args.max_iter),
            "elapsed_seconds": time.time() - started,
        }
        print(json.dumps(item, indent=2))
        completed[shot] = item
        _write_probe_results(args, output_path, int(train_features.shape[1]), completed)

    _write_probe_results(args, output_path, int(train_features.shape[1]), completed)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shots", type=int, nargs="+", default=list(DEFAULT_SHOTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--c", type=float, default=DEFAULT_C, help="Inverse L2 strength; OpenAI README example uses 0.316.")
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--logreg-verbose", type=int, default=1)
    parser.add_argument("--prepare-split-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--overwrite-probe", action="store_true")
    args = parser.parse_args()

    args.shots = sorted(set(args.shots))
    if not args.shots or args.shots[0] < 1:
        parser.error("--shots must contain positive integers")
    args.max_shot = max(args.shots)
    args.split_path = args.output_root / "splits" / f"imagenet1k_seed{args.seed}_max{args.max_shot}.npz"
    return args


def main():
    args = parse_args()
    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Expected ImageNet train/ and val/ under {args.data_root}")

    target_dataset = datasets.ImageFolder(train_dir)
    if len(target_dataset) != 1_281_167 or len(target_dataset.classes) != NUM_CLASSES:
        raise ValueError(f"Unexpected ImageNet train set: samples={len(target_dataset)}, classes={len(target_dataset.classes)}")
    support = load_or_create_support(target_dataset, args.split_path, args.max_shot, args.seed)
    print(f"Support split: {args.split_path} shape={support.shape}")
    if args.prepare_split_only:
        return

    model_root = args.output_root / args.model / f"seed{args.seed}"
    train_cache = model_root / f"features_train_max{args.max_shot}"
    val_cache = model_root / "features_val"
    output_path = model_root / "results.json"

    if not args.probe_only:
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
            train_cache,
            args.batch_size,
            args.num_workers,
            device,
            autocast_factory,
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
        )
        del encoder
        gc.collect()
        torch.cuda.empty_cache()

    run_probe(args, support, train_cache, val_cache, output_path)


if __name__ == "__main__":
    main()
