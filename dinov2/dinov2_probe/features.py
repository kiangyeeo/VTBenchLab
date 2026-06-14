from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import ensure_dir
from .datasets import build_dataset, collate_with_flexible_targets, dataset_spec
from .model_zoo import build_encoder


def feature_cache_path(
    feature_root: os.PathLike,
    model_alias: str,
    dataset_name: str,
    split: str,
    *,
    aggregation: str = "image",
) -> Path:
    safe = f"{split}_{aggregation}".replace("/", "_")
    return Path(feature_root) / model_alias / dataset_name / f"{safe}.pt"


def _targets_to_save(targets):
    if all(isinstance(t, torch.Tensor) and t.ndim == 0 for t in targets):
        return torch.stack(targets).long()
    if all(isinstance(t, int) for t in targets):
        return torch.tensor(targets, dtype=torch.long)
    return targets


def extract_features(
    *,
    model_alias: str,
    dataset_name: str,
    split: str,
    data_root: os.PathLike,
    feature_root: os.PathLike,
    model_root: os.PathLike,
    model_config: Optional[str] = None,
    dataset_config: Optional[str] = None,
    batch_size: int = 256,
    num_workers: int = 8,
    allow_download: bool = False,
    dataset_download: bool = False,
    overwrite: bool = False,
    smoke: bool = False,
    device: str = "cuda",
    amp_dtype: str = "fp16",
) -> Path:
    spec = dataset_spec(dataset_name, dataset_config)
    aggregation = spec.get("aggregation", "image")
    out_path = feature_cache_path(feature_root, model_alias, dataset_name, split, aggregation=aggregation)
    if out_path.exists() and not overwrite:
        print(f"[skip] feature cache exists: {out_path}")
        return out_path

    encoder = build_encoder(
        model_alias,
        config_path=model_config,
        model_root=model_root,
        device=device,
        allow_download=allow_download,
        video_aggregation=aggregation,
    )
    dataset = build_dataset(
        dataset_name,
        split,
        transform=encoder.image_transform,
        data_root=data_root,
        config_path=dataset_config,
        download=dataset_download,
        smoke=smoke,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate_with_flexible_targets,
    )

    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16 if amp_dtype == "bf16" else torch.float32
    features = []
    targets = []
    start = time.time()
    seen = 0
    ensure_dir(out_path.parent)

    pbar = tqdm(loader, desc=f"features {model_alias}/{dataset_name}/{split}", dynamic_ncols=True)
    with torch.no_grad():
        for images, target in pbar:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda")):
                feat = encoder(images)
            features.append(feat.cpu())
            if isinstance(target, torch.Tensor):
                targets.extend([t.cpu() for t in target])
            else:
                targets.extend(target)
            seen += len(images)
            elapsed = max(time.time() - start, 1e-6)
            pbar.set_postfix(samples=seen, img_s=f"{seen / elapsed:.1f}")

    feature_tensor = torch.cat(features, dim=0)
    payload: Dict[str, Any] = {
        "features": feature_tensor,
        "targets": _targets_to_save(targets),
        "metadata": {
            "model": model_alias,
            "dataset": dataset_name,
            "split": split,
            "aggregation": aggregation,
            "feature_dim": int(feature_tensor.shape[1]),
            "num_samples": int(feature_tensor.shape[0]),
            "model_metadata": encoder.metadata,
            "smoke": smoke,
        },
    }
    torch.save(payload, out_path)
    print(f"[done] wrote {out_path} shape={tuple(feature_tensor.shape)}")
    return out_path


def load_feature_cache(path: os.PathLike) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def ensure_features(**kwargs) -> Path:
    return extract_features(**kwargs)
