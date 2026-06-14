from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder

from .common import load_dataset_config, workspace_path
from .video_dataset import VideoClassificationDataset


def _clip_builder():
    try:
        from clip_benchmark.datasets.builder import build_dataset
    except ImportError:
        import sys

        sys.path.insert(0, str(workspace_path("CLIP_benchmark")))
        from clip_benchmark.datasets.builder import build_dataset
    return build_dataset


def imagenet_wnids() -> List[str]:
    try:
        from clip_benchmark.datasets.builder import all_imagenet_wordnet_ids
    except ImportError:
        import sys

        sys.path.insert(0, str(workspace_path("CLIP_benchmark")))
        from clip_benchmark.datasets.builder import all_imagenet_wordnet_ids
    return list(all_imagenet_wordnet_ids)


class TargetRemapDataset(Dataset):
    def __init__(self, dataset: Dataset, mapping: Dict[int, int]) -> None:
        self.dataset = dataset
        self.mapping = mapping

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        return image, self.mapping[int(target)]


class ImageNetRealDataset(Dataset):
    """ImageNet val with optional ReaL multi-label targets.

    If ReaL labels are absent, this falls back to ImageFolder labels and records a warning
    in the instance metadata so the result file remains honest.
    """

    def __init__(self, root: os.PathLike, *, transform: Optional[Callable] = None) -> None:
        self.dataset = ImageFolder(root, transform=transform)
        self.metadata: Dict[str, Any] = {}
        self.real_targets = self._load_real_targets(Path(root))

    def _load_real_targets(self, val_root: Path):
        candidates = [
            val_root.parent / "imagenet_real.json",
            val_root.parent / "real.json",
            val_root / "imagenet_real.json",
            val_root / "real.json",
        ]
        for path in candidates:
            if path.exists():
                with open(path, "r") as f:
                    labels = json.load(f)
                self.metadata["target_source"] = str(path)
                return labels
        self.metadata["warning"] = "ImageNet-ReaL labels not found; using ImageNet val labels."
        return None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        if self.real_targets is None:
            return image, target
        labels = self.real_targets[index]
        return image, labels if isinstance(labels, list) else [labels]


class ImageNetCDataset(Dataset):
    def __init__(
        self,
        root: os.PathLike,
        *,
        transform: Optional[Callable] = None,
        severity: str = "all",
        corruption: str = "all",
        smoke: bool = False,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples = self._discover(severity=severity, corruption=corruption)
        if smoke:
            self.samples = self.samples[: min(256, len(self.samples))]
        if not self.samples:
            raise FileNotFoundError(f"No ImageNet-C samples found in {self.root}")

    def _discover(self, *, severity: str, corruption: str) -> List[Tuple[Path, int]]:
        wnid_to_idx = {wnid: i for i, wnid in enumerate(imagenet_wnids())}
        corruptions = [Path(corruption)] if corruption != "all" else sorted(p.name for p in self.root.iterdir() if p.is_dir())
        samples: List[Tuple[Path, int]] = []
        severities = None if severity == "all" else {str(severity)}
        for corr in corruptions:
            corr_root = self.root / corr
            if not corr_root.exists():
                continue
            for sev_dir in sorted(p for p in corr_root.iterdir() if p.is_dir()):
                if severities is not None and sev_dir.name not in severities:
                    continue
                for class_dir in sorted(p for p in sev_dir.iterdir() if p.is_dir()):
                    if class_dir.name not in wnid_to_idx:
                        continue
                    target = wnid_to_idx[class_dir.name]
                    for image in sorted(class_dir.iterdir()):
                        if image.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                            samples.append((image, target))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def collate_with_flexible_targets(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    if all(isinstance(t, int) for t in targets):
        return images, torch.tensor(targets, dtype=torch.long)
    return images, list(targets)


def _subset_for_smoke(dataset: Dataset, enabled: bool, n: int = 256) -> Dataset:
    if not enabled:
        return dataset
    return Subset(dataset, list(range(min(n, len(dataset)))))


def dataset_spec(name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_dataset_config(config_path)
    datasets = cfg.get("datasets", {})
    if name not in datasets:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(sorted(datasets))}")
    return datasets[name]


def linear_dataset_names(config_path: Optional[str] = None) -> List[str]:
    cfg = load_dataset_config(config_path)
    return list(cfg.get("linear_evaluation_sets", []))


def resolve_split(name: str, split_role: str, config_path: Optional[str] = None) -> Tuple[str, str]:
    spec = dataset_spec(name, config_path)
    if split_role == "train" and spec.get("train_dataset"):
        train_name = spec["train_dataset"]
        train_spec = dataset_spec(train_name, config_path)
        return train_name, train_spec.get("train_split", "train")
    if split_role == "train":
        return name, spec.get("train_split", "train")
    return name, spec.get("eval_split", "test")


def build_dataset(
    name: str,
    split: str,
    *,
    transform,
    data_root: os.PathLike,
    config_path: Optional[str] = None,
    download: bool = False,
    smoke: bool = False,
) -> Dataset:
    spec = dataset_spec(name, config_path)
    ds_type = spec["type"]
    root = Path(data_root) / spec.get("root", name)

    if ds_type == "clip_benchmark":
        ds = _clip_builder()(
            spec.get("clip_name", name),
            root=str(root),
            transform=transform,
            split=split,
            download=download,
            task="linear_probe",
        )
        return _subset_for_smoke(ds, smoke)

    if ds_type == "imagefolder":
        split_root = root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Missing dataset split: {split_root}")
        return _subset_for_smoke(ImageFolder(split_root, transform=transform), smoke)

    if ds_type == "imagenet_wnid_folder":
        if not root.exists():
            raise FileNotFoundError(f"Missing ImageNet-style dataset root: {root}")
        ds = ImageFolder(root, transform=transform)
        wnid_to_idx = {wnid: i for i, wnid in enumerate(imagenet_wnids())}
        mapping = {local_idx: wnid_to_idx[wnid] for wnid, local_idx in ds.class_to_idx.items() if wnid in wnid_to_idx}
        if len(mapping) != len(ds.class_to_idx):
            missing = sorted(set(ds.class_to_idx) - set(wnid_to_idx))
            raise ValueError(f"Found non-ImageNet WNID folders in {root}: {missing[:5]}")
        return _subset_for_smoke(TargetRemapDataset(ds, mapping), smoke)

    if ds_type == "imagenet_real":
        if not root.exists():
            raise FileNotFoundError(f"Missing ImageNet-ReaL val root: {root}")
        return _subset_for_smoke(ImageNetRealDataset(root, transform=transform), smoke)

    if ds_type == "imagenet_c":
        return ImageNetCDataset(
            root,
            transform=transform,
            severity=str(spec.get("severity", "all")),
            corruption=str(spec.get("corruption", "all")),
            smoke=smoke,
        )

    if ds_type == "video_folder":
        return VideoClassificationDataset(
            root,
            split,
            transform=transform,
            frame_count=int(spec.get("frame_count", 8)),
            smoke=smoke,
        )

    raise ValueError(f"Unsupported dataset type for {name}: {ds_type}")


def get_targets(dataset: Dataset) -> List[Any]:
    if isinstance(dataset, Subset):
        base_targets = get_targets(dataset.dataset)
        return [base_targets[i] for i in dataset.indices]
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        return targets.tolist() if hasattr(targets, "tolist") else list(targets)
    if hasattr(dataset, "samples"):
        return [target for _, target in getattr(dataset, "samples")]
    return [dataset[i][1] for i in range(len(dataset))]


def class_count_from_targets(targets: Sequence[Any]) -> int:
    ints = [int(t) for t in targets if isinstance(t, (int, float))]
    if not ints:
        raise ValueError("Cannot infer class count from non-integer targets")
    return max(ints) + 1
