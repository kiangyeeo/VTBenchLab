from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".webm", ".mov"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_video_decord(path: Path, frame_count: int) -> List[Image.Image]:
    from decord import VideoReader, cpu

    vr = VideoReader(str(path), ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise RuntimeError(f"No frames in video: {path}")
    indices = torch.linspace(0, total - 1, steps=frame_count).round().long().tolist()
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


def _read_video_pyav(path: Path, frame_count: int) -> List[Image.Image]:
    import av

    container = av.open(str(path))
    stream = container.streams.video[0]
    total = stream.frames or 0
    decoded = [frame.to_image().convert("RGB") for frame in container.decode(stream)]
    if not decoded:
        raise RuntimeError(f"No frames in video: {path}")
    indices = torch.linspace(0, len(decoded) - 1, steps=frame_count).round().long().tolist()
    return [decoded[i] for i in indices]


def read_video_frames(path: Path, frame_count: int) -> List[Image.Image]:
    try:
        return _read_video_decord(path, frame_count)
    except Exception:
        return _read_video_pyav(path, frame_count)


def read_frame_folder(path: Path, frame_count: int) -> List[Image.Image]:
    frames = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not frames:
        raise RuntimeError(f"No image frames found in {path}")
    indices = torch.linspace(0, len(frames) - 1, steps=frame_count).round().long().tolist()
    return [Image.open(frames[i]).convert("RGB") for i in indices]


class VideoClassificationDataset(Dataset):
    """Video dataset supporting class folders or CSV manifests.

    CSV format: path,label where path is relative to the split root unless absolute.
    """

    def __init__(
        self,
        root: os.PathLike,
        split: str,
        *,
        transform: Optional[Callable] = None,
        frame_count: int = 8,
        smoke: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.split_root = self.root / split
        self.transform = transform
        self.frame_count = frame_count
        self.samples = self._discover_samples()
        if smoke:
            self.samples = self.samples[: min(32, len(self.samples))]
        if not self.samples:
            raise FileNotFoundError(f"No video samples found under {self.split_root}")

    def _discover_samples(self) -> List[Tuple[Path, int]]:
        manifest = self.root / f"{self.split}.csv"
        if manifest.exists():
            samples = []
            with open(manifest, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rel = row.get("path") or row.get("video") or row.get("filename")
                    label = row.get("label") or row.get("target") or row.get("class")
                    if rel is None or label is None:
                        raise ValueError(f"Manifest {manifest} needs path/video and label/target columns")
                    p = Path(rel)
                    samples.append((p if p.is_absolute() else self.split_root / p, int(label)))
            return samples

        if not self.split_root.exists():
            raise FileNotFoundError(f"Missing video split directory: {self.split_root}")
        classes = sorted(p.name for p in self.split_root.iterdir() if p.is_dir())
        class_to_idx = {name: i for i, name in enumerate(classes)}
        samples: List[Tuple[Path, int]] = []
        for cls in classes:
            for p in sorted((self.split_root / cls).iterdir()):
                if p.is_dir() or p.suffix.lower() in VIDEO_EXTS:
                    samples.append((p, class_to_idx[cls]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        frames = read_frame_folder(path, self.frame_count) if path.is_dir() else read_video_frames(path, self.frame_count)
        if self.transform is not None:
            frames = [self.transform(frame) for frame in frames]
        return torch.stack(frames, dim=0), target

