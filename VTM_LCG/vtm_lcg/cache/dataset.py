from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

from vtm_lcg.utils import sha256_file, sha256_json


@dataclass(frozen=True)
class CocoRecord:
    image_id: int
    file_name: str
    image_path: Path
    caption_ids: tuple[int, ...]
    captions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "file_name": self.file_name,
            "image_path": str(self.image_path),
            "caption_ids": list(self.caption_ids),
            "captions": list(self.captions),
        }


def load_coco_karpathy_records(
    annotations_path: Path,
    image_root: Path,
    *,
    limit: int | None,
) -> tuple[list[CocoRecord], dict[str, Any]]:
    annotations_path = annotations_path.resolve()
    image_root = image_root.resolve()
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Missing COCO annotation file: {annotations_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing COCO image root: {image_root}")
    with annotations_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO annotations must contain image and annotation lists")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive or None, got {limit}")

    captions_by_image: dict[int, list[tuple[int, str]]] = {}
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        captions_by_image.setdefault(image_id, []).append(
            (int(annotation["id"]), str(annotation["caption"]))
        )

    selected_images = images if limit is None else images[:limit]
    records: list[CocoRecord] = []
    seen_image_ids: set[int] = set()
    for image in selected_images:
        image_id = int(image["id"])
        if image_id in seen_image_ids:
            raise ValueError(f"Duplicate COCO image id: {image_id}")
        seen_image_ids.add(image_id)
        file_name = str(image["file_name"])
        image_path = image_root / file_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing COCO image: {image_path}")
        caption_entries = captions_by_image.get(image_id, [])
        if not caption_entries:
            raise ValueError(f"COCO image {image_id} has no captions")
        records.append(
            CocoRecord(
                image_id=image_id,
                file_name=file_name,
                image_path=image_path.resolve(),
                caption_ids=tuple(item[0] for item in caption_entries),
                captions=tuple(item[1] for item in caption_entries),
            )
        )
    if not records:
        raise ValueError("COCO selection is empty")

    annotation_sha256 = sha256_file(annotations_path)
    subset_payload = [
        {
            "image_id": record.image_id,
            "file_name": record.file_name,
            "caption_ids": list(record.caption_ids),
        }
        for record in records
    ]
    metadata = {
        "annotations": str(annotations_path),
        "annotations_sha256": annotation_sha256,
        "image_root": str(image_root),
        "record_count": len(records),
        "ordered_subset_sha256": sha256_json(subset_payload),
    }
    metadata["dataset_fingerprint"] = sha256_json(metadata)
    return records, metadata


class CocoImageTensorDataset(Dataset):
    def __init__(
        self,
        records: Sequence[CocoRecord],
        indices: Sequence[int],
        transform: Callable,
    ) -> None:
        self.records = records
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        record_index = self.indices[item]
        record = self.records[record_index]
        with Image.open(record.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, record.image_id, record_index

