#!/usr/bin/env python
"""Deterministic dataset manifests shared by LAR extraction scripts."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
COCO_ROOT = WORKSPACE / "data" / "gvt" / "raw" / "coco"
CAPTIONS_PATH = COCO_ROOT / "annotations" / "captions_val2017.json"
VQA_PATH = WORKSPACE / "data" / "gvt" / "raw" / "vqa" / "v2_mscoco_val2014_annotations.json"
IMAGENET_ROOT = WORKSPACE / "data" / "imagenet1k"
CONFIG_ROOT = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class TextDataset:
    domain: str
    image_set: str
    ids: list[str]
    texts: list[str]
    answer_counts: list[int] | None = None


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _caption_records() -> tuple[dict[int, str], set[int]]:
    payload = _load_json(CAPTIONS_PATH)
    annotations = sorted(payload["annotations"], key=lambda row: int(row["id"]))
    first_by_image: dict[int, str] = {}
    for row in annotations:
        first_by_image.setdefault(int(row["image_id"]), str(row["caption"]))
    image_ids = {int(row["id"]) for row in payload["images"]}
    if set(first_by_image) != image_ids:
        raise RuntimeError("COCO caption annotations do not cover exactly the val2017 images")
    return first_by_image, image_ids


def _answer_records() -> tuple[dict[int, str], dict[int, int]]:
    payload = _load_json(VQA_PATH)
    candidates: dict[int, list[dict]] = defaultdict(list)
    for row in payload["annotations"]:
        if row.get("answer_type") == "other":
            candidates[int(row["image_id"])].append(row)

    selected: dict[int, str] = {}
    counts: dict[int, int] = {}
    for image_id, rows in candidates.items():
        row = min(rows, key=lambda item: int(item["question_id"]))
        selected[image_id] = str(row["multiple_choice_answer"])
        counts[image_id] = len(rows)
    return selected, counts


def _imagenet_records(seed: int = 0, sample_size: int = 10_000) -> tuple[list[str], list[str]]:
    val_root = IMAGENET_ROOT / "val"
    class_names_path = IMAGENET_ROOT / "extra" / "class-names-VAL.npy"
    if not val_root.is_dir() or not class_names_path.is_file():
        raise FileNotFoundError("ImageNet val or class-name metadata is missing")

    paths = sorted(path for path in val_root.glob("*/*") if path.is_file())
    if len(paths) < sample_size:
        raise ValueError(f"Requested {sample_size} ImageNet images, found only {len(paths)}")
    selected = random.Random(seed).sample(paths, sample_size)
    class_names = np.load(class_names_path, allow_pickle=False)
    if class_names.shape != (1000,):
        raise RuntimeError(f"Expected 1000 ImageNet class names, got {class_names.shape}")

    ids: list[str] = []
    texts: list[str] = []
    for path in selected:
        relative = path.relative_to(IMAGENET_ROOT).as_posix()
        folder = path.parent.name
        if not folder.startswith("class_"):
            raise RuntimeError(f"Unexpected ImageNet class directory: {folder}")
        class_index = int(folder.removeprefix("class_"))
        ids.append(relative)
        texts.append(f"a photo of a {class_names[class_index]}")
    return ids, texts


def build_text_dataset(domain: str, image_set: str, *, seed: int = 0) -> TextDataset:
    if domain == "imagenet" or image_set == "in1k10k":
        if domain != "imagenet" or image_set != "in1k10k":
            raise ValueError("ImageNet text requires domain=imagenet and image_set=in1k10k")
        ids, texts = _imagenet_records(seed=seed)
        return TextDataset(domain, image_set, ids, texts)

    captions, all_ids = _caption_records()
    answers, answer_counts = _answer_records()
    common_ids = sorted(all_ids & set(answers))
    if len(all_ids) != 5000 or len(common_ids) != 4618:
        raise RuntimeError(
            f"Dataset cardinality changed: COCO={len(all_ids)}, answer-covered={len(common_ids)}"
        )

    if image_set == "coco4618":
        ids_int = common_ids
    elif image_set == "coco5000":
        if domain != "caption":
            raise ValueError("Only caption is defined on coco5000")
        ids_int = sorted(all_ids)
    else:
        raise ValueError(f"Unsupported COCO image set: {image_set}")

    if domain == "caption":
        texts = [captions[image_id] for image_id in ids_int]
        counts = None
    elif domain == "answer":
        texts = [answers[image_id] for image_id in ids_int]
        counts = [answer_counts[image_id] for image_id in ids_int]
    else:
        raise ValueError(f"Unsupported text domain: {domain}")
    return TextDataset(domain, image_set, [str(value) for value in ids_int], texts, counts)


def image_path(image_set: str, identifier: str) -> Path:
    if image_set.startswith("coco"):
        return COCO_ROOT / "val2017" / f"{int(identifier):012d}.jpg"
    if image_set == "in1k10k":
        return IMAGENET_ROOT / identifier
    raise ValueError(f"Unsupported image set: {image_set}")


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def write_manifests(seed: int = 0) -> dict[str, Path]:
    datasets = (
        build_text_dataset("caption", "coco4618", seed=seed),
        build_text_dataset("answer", "coco4618", seed=seed),
        build_text_dataset("caption", "coco5000", seed=seed),
        build_text_dataset("imagenet", "in1k10k", seed=seed),
    )
    paths: dict[str, Path] = {}
    for dataset in datasets:
        id_path = CONFIG_ROOT / f"image_ids_{dataset.image_set}.txt"
        if not id_path.exists():
            write_lines(id_path, dataset.ids)
        elif id_path.read_text(encoding="utf-8").splitlines() != dataset.ids:
            raise RuntimeError(f"Existing manifest disagrees with seed={seed}: {id_path}")
        paths[dataset.image_set] = id_path
    return paths

