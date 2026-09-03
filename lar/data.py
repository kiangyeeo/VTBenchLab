#!/usr/bin/env python
"""Deterministic dataset manifests shared by LAR extraction scripts."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
COCO_ROOT = WORKSPACE / "data" / "gvt" / "raw" / "coco"
CAPTIONS_PATH = COCO_ROOT / "annotations" / "captions_val2017.json"
VQA_PATH = WORKSPACE / "data" / "gvt" / "raw" / "vqa" / "v2_mscoco_val2014_annotations.json"
VQA_QUESTIONS_PATH = (
    WORKSPACE / "data" / "gvt" / "raw" / "vqa" /
    "v2_OpenEnded_mscoco_val2014_questions.json"
)
CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
EVAL_ANSWER_SOURCES = CONFIG_ROOT / "eval_answer_sources.yaml"
IMAGENET_ROOT = WORKSPACE / "data" / "imagenet1k"


@dataclass(frozen=True)
class TextDataset:
    domain: str
    image_set: str
    ids: list[str]
    texts: list[str]
    answer_counts: list[int] | None = None
    metadata: dict[str, object] | None = None


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=1)
def _vqa_records() -> dict[str, tuple[dict[int, str], dict[int, str], dict[int, int]]]:
    """Load VQAv2 once and select one row per image for both E4 rules."""
    payload = _load_json(VQA_PATH)
    question_payload = _load_json(VQA_QUESTIONS_PATH)
    questions = {
        int(row["question_id"]): str(row["question"])
        for row in question_payload["questions"]
    }
    candidates_all: dict[int, list[dict]] = defaultdict(list)
    candidates_other: dict[int, list[dict]] = defaultdict(list)
    for row in payload["annotations"]:
        image_id = int(row["image_id"])
        candidates_all[image_id].append(row)
        if row.get("answer_type") == "other":
            candidates_other[image_id].append(row)

    def select(candidates: dict[int, list[dict]]):
        selected_answers: dict[int, str] = {}
        selected_questions: dict[int, str] = {}
        counts: dict[int, int] = {}
        for image_id, rows in candidates.items():
            row = min(rows, key=lambda item: int(item["question_id"]))
            question_id = int(row["question_id"])
            if question_id not in questions:
                raise RuntimeError(f"VQAv2 question text missing for question_id={question_id}")
            selected_answers[image_id] = str(row["multiple_choice_answer"])
            selected_questions[image_id] = questions[question_id]
            counts[image_id] = len(rows)
        return selected_answers, selected_questions, counts

    return {"other": select(candidates_other), "all": select(candidates_all)}


def _coerce_image_id(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value)
    digits = "".join(character for character in text if character.isdigit())
    if not digits:
        return None
    return int(digits[-12:])


def _source_coco_id(
    row: dict, source: dict, mapping: dict[str, object], raw_id: object,
) -> int | None:
    """Return a COCO ID only when the namespace/mapping is explicit."""
    explicit = row.get("coco_image_id", row.get("cocoImageId"))
    if explicit is not None:
        return _coerce_image_id(explicit)
    mapped = mapping.get(str(raw_id))
    if mapped is not None:
        return _coerce_image_id(mapped)
    if source.get("id_namespace") == "coco":
        return _coerce_image_id(raw_id)
    return None


def _eval_answer_records(source_config: Path) -> tuple[dict[int, str], dict[str, object]]:
    import yaml

    if not source_config.is_file():
        return {}, {"source_config": str(source_config), "available_sources": [], "matched_by_source": {}}
    with source_config.open("r", encoding="utf-8") as handle:
        sources = (yaml.safe_load(handle) or {}).get("sources", [])
    merged: dict[int, tuple[int, str, str]] = {}
    available: list[str] = []
    matched_by_source: dict[str, int] = {}
    for priority, source in enumerate(sources):
        name = str(source.get("name", f"source_{priority}"))
        kind = str(source.get("kind", ""))
        path = WORKSPACE / str(source.get("path", ""))
        if not path.is_file():
            continue
        available.append(name)
        payload = _load_json(path)
        mapping_path_value = source.get("coco_id_map")
        mapping_path = WORKSPACE / str(mapping_path_value) if mapping_path_value else None
        mapping_payload = _load_json(mapping_path) if mapping_path and mapping_path.is_file() else {}
        mapping = mapping_payload.get("mapping", mapping_payload) if isinstance(mapping_payload, dict) else {}
        records: list[tuple[int, str]] = []
        if kind == "vqav2":
            questions_path = WORKSPACE / str(source.get("questions", ""))
            questions_payload = _load_json(questions_path)
            questions = {
                int(row["question_id"]): row for row in questions_payload["questions"]
            }
            for row in payload["annotations"]:
                question = questions.get(int(row["question_id"]))
                if question is None:
                    continue
                raw_id = row.get("image_id")
                image_id = _source_coco_id(row, source, mapping, raw_id)
                if image_id is not None:
                    records.append((image_id, str(row["multiple_choice_answer"])))
        elif kind == "gqa":
            iterable = payload.values() if isinstance(payload, dict) else payload
            for row in iterable:
                raw_id = row.get("imageId", row.get("image_id"))
                image_id = _source_coco_id(row, source, mapping, raw_id)
                answer = row.get("answer")
                if image_id is not None and answer is not None:
                    records.append((image_id, str(answer)))
        elif kind == "textvqa":
            iterable = payload.get("data", payload if isinstance(payload, list) else [])
            for row in iterable:
                raw_id = row.get("image_id", row.get("imageId"))
                image_id = _source_coco_id(row, source, mapping, raw_id)
                answers = row.get("answers", [])
                answer = answers[0] if answers else row.get("answer")
                if image_id is not None and answer is not None:
                    records.append((image_id, str(answer)))
        else:
            raise ValueError(f"Unsupported eval-answer source kind {kind!r} in {source_config}")
        source_selected: dict[int, str] = {}
        for image_id, answer in records:
            source_selected.setdefault(image_id, answer)
        matched_by_source[name] = len(source_selected)
        for image_id, answer in source_selected.items():
            merged.setdefault(image_id, (priority, answer, name))
    return (
        {image_id: value[1] for image_id, value in merged.items()},
        {
            "source_config": str(source_config),
            "available_sources": available,
            "matched_by_source_before_coco_filter": matched_by_source,
            "_source_for_image": {image_id: value[2] for image_id, value in merged.items()},
        },
    )


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


def build_text_dataset(
    domain: str,
    image_set: str,
    *,
    seed: int = 0,
    eval_answer_sources: Path = EVAL_ANSWER_SOURCES,
) -> TextDataset:
    if domain == "imagenet" or image_set == "in1k10k":
        if domain != "imagenet" or image_set != "in1k10k":
            raise ValueError("ImageNet text requires domain=imagenet and image_set=in1k10k")
        ids, texts = _imagenet_records(seed=seed)
        return TextDataset(domain, image_set, ids, texts)

    captions, all_ids = _caption_records()
    vqa = _vqa_records()
    answers_other, questions_other, answer_counts = vqa["other"]
    answers_all, _questions_all, all_answer_counts = vqa["all"]
    common_ids = sorted(all_ids & set(answers_other))
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
    elif domain in {"answer", "answer_other"}:
        texts = [answers_other[image_id] for image_id in ids_int]
        counts = [answer_counts[image_id] for image_id in ids_int]
    elif domain == "question_other":
        texts = [questions_other[image_id] for image_id in ids_int]
        counts = [answer_counts[image_id] for image_id in ids_int]
    elif domain == "qa_concat":
        texts = [
            f"{questions_other[image_id]} {answers_other[image_id]}" for image_id in ids_int
        ]
        counts = [answer_counts[image_id] for image_id in ids_int]
    elif domain == "answer_all_types":
        texts = [answers_all[image_id] for image_id in ids_int]
        counts = [all_answer_counts[image_id] for image_id in ids_int]
    elif domain == "eval_answer":
        eval_answers, eval_metadata = _eval_answer_records(eval_answer_sources)
        matched_ids = [image_id for image_id in ids_int if image_id in eval_answers]
        source_for_image = eval_metadata.pop("_source_for_image", {})
        matched_by_source = {
            source: sum(source_for_image.get(image_id) == source for image_id in matched_ids)
            for source in eval_metadata.get("available_sources", [])
        }
        return TextDataset(
            domain,
            image_set,
            [str(value) for value in matched_ids],
            [eval_answers[image_id] for image_id in matched_ids],
            None,
            {
                **eval_metadata,
                "reference_N": len(ids_int),
                "matched_N": len(matched_ids),
                "coverage": len(matched_ids) / len(ids_int),
                "matched_by_source": matched_by_source,
                "coverage_by_source": {
                    source: count / len(ids_int) for source, count in matched_by_source.items()
                },
            },
        )
    else:
        raise ValueError(f"Unsupported text domain: {domain}")
    return TextDataset(
        domain,
        image_set,
        [str(value) for value in ids_int],
        texts,
        counts,
        {"reference_N": len(ids_int), "matched_N": len(ids_int), "coverage": 1.0},
    )


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
