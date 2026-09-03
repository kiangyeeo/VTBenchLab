from __future__ import annotations

import argparse
import io
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .utils import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_hash,
    load_config,
    read_jsonl,
    resolve_path,
    stable_seed,
)


@dataclass(frozen=True)
class Example:
    record_id: str
    domain: str
    image_id: str
    prompt: str
    answer: str
    image: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Example":
        return cls(**value)


def _sample_indices(length: int, count: int, seed: int) -> list[int]:
    if count > length:
        raise ValueError(f"Requested {count} rows from a source with only {length}")
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return indices[:count]


def _load_coco_rows(annotation_path: Path, image_root: Path) -> list[dict[str, Any]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in payload["images"]}
    first_caption: dict[int, tuple[int, str]] = {}
    for row in payload["annotations"]:
        image_id = int(row["image_id"])
        candidate = (int(row["id"]), str(row["caption"]).strip())
        if image_id not in first_caption or candidate[0] < first_caption[image_id][0]:
            first_caption[image_id] = candidate
    rows = []
    for image_id in sorted(set(images) & set(first_caption)):
        image_path = (image_root / images[image_id]["file_name"]).resolve()
        if not image_path.is_file():
            continue
        rows.append(
            {
                "image_id": str(image_id),
                "path": str(image_path),
                "caption": first_caption[image_id][1],
            }
        )
    return rows


def _caption_examples(
    rows: list[dict[str, Any]],
    indices: list[int],
    domain: str,
    prompt: str,
) -> list[Example]:
    return [
        Example(
            record_id=f"{domain}:coco:{rows[index]['image_id']}",
            domain=domain,
            image_id=f"coco:{rows[index]['image_id']}",
            prompt=prompt,
            answer=rows[index]["caption"],
            image={"kind": "file", "path": rows[index]["path"]},
        )
        for index in indices
    ]


def _arrow_examples(
    path: Path,
    count: int,
    seed: int,
    record_prefix: str,
) -> list[Example]:
    import pyarrow.ipc as ipc

    with path.open("rb") as handle:
        table = ipc.RecordBatchFileReader(handle).read_all()
    selected = _sample_indices(table.num_rows, count, seed)
    examples = []
    for index in selected:
        row = table.slice(index, 1).to_pydict()
        prompts = row["caption"][0]
        answers = row["answer"][0]
        prompt = str(prompts[0] if isinstance(prompts, list) else prompts).strip()
        answer = str(answers[0] if isinstance(answers, list) else answers).strip()
        raw_image_id = row["image_id"][0]
        if isinstance(raw_image_id, list):
            raw_image_id = raw_image_id[0]
        image_id = f"{record_prefix}:{raw_image_id}"
        examples.append(
            Example(
                record_id=f"vqa:{record_prefix}:{index}",
                domain="vqa",
                image_id=image_id,
                prompt=prompt,
                answer=answer,
                image={"kind": "arrow", "path": str(path), "row": index},
            )
        )
    return examples


def _majority_answer(answers: list[str]) -> str:
    cleaned = [str(answer).strip() for answer in answers if str(answer).strip()]
    if not cleaned:
        raise ValueError("Encountered an example with no non-empty answer")
    normalized = [answer.casefold() for answer in cleaned]
    counts = Counter(normalized)
    winner = max(counts, key=lambda item: (counts[item], -normalized.index(item)))
    return cleaned[normalized.index(winner)]


def _parquet_references(paths: list[Path]) -> list[tuple[Path, int]]:
    import pyarrow.parquet as pq

    references = []
    for path in paths:
        row_count = pq.ParquetFile(path).metadata.num_rows
        references.extend((path, index) for index in range(row_count))
    return references


def _textvqa_examples(paths: list[Path], count: int, seed: int) -> list[Example]:
    import pyarrow.parquet as pq

    references = _parquet_references(paths)
    selected = [references[index] for index in _sample_indices(len(references), count, seed)]
    tables = {
        path: pq.read_table(path, columns=["image_id", "question_id", "question", "answers"])
        for path in sorted({path for path, _ in selected})
    }
    examples = []
    for path, row_index in selected:
        row = tables[path].slice(row_index, 1).to_pydict()
        image_id = str(row["image_id"][0])
        question_id = int(row["question_id"][0])
        examples.append(
            Example(
                record_id=f"ocr:textvqa:{question_id}",
                domain="ocr",
                image_id=f"textvqa:{image_id}",
                prompt=(
                    "Answer using the text visible in the image. "
                    + str(row["question"][0]).strip()
                ),
                answer=_majority_answer(row["answers"][0]),
                image={"kind": "parquet", "path": str(path), "row": row_index},
            )
        )
    return examples


def _scienceqa_examples(paths: list[Path], count: int, seed: int) -> list[Example]:
    import pyarrow.parquet as pq

    references = _parquet_references(paths)
    selected = [references[index] for index in _sample_indices(len(references), count, seed)]
    tables = {
        path: pq.read_table(path, columns=["question", "choices", "answer", "hint"])
        for path in sorted({path for path, _ in selected})
    }
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    examples = []
    for ordinal, (path, row_index) in enumerate(selected):
        row = tables[path].slice(row_index, 1).to_pydict()
        choices = [str(choice).strip() for choice in row["choices"][0]]
        answer_index = int(row["answer"][0])
        if not 0 <= answer_index < len(choices):
            raise ValueError(f"Invalid ScienceQA answer index {answer_index}")
        choice_text = "\n".join(
            f"{letters[index]}. {choice}" for index, choice in enumerate(choices)
        )
        hint = str(row["hint"][0] or "").strip()
        hint_text = f"\nContext: {hint}" if hint else ""
        prompt = (
            f"{str(row['question'][0]).strip()}{hint_text}\n"
            f"Choices:\n{choice_text}\nAnswer with the correct choice."
        )
        examples.append(
            Example(
                record_id=f"reasoning:scienceqa:{ordinal}:{row_index}",
                domain="reasoning",
                image_id=f"scienceqa:{path.name}:{row_index}",
                prompt=prompt,
                answer=choices[answer_index],
                image={"kind": "parquet", "path": str(path), "row": row_index},
            )
        )
    return examples


def build_manifest(config: dict[str, Any], force: bool = False) -> Path:
    data_config = config["data"]
    artifact_root = resolve_path(config, config["runtime"]["artifact_root"])
    manifest_dir = artifact_root / "manifest"
    records_path = manifest_dir / "records.jsonl"
    metadata_path = manifest_dir / "manifest.json"
    data_fingerprint = canonical_hash(data_config)
    if metadata_path.is_file() and records_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("data_fingerprint") == data_fingerprint:
            print(f"Reusing manifest: {records_path}")
            return records_path
        raise RuntimeError(
            f"Existing manifest at {manifest_dir} has a different data fingerprint; "
            "use --force or a different artifact_root"
        )

    base_seed = int(data_config["sample_seed"])
    caption_prompt = str(data_config["caption_prompt"])
    train_rows = _load_coco_rows(
        resolve_path(config, data_config["coco_train_annotations"]),
        resolve_path(config, data_config["coco_train_images"]),
    )
    warmup_count = int(data_config["counts"]["warmup"])
    generic_count = int(data_config["counts"]["generic"])
    train_order = _sample_indices(
        len(train_rows), warmup_count + generic_count, stable_seed(base_seed, "coco_train")
    )
    examples = _caption_examples(
        train_rows, train_order[:warmup_count], "warmup", caption_prompt
    )
    examples += _caption_examples(
        train_rows, train_order[warmup_count:], "generic", caption_prompt
    )

    val_rows = _load_coco_rows(
        resolve_path(config, data_config["coco_val_annotations"]),
        resolve_path(config, data_config["coco_val_images"]),
    )
    caption_indices = _sample_indices(
        len(val_rows),
        int(data_config["counts"]["caption"]),
        stable_seed(base_seed, "coco_val"),
    )
    examples += _caption_examples(val_rows, caption_indices, "caption", caption_prompt)

    vqa_count = int(data_config["counts"]["vqa"])
    if vqa_count:
        existence_count = vqa_count // 2
        counting_count = vqa_count - existence_count
        examples += _arrow_examples(
            resolve_path(config, data_config["coco_existence_arrow"]),
            existence_count,
            stable_seed(base_seed, "coco_existence"),
            "existence",
        )
        examples += _arrow_examples(
            resolve_path(config, data_config["coco_counting_arrow"]),
            counting_count,
            stable_seed(base_seed, "coco_counting"),
            "counting",
        )

    ocr_count = int(data_config["counts"]["ocr"])
    if ocr_count:
        textvqa_paths = sorted(
            resolve_path(config, data_config["textvqa_dir"]).glob("*.parquet")
        )
        if not textvqa_paths:
            raise FileNotFoundError(f"No TextVQA parquet files in {data_config['textvqa_dir']}")
        examples += _textvqa_examples(
            textvqa_paths,
            ocr_count,
            stable_seed(base_seed, "textvqa"),
        )

    reasoning_count = int(data_config["counts"]["reasoning"])
    if reasoning_count:
        scienceqa_paths = sorted(
            resolve_path(config, data_config["scienceqa_dir"]).glob("*.parquet")
        )
        if not scienceqa_paths:
            raise FileNotFoundError(
                f"No ScienceQA parquet files in {data_config['scienceqa_dir']}"
            )
        examples += _scienceqa_examples(
            scienceqa_paths,
            reasoning_count,
            stable_seed(base_seed, "scienceqa"),
        )

    record_ids = [example.record_id for example in examples]
    if len(record_ids) != len(set(record_ids)):
        duplicates = [key for key, count in Counter(record_ids).items() if count > 1]
        raise RuntimeError(f"Duplicate manifest record IDs: {duplicates[:10]}")
    domain_counts = Counter(example.domain for example in examples)
    atomic_write_jsonl(records_path, (asdict(example) for example in examples))
    atomic_write_json(
        metadata_path,
        {
            "schema_version": 1,
            "data_fingerprint": data_fingerprint,
            "records_sha256": canonical_hash([asdict(example) for example in examples]),
            "record_count": len(examples),
            "domain_counts": dict(sorted(domain_counts.items())),
            "sample_seed": base_seed,
        },
    )
    print(f"Wrote {len(examples)} records to {records_path}: {dict(domain_counts)}")
    return records_path


def load_examples(config: dict[str, Any]) -> list[Example]:
    root = resolve_path(config, config["runtime"]["artifact_root"])
    return [Example.from_dict(row) for row in read_jsonl(root / "manifest" / "records.jsonl")]


class ImageResolver:
    def __init__(self) -> None:
        self._arrow_tables: dict[str, Any] = {}
        self._parquet_tables: dict[str, Any] = {}

    def load(self, example: Example) -> Image.Image:
        locator = example.image
        kind = locator["kind"]
        if kind == "file":
            with Image.open(locator["path"]) as image:
                return image.convert("RGB")
        if kind == "arrow":
            import pyarrow.ipc as ipc

            path = locator["path"]
            if path not in self._arrow_tables:
                with Path(path).open("rb") as handle:
                    self._arrow_tables[path] = ipc.RecordBatchFileReader(handle).read_all()
            payload = self._arrow_tables[path]["image"][int(locator["row"])].as_py()
            return Image.open(io.BytesIO(payload)).convert("RGB")
        if kind == "parquet":
            import pyarrow.parquet as pq

            path = locator["path"]
            if path not in self._parquet_tables:
                self._parquet_tables[path] = pq.read_table(path, columns=["image"])
            payload = self._parquet_tables[path]["image"][int(locator["row"])].as_py()
            image_bytes = payload["bytes"] if isinstance(payload, dict) else payload
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        raise ValueError(f"Unsupported image locator kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed PE-pair probe manifest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    build_manifest(config, force=args.force)


if __name__ == "__main__":
    main()
