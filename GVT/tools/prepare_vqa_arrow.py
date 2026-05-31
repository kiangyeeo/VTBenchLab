#!/usr/bin/env python3
"""Build the VQAv2 validation Arrow used by GVT evaluation."""

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm


QUESTION_FILE = "v2_OpenEnded_mscoco_val2014_questions.json"
ANNOTATION_FILE = "v2_mscoco_val2014_annotations.json"

MANUAL_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
ARTICLES = {"a", "an", "the"}
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
PUNCT = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def normalize_answer(answer):
    token = answer
    for punc in PUNCT:
        if (punc + " " in token or " " + punc in token) or re.search(COMMA_STRIP, token):
            token = token.replace(punc, "")
        else:
            token = token.replace(punc, " ")
    token = PERIOD_STRIP.sub("", token, re.UNICODE)
    words = []
    for word in token.lower().split():
        word = MANUAL_MAP.get(word, word)
        if word not in ARTICLES:
            words.append(word)
    return " ".join(words).replace(",", "")


def vqa_score(count):
    if count == 0:
        return 0.0
    if count == 1:
        return 0.3
    if count == 2:
        return 0.6
    if count == 3:
        return 0.9
    return 1.0


def read_json(path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def read_image(path):
    with path.open("rb") as fp:
        return fp.read()


def find_val2014_dir(vqa_root, coco_val2014=None):
    if coco_val2014 is not None:
        if coco_val2014.is_dir():
            return coco_val2014
        raise FileNotFoundError(f"--coco-val2014 is not a directory: {coco_val2014}")

    candidates = [
        vqa_root / "val2014",
        vqa_root / "coco" / "val2014",
        vqa_root / "COCO" / "val2014",
        vqa_root / "images" / "val2014",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find COCO val2014 images. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def write_arrow(rows, output_path):
    import pyarrow as pa

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(output_path), "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)


def copy_eval_gt(vqa_root, save_dir):
    eval_gt = save_dir / "eval_gt"
    eval_gt.mkdir(parents=True, exist_ok=True)
    for filename in [QUESTION_FILE, ANNOTATION_FILE]:
        shutil.copy2(vqa_root / filename, eval_gt / filename)


def build_rows(vqa_root, coco_val2014=None):
    questions = read_json(vqa_root / QUESTION_FILE)["questions"]
    annotations = read_json(vqa_root / ANNOTATION_FILE)["annotations"]
    val2014_dir = find_val2014_dir(vqa_root, coco_val2014=coco_val2014)

    answer_vocab = sorted(
        {
            normalize_answer(answer["answer"])
            for annotation in annotations
            for answer in annotation["answers"]
        }
    )
    answer2label = {answer: idx for idx, answer in enumerate(answer_vocab)}

    by_image = defaultdict(dict)
    for question in questions:
        by_image[question["image_id"]][question["question_id"]] = {
            "question": question["question"],
        }

    for annotation in annotations:
        image_id = annotation["image_id"]
        question_id = annotation["question_id"]
        answer_counts = Counter(normalize_answer(a["answer"]) for a in annotation["answers"])
        answers = []
        labels = []
        scores = []
        for answer, count in sorted(answer_counts.items()):
            answers.append(answer)
            labels.append(answer2label[answer])
            scores.append(vqa_score(count))

        if image_id in by_image and question_id in by_image[image_id]:
            by_image[image_id][question_id].update(
                {
                    "answers": answers,
                    "answer_labels": labels,
                    "answer_scores": scores,
                }
            )

    rows = []
    missing = []
    for image_id in tqdm(sorted(by_image), desc="vqav2_rest_val"):
        qas = []
        for question_id, qa in sorted(by_image[image_id].items()):
            if qa.get("answer_scores") and sum(qa["answer_scores"]) > 0:
                qas.append((question_id, qa))

        if not qas:
            continue

        image_path = val2014_dir / f"COCO_val2014_{int(image_id):012d}.jpg"
        if not image_path.is_file():
            missing.append(str(image_path))
            continue

        rows.append(
            {
                "image": read_image(image_path),
                "questions": [qa["question"] for _, qa in qas],
                "answers": [qa["answers"] for _, qa in qas],
                "answer_labels": [qa["answer_labels"] for _, qa in qas],
                "answer_scores": [qa["answer_scores"] for _, qa in qas],
                "image_id": int(image_id),
                "question_id": [question_id for question_id, _ in qas],
                "split": "val",
            }
        )

    if missing:
        examples = "\n  ".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} VQAv2 image files were not found. Examples:\n  {examples}"
        )

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VQAv2 validation data into vqav2_rest_val.arrow.")
    parser.add_argument("--vqa-root", required=True, type=Path, help="Directory with VQAv2 val json files and val2014/.")
    parser.add_argument("--coco-val2014", type=Path, help="Optional explicit COCO val2014 image directory.")
    parser.add_argument("--save-dir", required=True, type=Path, help="Directory to write Arrow and eval_gt files.")
    return parser.parse_args()


def check_dependencies():
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Missing dependency: pyarrow. Install it before converting data, e.g. `pip install pyarrow`.") from exc


def main():
    check_dependencies()
    args = parse_args()
    for filename in [QUESTION_FILE, ANNOTATION_FILE]:
        if not (args.vqa_root / filename).is_file():
            raise FileNotFoundError(f"Missing VQAv2 file: {args.vqa_root / filename}")

    print("Preparing VQAv2 validation Arrow. This reads the VQA json and COCO val2014 images.", flush=True)
    rows = build_rows(args.vqa_root, coco_val2014=args.coco_val2014)
    output_path = args.save_dir / "vqav2_rest_val.arrow"
    write_arrow(rows, output_path)
    copy_eval_gt(args.vqa_root, args.save_dir)
    print(f"saved {len(rows)} rows to {output_path}")
    print(f"copied VQA eval annotations to {args.save_dir / 'eval_gt'}")


if __name__ == "__main__":
    main()
