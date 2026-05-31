#!/usr/bin/env python3
"""Build COCO Karpathy validation Arrow and caption evaluation ground truth."""

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def read_json(path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def read_image(path):
    with path.open("rb") as fp:
        return fp.read()


def parse_coco_id(entry):
    if "cocoid" in entry:
        return int(entry["cocoid"])
    return int(Path(entry["filename"]).stem.split("_")[-1])


def build_image_index(coco_root):
    index = {}
    for split in ["train2014", "val2014"]:
        image_dir = coco_root / split
        if image_dir.is_dir():
            for path in image_dir.glob("*.jpg"):
                index[path.name] = path
    return index


def write_arrow(rows, output_path):
    import pyarrow as pa

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(output_path), "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)


def write_caption_gt(entries, output_path):
    images = []
    annotations = []
    annotation_id = 0

    for entry in entries:
        image_id = parse_coco_id(entry)
        images.append({"id": image_id, "file_name": entry["filename"]})
        for sentence in entry["sentences"]:
            annotation_id += 1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "caption": sentence["raw"],
                }
            )

    output = {
        "info": {},
        "licenses": [],
        "type": "captions",
        "images": images,
        "annotations": annotations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(output, fp)


def build_rows(coco_root, karpathy_json):
    data = read_json(karpathy_json)["images"]
    image_index = build_image_index(coco_root)

    val_entries = [entry for entry in data if entry["split"] == "val"]
    rows = []
    missing = []

    for entry in tqdm(val_entries, desc="coco_caption_karpathy_val"):
        filename = entry["filename"]
        image_path = image_index.get(filename)
        if image_path is None:
            missing.append(filename)
            continue

        rows.append(
            {
                "image": read_image(image_path),
                "caption": [sentence["raw"] for sentence in entry["sentences"]],
                "image_id": filename,
                "split": "val",
            }
        )

    if missing:
        examples = "\n  ".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} COCO Karpathy val images were not found under train2014/ or val2014/. Examples:\n  {examples}"
        )

    return rows, val_entries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert COCO Karpathy val split into coco_caption_karpathy_val.arrow."
    )
    parser.add_argument("--coco-root", required=True, type=Path, help="COCO root containing train2014/ and val2014/.")
    parser.add_argument("--karpathy-json", required=True, type=Path, help="Path to Karpathy dataset_coco.json.")
    parser.add_argument("--save-dir", required=True, type=Path, help="Directory to write Arrow and eval_gt files.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows, val_entries = build_rows(args.coco_root, args.karpathy_json)

    arrow_path = args.save_dir / "coco_caption_karpathy_val.arrow"
    gt_path = args.save_dir / "eval_gt" / "coco_karpathy_val_gt.json"

    write_arrow(rows, arrow_path)
    write_caption_gt(val_entries, gt_path)
    print(f"saved {len(rows)} rows to {arrow_path}")
    print(f"saved caption eval ground truth to {gt_path}")


if __name__ == "__main__":
    main()
