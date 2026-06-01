#!/usr/bin/env python3
"""Build Arrow files for GVTBench OC/MCI evaluation annotations."""

import argparse
import json
from pathlib import Path

from tqdm import tqdm


ANNOTATION_FILES = {
    "coco_oc": "coco_oc.json",
    "coco_mci": "coco_mci.json",
    "vcr_oc": "vcr_oc.json",
    "vcr_mci": "vcr_mci.json",
}


def read_json(path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def read_image(path):
    with path.open("rb") as fp:
        return fp.read()


def coco_image_path(image_id, coco_val2017):
    return coco_val2017 / f"{int(image_id):012d}.jpg"


def vcr_image_path(image_id, vcr_root):
    image_path = Path(str(image_id))
    if image_path.parts and image_path.parts[0] == "vcr1images":
        return vcr_root / image_path
    return vcr_root / "vcr1images" / image_path


def write_arrow(rows, output_path):
    import pyarrow as pa

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(output_path), "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)


def build_rows(tasks, task_name, coco_val2017, vcr_root):
    rows = []
    missing = []

    for task in tqdm(tasks, desc=task_name):
        if task_name.startswith("coco_"):
            image_path = coco_image_path(task["image_id"], coco_val2017)
        else:
            image_path = vcr_image_path(task["image_id"], vcr_root)

        if not image_path.is_file():
            missing.append(str(image_path))
            if len(missing) >= 10:
                continue
            continue

        rows.append(
            {
                "image": read_image(image_path),
                "caption": [task["text_in"]],
                "answer": [task["text_out"]],
                "image_id": [task["image_id"]],
                "n_obj_exist": [task["n_obj_exist"]],
            }
        )

    if missing:
        examples = "\n  ".join(missing[:10])
        raise FileNotFoundError(
            f"{task_name}: {len(missing)} image files were not found. Examples:\n  {examples}"
        )

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert GVTBench coco/vcr OC and MCI annotations into Arrow files."
    )
    parser.add_argument("--anno-dir", required=True, type=Path, help="Directory with GVTBench json files.")
    parser.add_argument("--coco-val2017", type=Path, help="COCO val2017 image directory.")
    parser.add_argument("--vcr-root", type=Path, help="VCR root directory containing vcr1images/.")
    parser.add_argument("--save-dir", required=True, type=Path, help="Directory to write Arrow files.")
    parser.add_argument(
        "--tasks",
        default=",".join(ANNOTATION_FILES),
        help="Comma-separated task names to convert. Choices: %s" % ", ".join(ANNOTATION_FILES),
    )
    return parser.parse_args()


def check_dependencies():
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Missing dependency: pyarrow. Install it before converting data, e.g. `pip install pyarrow`.") from exc


def main():
    check_dependencies()
    args = parse_args()

    selected_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown_tasks = sorted(set(selected_tasks) - set(ANNOTATION_FILES))
    if unknown_tasks:
        raise ValueError("Unknown tasks: %s" % ", ".join(unknown_tasks))

    if any(task.startswith("coco_") for task in selected_tasks) and args.coco_val2017 is None:
        raise ValueError("--coco-val2017 is required for COCO GVTBench tasks.")
    if any(task.startswith("vcr_") for task in selected_tasks) and args.vcr_root is None:
        raise ValueError("--vcr-root is required for VCR GVTBench tasks.")

    for name in selected_tasks:
        filename = ANNOTATION_FILES[name]
        anno_path = args.anno_dir / filename
        if not anno_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {anno_path}")

        tasks = read_json(anno_path)
        rows = build_rows(tasks, name, args.coco_val2017, args.vcr_root)
        output_path = args.save_dir / f"{name}.arrow"
        write_arrow(rows, output_path)
        print(f"saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
