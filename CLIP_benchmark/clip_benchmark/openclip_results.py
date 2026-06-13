"""Export CLIP benchmark outputs in an OpenCLIP-results-like wide table."""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict


OPENCLIP_DATASET_COLUMNS = [
    "ImageNet 1k",
    "Caltech-101",
    "CIFAR-10",
    "CIFAR-100",
    "CLEVR Counts",
    "CLEVR Distance",
    "Country211",
    "Describable Textures",
    "EuroSAT",
    "FGVC Aircraft",
    "Food-101",
    "GTSRB",
    "ImageNet Sketch",
    "ImageNet v2",
    "ImageNet-A",
    "ImageNet-O",
    "ImageNet-R",
    "KITTI Vehicle Distance",
    "MNIST",
    "ObjectNet",
    "Oxford Flowers-102",
    "Oxford-IIIT Pet",
    "Pascal VOC 2007",
    "PatchCamelyon",
    "Rendered SST2",
    "RESISC45",
    "Stanford Cars",
    "STL-10",
    "SUN397",
    "SVHN",
    "Flickr",
    "MSCOCO",
    "WinoGAViL",
    "iWildCam",
    "Camelyon17",
    "FMoW",
    "Dollar Street",
    "GeoDE",
]


DATASET_TO_OPENCLIP_COLUMN = {
    "imagenet1k": "ImageNet 1k",
    "caltech101": "Caltech-101",
    "vtab/caltech101": "Caltech-101",
    "cifar10": "CIFAR-10",
    "vtab/cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "vtab/cifar100": "CIFAR-100",
    "clevr_count_all": "CLEVR Counts",
    "vtab/clevr_count_all": "CLEVR Counts",
    "clevr_closest_object_distance": "CLEVR Distance",
    "vtab/clevr_closest_object_distance": "CLEVR Distance",
    "country211": "Country211",
    "dtd": "Describable Textures",
    "vtab/dtd": "Describable Textures",
    "eurosat": "EuroSAT",
    "vtab/eurosat": "EuroSAT",
    "fgvc_aircraft": "FGVC Aircraft",
    "food101": "Food-101",
    "gtsrb": "GTSRB",
    "imagenet_sketch": "ImageNet Sketch",
    "imagenetv2": "ImageNet v2",
    "imagenet-a": "ImageNet-A",
    "imagenet-o": "ImageNet-O",
    "imagenet-r": "ImageNet-R",
    "kitti": "KITTI Vehicle Distance",
    "vtab/kitti_closest_vehicle_distance": "KITTI Vehicle Distance",
    "mnist": "MNIST",
    "objectnet": "ObjectNet",
    "flowers102": "Oxford Flowers-102",
    "vtab/flowers": "Oxford Flowers-102",
    "pets": "Oxford-IIIT Pet",
    "oxford_iiit_pet": "Oxford-IIIT Pet",
    "vtab/pets": "Oxford-IIIT Pet",
    "voc2007": "Pascal VOC 2007",
    "voc2007_multilabel": "Pascal VOC 2007",
    "pcam": "PatchCamelyon",
    "vtab/pcam": "PatchCamelyon",
    "renderedsst2": "Rendered SST2",
    "resisc45": "RESISC45",
    "vtab/resisc45": "RESISC45",
    "cars": "Stanford Cars",
    "stl10": "STL-10",
    "sun397": "SUN397",
    "svhn": "SVHN",
    "vtab/svhn": "SVHN",
    "flickr30k": "Flickr",
    "flickr8k": "Flickr",
    "mscoco_captions": "MSCOCO",
    "winogavil": "WinoGAViL",
    "iwildcam": "iWildCam",
    "camelyon17": "Camelyon17",
    "fmow": "FMoW",
    "dollar_street": "Dollar Street",
    "geode": "GeoDE",
}


DATASET_PRIORITY = defaultdict(
    lambda: 100,
    {
        # The OpenCLIP reference table uses canonical CIFAR/VOC/Flickr columns.
        # VTAB or alternate variants are useful fallbacks but should not replace
        # a more direct dataset if both are present.
        "vtab/cifar10": 50,
        "vtab/cifar100": 50,
        "voc2007": 80,
        "voc2007_multilabel": 100,
        "flickr8k": 40,
        "flickr30k": 100,
    },
)


def _expand_inputs(inputs):
    paths = []
    for item in inputs:
        matches = sorted(glob.glob(item))
        if matches:
            paths.extend(matches)
        else:
            paths.append(item)
    return paths


def _normalize_dataset_name(dataset):
    dataset = dataset.strip()
    if dataset.startswith("wds/"):
        dataset = dataset[4:]
    return dataset


def _metric_value(row):
    if row.get("acc1"):
        return float(row["acc1"])
    if row.get("mean_average_precision"):
        return float(row["mean_average_precision"])

    image_recall = row.get("image_retrieval_recall@5")
    text_recall = row.get("text_retrieval_recall@5")
    if image_recall and text_recall:
        return (float(image_recall) + float(text_recall)) / 2.0
    if image_recall:
        return float(image_recall)
    if text_recall:
        return float(text_recall)

    return None


def _read_json(path):
    with open(path) as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    row = {
        "dataset": data.get("dataset", ""),
        "model": data.get("model", ""),
        "pretrained": data.get("pretrained", ""),
    }
    row.update(metrics)
    return [row]


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_rows(paths):
    rows = []
    for path in paths:
        if os.path.isdir(path):
            nested = sorted(glob.glob(os.path.join(path, "*.json")))
            rows.extend(_read_rows(nested))
        elif path.endswith(".json"):
            rows.extend(_read_json(path))
        elif path.endswith(".csv"):
            rows.extend(_read_csv(path))
        else:
            raise ValueError(f"Unsupported input file: {path}")
    return rows


def _read_reference(path):
    if not path:
        return {}, OPENCLIP_DATASET_COLUMNS

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = [
            field
            for field in reader.fieldnames
            if field not in {"name", "pretrained", "params (M)", "FLOPs (B)"}
            and not field.startswith("Average perf.")
        ]

    meta = {
        (row["name"], row["pretrained"]): {
            "params (M)": row.get("params (M)", ""),
            "FLOPs (B)": row.get("FLOPs (B)", ""),
        }
        for row in rows
    }
    return meta, columns


def build_openclip_rows(rows, reference_meta=None, dataset_columns=None):
    reference_meta = reference_meta or {}
    dataset_columns = dataset_columns or OPENCLIP_DATASET_COLUMNS
    grouped = defaultdict(dict)

    for row in rows:
        model = row.get("model") or row.get("name")
        pretrained = row.get("pretrained")
        dataset = _normalize_dataset_name(row.get("dataset", ""))
        column = DATASET_TO_OPENCLIP_COLUMN.get(dataset)
        value = _metric_value(row)

        if not model or not pretrained or not column or value is None:
            continue

        existing = grouped[(model, pretrained)].get(column)
        priority = DATASET_PRIORITY[dataset]
        if existing is None or priority >= existing[1]:
            grouped[(model, pretrained)][column] = (value, priority)

    output_rows = []
    for (model, pretrained), values in sorted(grouped.items()):
        meta = reference_meta.get((model, pretrained), {})
        available = [values[column][0] for column in dataset_columns if column in values]
        avg = sum(available) / len(available) if available else None
        out = {
            "name": model,
            "pretrained": pretrained,
            "params (M)": meta.get("params (M)", ""),
            "FLOPs (B)": meta.get("FLOPs (B)", ""),
            "Average perf. on available datasets": "" if avg is None else f"{avg:.4f}",
        }
        for column in dataset_columns:
            stored = values.get(column)
            value = None if stored is None else stored[0]
            out[column] = "" if value is None else f"{value:.4f}"
        output_rows.append(out)

    return output_rows


def main():
    parser = argparse.ArgumentParser(
        description="Convert clip_benchmark JSON/CSV outputs to an OpenCLIP-style wide CSV."
    )
    parser.add_argument("inputs", nargs="+", help="Input JSON/CSV files, directories, or glob patterns.")
    parser.add_argument("--output", required=True, help="Path to write the wide CSV.")
    parser.add_argument(
        "--reference-openclip-csv",
        default=None,
        help="Optional official openclip_results.csv to borrow column order, params, and FLOPs.",
    )
    args = parser.parse_args()

    input_paths = _expand_inputs(args.inputs)
    rows = _read_rows(input_paths)
    reference_meta, dataset_columns = _read_reference(args.reference_openclip_csv)
    output_rows = build_openclip_rows(rows, reference_meta, dataset_columns)

    fieldnames = [
        "name",
        "pretrained",
        "params (M)",
        "FLOPs (B)",
        "Average perf. on available datasets",
    ] + dataset_columns

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
