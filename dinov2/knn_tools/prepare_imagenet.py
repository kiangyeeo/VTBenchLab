#!/usr/bin/env python
"""Download ImageNet-1k via a HuggingFace mirror and lay it out exactly the way
DINOv2's `ImageNet` dataset class expects.

DINOv2 (dinov2/data/datasets/image_net.py) reconstructs image paths from parsed
filenames, so the on-disk names MUST follow these conventions:

    <ROOT>/train/<cls>/<cls>_<idx>.JPEG          # idx is a plain int, NOT zero-padded
    <ROOT>/val/<cls>/ILSVRC2012_val_<idx08d>.JPEG # idx is 8-digit zero-padded
    <ROOT>/labels.txt                             # one "class_id,class_name" row per class

Class folders are named class_0000 .. class_0999 so that ImageFolder's sorted
ordering equals the HF integer label ordering (canonical ImageNet index order).
k-NN only needs train/val labels to be consistent integers, which this guarantees.

Raw original JPEG bytes are written without re-encoding (decode=False).

Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    huggingface-cli login          # once, if the repo is gated
    python knn_tools/prepare_imagenet.py --out /cache/ma-user/VTBenchLab/data/imagenet1k
"""
import argparse
import csv
import os

from datasets import Image, load_dataset
from tqdm import tqdm

NUM_CLASSES = 1000


def cls_dir(label: int) -> str:
    return f"class_{label:04d}"


def write_labels(out_root: str, class_names) -> None:
    path = os.path.join(out_root, "labels.txt")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for i in range(NUM_CLASSES):
            # csv.writer quotes names that contain commas (e.g. "tench, Tinca tinca")
            w.writerow([cls_dir(i), class_names[i]])
    print(f">> wrote {path}")


def export_split(ds, out_root: str, split_dir: str):
    base = os.path.join(out_root, split_dir)
    for i in range(NUM_CLASSES):
        os.makedirs(os.path.join(base, cls_dir(i)), exist_ok=True)

    per_class_counter = [0] * NUM_CLASSES  # for train file naming
    for global_idx, ex in enumerate(tqdm(ds, desc=f"writing {split_dir}")):
        label = int(ex["label"])
        img = ex["image"]  # decode=False -> {"bytes": ..., "path": ...}
        data = img["bytes"]
        cdir = cls_dir(label)
        if split_dir == "train":
            idx = per_class_counter[label]
            per_class_counter[label] += 1
            fname = f"{cdir}_{idx}.JPEG"
        else:  # val
            fname = f"ILSVRC2012_val_{global_idx:08d}.JPEG"
        with open(os.path.join(base, cdir, fname), "wb") as f:
            f.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output ImageNet root dir")
    ap.add_argument("--repo", default="ILSVRC/imagenet-1k",
                    help="HF dataset repo id (use a non-gated mirror to skip login)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    choices=["train", "val"])
    ap.add_argument("--num-proc", type=int, default=8,
                    help="parallel download/extract workers for load_dataset")
    args = ap.parse_args()

    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f">> HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")

    os.makedirs(args.out, exist_ok=True)

    # HF uses "validation" as the split name for the 50k val set.
    split_map = {"train": "train", "val": "validation"}
    class_names = None
    for split in args.splits:
        print(f"\n>> loading {args.repo} split={split_map[split]} ...")
        ds = load_dataset(args.repo, split=split_map[split], num_proc=args.num_proc)
        if class_names is None:
            class_names = ds.features["label"].names  # index-ordered human names
        ds = ds.cast_column("image", Image(decode=False))  # raw bytes, no re-encode
        export_split(ds, args.out, split)

    if class_names is not None:
        write_labels(args.out, class_names)
    print("\n>> ImageNet preparation complete.")


if __name__ == "__main__":
    main()
