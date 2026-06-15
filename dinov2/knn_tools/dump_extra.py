#!/usr/bin/env python
"""Generate the `extra` metadata (.npy) files DINOv2's ImageNet class needs:
    entries-TRAIN.npy / entries-VAL.npy
    class-ids-TRAIN.npy / class-ids-VAL.npy
    class-names-TRAIN.npy / class-names-VAL.npy

Run this AFTER prepare_imagenet.py has produced the ImageFolder layout + labels.txt.

Usage:
    PYTHONPATH=. python knn_tools/dump_extra.py \
        --root  /cache/ma-user/VTBenchLab/data/imagenet1k \
        --extra /cache/ma-user/VTBenchLab/data/imagenet1k/extra
"""
import argparse

from dinov2.data.datasets import ImageNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--extra", required=True)
    args = ap.parse_args()

    for split in (ImageNet.Split.TRAIN, ImageNet.Split.VAL):
        print(f">> dumping extra for split={split.value} ...")
        ds = ImageNet(split=split, root=args.root, extra=args.extra)
        ds.dump_extra()
    print(">> done. extra files written to", args.extra)


if __name__ == "__main__":
    main()
