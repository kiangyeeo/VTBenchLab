#!/usr/bin/env python
"""Generic k-NN eval for non-DINOv2 backbones loadable via timm (MAE / DINO / SEER ...).

Reuses DINOv2's k-NN machinery (dinov2.eval.knn.eval_knn_with_model) on the same
ImageNet-1k layout + same protocol (nb_knn, temperature, eval transform), but swaps
the backbone for any timm model so results are directly comparable to the DINOv2 runs.

Run under torchrun (single GPU shown):

    export HF_ENDPOINT=https://hf-mirror.com    # weights are pulled from HF
    DATA=/cache/ma-user/VTBenchLab/data/imagenet1k
    PYTHONPATH=. torchrun --nproc_per_node=1 knn_tools/run_knn_timm.py \
        --model vit_base_patch16_224.dino \
        --output-dir /cache/ma-user/VTBenchLab/outputs/knn_dino_vitb16 \
        --train-dataset "ImageNet:split=TRAIN:root=$DATA:extra=$DATA/extra" \
        --val-dataset   "ImageNet:split=VAL:root=$DATA:extra=$DATA/extra"
"""
import argparse
import logging
import os

import timm
import torch

import dinov2.distributed as distributed
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.eval.knn import eval_knn_with_model
from dinov2.eval.metrics import AccuracyAveraging
from dinov2.logging import setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="timm tag, e.g. vit_base_patch16_224.mae")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train-dataset", dest="train_dataset_str", required=True)
    ap.add_argument("--val-dataset", dest="val_dataset_str", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=5)
    ap.add_argument("--global-pool", default=None,
                    help="override timm global_pool: 'token' (CLS) or 'avg'. "
                         "Leave unset to use the model's default.")
    ap.add_argument("--nb_knn", nargs="+", type=int, default=[10, 20, 100, 200])
    ap.add_argument("--temperature", type=float, default=0.07)
    args = ap.parse_args()

    distributed.enable(overwrite=True)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    # num_classes=0 -> timm returns pooled features [B, D] (CLS for ViT, avgpool for RegNet).
    kw = {}
    if args.global_pool is not None:
        kw["global_pool"] = args.global_pool
    model = timm.create_model(args.model, pretrained=True, num_classes=0, **kw)
    model = model.cuda().eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded timm model '{args.model}'  params={n_params/1e6:.1f}M")

    # Same eval transform as the DINOv2 runs (resize 256 -> center-crop 224, imagenet norm)
    # so the comparison across models is apples-to-apples.
    transform = make_classification_eval_transform()

    eval_knn_with_model(
        model=model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        nb_knn=tuple(args.nb_knn),
        temperature=args.temperature,
        autocast_dtype=torch.float,
        accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
        transform=transform,
        gather_on_cpu=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_per_class_list=[-1],
        n_tries=1,
    )


if __name__ == "__main__":
    main()
