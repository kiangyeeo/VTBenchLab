#!/usr/bin/env python
"""Linear-probing eval for timm ViT backbones (e.g. the original DINO) using the
SAME DINOv2 linear protocol as dinov2/eval/linear.py:

    - frozen backbone
    - 52 linear heads in parallel: {1,4} last blocks x {avgpool on/off} x 13 LRs
    - best classifier on val is reported

timm's `get_intermediate_layers` has a different signature than DINOv2's, so we wrap
the model in a tiny adapter exposing `get_intermediate_layers(x, n, return_class_token)`
that returns per-block (patch_tokens, cls_token) tuples, exactly what DINOv2 expects.

Run under torchrun (single GPU shown):

    export HF_ENDPOINT=https://hf-mirror.com
    DATA=/cache/ma-user/VTBenchLab/data/imagenet1k
    PYTHONPATH=. torchrun --nproc_per_node=1 knn_tools/run_linear_timm.py \
        --model vit_base_patch16_224.dino \
        --output-dir /cache/ma-user/VTBenchLab/outputs/linear_dino_vitb16 \
        --train-dataset "ImageNet:split=TRAIN:root=$DATA:extra=$DATA/extra" \
        --val-dataset   "ImageNet:split=VAL:root=$DATA:extra=$DATA/extra"
"""
import argparse
import logging
import os

import timm
import torch
import torch.nn as nn

import dinov2.distributed as distributed
from dinov2.eval.linear import run_eval_linear
from dinov2.eval.metrics import MetricType
from dinov2.logging import setup_logging

DEFAULT_LRS = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]


class DinoV2IntermediateAdapter(nn.Module):
    """Expose a DINOv2-compatible get_intermediate_layers() on a timm ViT."""

    def __init__(self, timm_model):
        super().__init__()
        self.model = timm_model

    def get_intermediate_layers(self, x, n, return_class_token=False):
        outs = self.model.get_intermediate_layers(x, n, return_prefix_tokens=True, norm=True)
        # outs: list of (patch_tokens [B,N,D], prefix_tokens [B,P,D]); CLS = prefix[:,0]
        if return_class_token:
            return tuple((patch, prefix[:, 0]) for patch, prefix in outs)
        return tuple(patch for patch, _ in outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="timm ViT tag, e.g. vit_base_patch16_224.dino")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train-dataset", dest="train_dataset_str", required=True)
    ap.add_argument("--val-dataset", dest="val_dataset_str", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--epoch-length", type=int, default=1250)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--eval-period-iterations", type=int, default=1250)
    ap.add_argument("--save-checkpoint-frequency", type=int, default=20)
    ap.add_argument("--learning-rates", nargs="+", type=float, default=DEFAULT_LRS)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    distributed.enable(overwrite=True)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    timm_model = timm.create_model(args.model, pretrained=True)
    model = DinoV2IntermediateAdapter(timm_model).cuda().eval()
    n_params = sum(p.numel() for p in timm_model.parameters())
    logger.info(f"Loaded timm model '{args.model}'  params={n_params/1e6:.1f}M (frozen backbone)")

    run_eval_linear(
        model=model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        batch_size=args.batch_size,
        epochs=args.epochs,
        epoch_length=args.epoch_length,
        num_workers=args.num_workers,
        save_checkpoint_frequency=args.save_checkpoint_frequency,
        eval_period_iterations=args.eval_period_iterations,
        learning_rates=args.learning_rates,
        autocast_dtype=torch.float,
        resume=not args.no_resume,
        val_metric_type=MetricType.MEAN_ACCURACY,
    )


if __name__ == "__main__":
    main()
