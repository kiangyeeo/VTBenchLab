#!/usr/bin/env python
"""Linear-probing eval for the UniTok image tokenizer.

This reuses DINOv2's linear protocol while feeding UniTok's frozen image-token
features through a small adapter:

  - encoder / quant_proj / codebook lookup / post_quant_proj are frozen
  - the final quantized image tokens become patch tokens
  - the mean-pooled token feature becomes the class token
  - DINOv2 trains its usual grid of linear heads on top

Single A100-80G example:

    REPO=/cache/ma-user/VTBenchLab/dinov2
    DATA=/cache/ma-user/VTBenchLab/data/imagenet1k
    cd $REPO
    PYTHONPATH=. torchrun --standalone --nproc_per_node=1 knn_tools/run_linear_unitok.py \
        --output-dir /cache/ma-user/VTBenchLab/outputs/vae_linear_probing/unitok \
        --train-dataset "ImageNet:split=TRAIN:root=$DATA:extra=$DATA/extra" \
        --val-dataset   "ImageNet:split=VAL:root=$DATA:extra=$DATA/extra"
"""
import argparse
import logging
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

import dinov2.distributed as distributed
from dinov2.data.transforms import make_classification_eval_transform, make_classification_train_transform
from dinov2.eval.linear import run_eval_linear
from dinov2.eval.metrics import MetricType
from dinov2.logging import setup_logging


IMAGE_SCRIPTS = "/cache/ma-user/VTBenchLab/TokBench/tokenzier_vae_scripts/image_scripts"
MODELZOO = "/cache/ma-user/VTBenchLab/TokBench/tokenizer_modelzoo"
DEFAULT_CKPT = os.path.join(MODELZOO, "unitok_20250227", "unitok_tokenizer.pth")
DEFAULT_OUTPUT = "/cache/ma-user/VTBenchLab/outputs/vae_linear_probing/unitok"
DEFAULT_LRS = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]
PM1_MEAN = (0.5, 0.5, 0.5)
PM1_STD = (0.5, 0.5, 0.5)


class UniTokLinearAdapter(nn.Module):
    """Expose DINOv2-compatible intermediate layers for UniTok tokens."""

    def __init__(self, unitok, feature="quantized"):
        super().__init__()
        self.unitok = unitok
        self.feature = feature

    def _image_tokens(self, images):
        unitok = self.unitok
        tokens = unitok.encoder(images).float()
        if self.feature != "encoder":
            tokens = unitok.quant_proj(tokens)
            indices = unitok.quantizer.f_to_idx(tokens)
            tokens = unitok.quantizer.idx_to_f(indices)
            tokens = unitok.post_quant_proj(tokens)
        return tokens

    def get_intermediate_layers(self, images, n, return_class_token=False):
        tokens = self._image_tokens(images)

        if self.feature == "clip":
            patch_tokens = self.unitok.projection(self.unitok.fc_norm(tokens))
            class_token = self.unitok.projection(self.unitok.fc_norm(tokens.mean(dim=1)))
        else:
            patch_tokens = self.unitok.fc_norm(tokens)
            class_token = self.unitok.fc_norm(tokens.mean(dim=1))

        n = n if isinstance(n, int) else max(n)
        if return_class_token:
            return tuple((patch_tokens, class_token) for _ in range(n))
        return tuple(patch_tokens for _ in range(n))


def _build_transforms(image_size, eval_resize_size, hflip_prob):
    train_transform = make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=hflip_prob,
        mean=PM1_MEAN,
        std=PM1_STD,
    )
    eval_transform = make_classification_eval_transform(
        resize_size=eval_resize_size,
        crop_size=image_size,
        mean=PM1_MEAN,
        std=PM1_STD,
    )
    return train_transform, eval_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--train-dataset", dest="train_dataset_str", required=True)
    ap.add_argument("--val-dataset", dest="val_dataset_str", required=True)
    ap.add_argument("--test-datasets", dest="test_dataset_strs", nargs="+", default=None)
    ap.add_argument("--ckpt-path", default=DEFAULT_CKPT)
    ap.add_argument("--unitok-path", default=os.path.join(IMAGE_SCRIPTS, "UniTok"))
    ap.add_argument(
        "--feature",
        default="quantized",
        choices=["quantized", "encoder", "clip"],
        help="UniTok representation to probe. Default probes the quantized tokenizer tokens.",
    )
    ap.add_argument("--image-size", type=int, default=256, help="UniTok native crop size.")
    ap.add_argument("--eval-resize-size", type=int, default=256)
    ap.add_argument("--hflip-prob", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=10)
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

    if IMAGE_SCRIPTS not in sys.path:
        sys.path.insert(0, IMAGE_SCRIPTS)
    from unitok_vae_rec import load_unitok

    unitok, _ = load_unitok(SimpleNamespace(unitok_path=args.unitok_path, ckpt_path=args.ckpt_path))
    unitok.eval().requires_grad_(False)
    n_params = sum(p.numel() for p in unitok.parameters())
    logger.info(
        f"Loaded UniTok params={n_params / 1e6:.1f}M feature={args.feature} ckpt={args.ckpt_path}"
    )

    model = UniTokLinearAdapter(unitok, feature=args.feature).cuda().eval()
    train_transform, eval_transform = _build_transforms(args.image_size, args.eval_resize_size, args.hflip_prob)

    run_eval_linear(
        model=model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        test_dataset_strs=args.test_dataset_strs,
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
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


if __name__ == "__main__":
    main()
