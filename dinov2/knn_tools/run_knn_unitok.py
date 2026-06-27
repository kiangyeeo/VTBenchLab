#!/usr/bin/env python
"""k-NN eval for the UniTok unified tokenizer, on the same ImageNet-1k layout /
protocol as the DINOv2 runs (so numbers are directly comparable).

UniTok carries a CLIP-aligned image head: `unitok.encode_image(img)` returns a
global, semantic [B, embed_dim=768] vector built from the *quantized* tokens
(encoder -> quant_proj -> f_to_idx -> idx_to_f -> post_quant_proj -> mean ->
projection). That is exactly the per-image feature k-NN needs; we just feed it to
DINOv2's `eval_knn_with_model`, which L2-normalizes [B, D] and runs k-NN.

Single A100-80G, conda env `dino`. Run under torchrun:

    REPO=/cache/ma-user/VTBenchLab/dinov2
    DATA=/cache/ma-user/VTBenchLab/data/imagenet1k
    cd $REPO
    PYTHONPATH=. torchrun --nproc_per_node=1 knn_tools/run_knn_unitok.py \
        --train-dataset "ImageNet:split=TRAIN:root=$DATA:extra=$DATA/extra" \
        --val-dataset   "ImageNet:split=VAL:root=$DATA:extra=$DATA/extra"
"""
import argparse
import logging
import os
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torchvision import transforms

import dinov2.distributed as distributed
from dinov2.eval.knn import eval_knn_with_model
from dinov2.eval.metrics import AccuracyAveraging
from dinov2.logging import setup_logging

# TokBench image_scripts (holds UniTok/, resize_rec.py, unitok_vae_rec.py).
IMAGE_SCRIPTS = "/cache/ma-user/VTBenchLab/TokBench/tokenzier_vae_scripts/image_scripts"
MODELZOO = "/cache/ma-user/VTBenchLab/TokBench/tokenizer_modelzoo"
DEFAULT_CKPT = os.path.join(MODELZOO, "unitok_20250227", "unitok_tokenizer.pth")
DEFAULT_OUTPUT = "/cache/ma-user/VTBenchLab/outputs/rec_vae_knn/unitok"


class UniTokFeature(nn.Module):
    """Wraps UniTok so forward(images:[B,3,H,W]) -> features:[B,768]."""

    def __init__(self, unitok):
        super().__init__()
        self.unitok = unitok

    def forward(self, images):
        # encode_image returns the CLIP-aligned, quantized-token global feature.
        # Leave it un-normalized; eval_knn_with_model L2-normalizes [B, D] itself.
        return self.unitok.encode_image(images, normalize=False).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--train-dataset", dest="train_dataset_str", required=True)
    ap.add_argument("--val-dataset", dest="val_dataset_str", required=True)
    ap.add_argument("--ckpt-path", default=DEFAULT_CKPT)
    ap.add_argument("--unitok-path", default=os.path.join(IMAGE_SCRIPTS, "UniTok"))
    ap.add_argument("--image-size", type=int, default=256,
                    help="UniTok native resolution (the released tokenizer is 256).")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=5)
    ap.add_argument("--nb_knn", nargs="+", type=int, default=[10, 20, 100, 200])
    ap.add_argument("--temperature", type=float, default=0.07)
    args = ap.parse_args()

    distributed.enable(overwrite=True)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    # Reuse TokBench's debugged loader (timm-compat + causal-projection patches,
    # sys.path wiring, checkpoint key handling all live there).
    if IMAGE_SCRIPTS not in sys.path:
        sys.path.insert(0, IMAGE_SCRIPTS)
    from unitok_vae_rec import load_unitok

    unitok, base_preprocess = load_unitok(
        SimpleNamespace(unitok_path=args.unitok_path, ckpt_path=args.ckpt_path)
    )
    unitok.eval()
    n_params = sum(p.numel() for p in unitok.parameters())
    logger.info(f"Loaded UniTok  params={n_params / 1e6:.1f}M  ckpt={args.ckpt_path}")

    model = UniTokFeature(unitok).cuda().eval()

    # Match the model's native res + its own [-1,1] normalization (base_preprocess =
    # ToTensor + normalize_01_into_pm1); prepend the standard resize/center-crop.
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        *base_preprocess.transforms,
    ])

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
