#!/usr/bin/env python
"""k-NN eval for LlamaGen's VQ tokenizers (VQ-16 / VQ-8) on the DINOv2 ImageNet-1k
layout / protocol.

LlamaGen is a pure *reconstruction* VQ tokenizer (no semantic/CLIP alignment), so
there is no global feature head -- we global-average-pool a spatial latent.

  encode path:  encoder(x) -> [B, 256, h, w]  --quant_conv-->  [B, 8, h, w]  --quantize
                \__ continuous, z_channels=256        \__ quantized code embed (8-d)

  --feature enc   (default): GAP the *pre-quant* continuous feature -> [B, 256]
  --feature quant          : GAP the quantized code embedding       -> [B, 8]

The 8-d quantized code is nearly useless for k-NN; the 256-d pre-quant feature is the
standard "probe the tokenizer encoder" choice. Either way the [B, D] vector goes to
DINOv2's `eval_knn_with_model` (L2-normalize + k-NN). Expect *low* accuracy -- that is
the point for a reconstruction tokenizer.

Single A100-80G, conda env `dino` (no extra deps needed -- pure torch). Run under torchrun:

    REPO=/cache/ma-user/VTBenchLab/dinov2
    DATA=/cache/ma-user/VTBenchLab/data/imagenet1k
    cd $REPO
    PYTHONPATH=. torchrun --nproc_per_node=1 knn_tools/run_knn_llamagen.py --variant vq16 \
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

IMAGE_SCRIPTS = "/cache/ma-user/VTBenchLab/TokBench/tokenzier_vae_scripts/image_scripts"
MODELZOO = "/cache/ma-user/VTBenchLab/TokBench/tokenizer_modelzoo"
OUTPUT_ROOT = "/cache/ma-user/VTBenchLab/outputs/rec_vae_knn"

VARIANTS = {
    "vq16": {"model_name": "llamagen_vq16", "ckpt": os.path.join(MODELZOO, "LlamaGen", "vq_ds16_c2i.pt"),
             "name": "llamagen_vq16"},
    "vq8":  {"model_name": "llamagen_vq8",  "ckpt": os.path.join(MODELZOO, "LlamaGen", "vq_ds8_c2i.pt"),
             "name": "llamagen_vq8"},
}


class LlamaGenFeature(nn.Module):
    """Wraps a LlamaGen VQModel so forward(images:[B,3,H,W]) -> features:[B,D]."""

    def __init__(self, vq_model, feature):
        super().__init__()
        self.vq_model = vq_model
        self.feature = feature

    def forward(self, images):
        if self.feature == "quant":
            quant, _, _ = self.vq_model.encode(images)   # [B, 8, h, w]
            return quant.float().mean(dim=(2, 3))         # GAP -> [B, 8]
        feat = self.vq_model.encoder(images)              # [B, 256, h, w] (pre-quant)
        return feat.float().mean(dim=(2, 3))              # GAP -> [B, 256]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="vq16",
                    help="vq16 = VQ-16 (f16), vq8 = VQ-8 (f8)")
    ap.add_argument("--output-dir", default=None, help="default: rec_vae_knn/<variant name>")
    ap.add_argument("--train-dataset", dest="train_dataset_str", required=True)
    ap.add_argument("--val-dataset", dest="val_dataset_str", required=True)
    ap.add_argument("--ckpt-path", default=None, help="override vq_ds*_c2i.pt path")
    ap.add_argument("--llamagen-path", default=os.path.join(IMAGE_SCRIPTS, "LlamaGen"))
    ap.add_argument("--feature", choices=["enc", "quant"], default="enc",
                    help="enc = pre-quant continuous feature [B,256]; quant = code embed [B,8].")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=5)
    ap.add_argument("--nb_knn", nargs="+", type=int, default=[10, 20, 100, 200])
    ap.add_argument("--temperature", type=float, default=0.07)
    args = ap.parse_args()

    spec = VARIANTS[args.variant]
    output_dir = args.output_dir or os.path.join(OUTPUT_ROOT, spec["name"])

    distributed.enable(overwrite=True)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output=output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    if IMAGE_SCRIPTS not in sys.path:
        sys.path.insert(0, IMAGE_SCRIPTS)
    from llamagen_rec import load_llamagen_vq

    vq_model = load_llamagen_vq(
        SimpleNamespace(
            llamagen_path=args.llamagen_path,
            ckpt_path=args.ckpt_path or spec["ckpt"],
            model_name=spec["model_name"],
        ),
        device="cuda",
    )
    vq_model.eval()
    n_params = sum(p.numel() for p in vq_model.parameters())
    logger.info(f"Loaded LlamaGen {args.variant}  feature={args.feature}  "
                f"params={n_params / 1e6:.1f}M  ckpt={args.ckpt_path or spec['ckpt']}")

    model = LlamaGenFeature(vq_model, args.feature).cuda().eval()

    # Same [-1,1] normalization as the rec script; resize/center-crop to image_size.
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    eval_knn_with_model(
        model=model,
        output_dir=output_dir,
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
