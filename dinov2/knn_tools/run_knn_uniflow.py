#!/usr/bin/env python
"""k-NN eval for the UniFlow unified pixel-flow tokenizer, on the same ImageNet-1k
layout / protocol as the DINOv2 runs (so numbers are directly comparable).

UniFlow tokenizes an image with a pretrained ViT encoder and a channel projection
(`chal_proj`) down to `latent_ch=64` continuous tokens -- these latent tokens are the
tokenizer's actual output (what a downstream MLLM consumes for understanding). For a
per-image k-NN feature we run the encoder and mean-pool a token representation, then
hand the [B, D] vector to DINOv2's `eval_knn_with_model` (it L2-normalizes and runs k-NN).

`--feature` selects which representation to probe:
  * latent_mean (default): mean of the 64-d latent tokens  -> the tokenizer output
  * patch_mean           : mean of the ViT patch tokens (1024-d)
  * cls                  : the ViT [CLS] token (1024-d)

Single A100-80G, conda env `dino`. Run under torchrun:

    REPO=/cache/ma-user/VTBenchLab/dinov2
    DATA=/cache/ma-user/VTBenchLab/data/imagenet1k
    cd $REPO
    PYTHONPATH=. torchrun --nproc_per_node=1 knn_tools/run_knn_uniflow.py \
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

# TokBench image_scripts holds uniflow_rec.py (shared loader) + UniFlow/ (modeling code).
IMAGE_SCRIPTS = "/cache/ma-user/VTBenchLab/TokBench/tokenzier_vae_scripts/image_scripts"
MODELZOO = "/cache/ma-user/VTBenchLab/TokBench/tokenizer_modelzoo"
DEFAULT_MODEL_DIR = os.path.join(MODELZOO, "uniflow")
DEFAULT_OUTPUT = "/cache/ma-user/VTBenchLab/outputs/rec_vae_knn/uniflow"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class UniFlowFeature(nn.Module):
    """Wraps UniFlow so forward(images:[B,3,448,448]) -> features:[B,D].

    Replays the encoder half of UniFlowVisionModel.forward and pools a single
    global vector. Left un-normalized; the k-NN harness L2-normalizes [B, D].
    """

    def __init__(self, model, feature="latent_mean"):
        super().__init__()
        self.model = model
        self.feature = feature

    def forward(self, images):
        m = self.model
        images = images.to(dtype=m.embeddings.patch_embedding.weight.dtype)
        hidden = m.embeddings(images)                                   # [B, N+1, C]
        enc = m.encoder(inputs_embeds=hidden, output_hidden_states=False)
        tokens = enc.last_hidden_state                                  # [B, N+1, C], 0 = CLS
        if self.feature == "cls":
            feat = tokens[:, 0, :]
        elif self.feature == "patch_mean":
            feat = tokens[:, 1:, :].mean(dim=1)
        elif self.feature == "latent_mean":
            feat = m.chal_proj(tokens[:, 1:, :]).mean(dim=1)            # the 64-d latent tokens
        else:
            raise ValueError(f"unknown feature mode: {self.feature}")
        return feat.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=None, help="default: rec_vae_knn/uniflow[_<feature>]")
    ap.add_argument("--train-dataset", dest="train_dataset_str", required=True)
    ap.add_argument("--val-dataset", dest="val_dataset_str", required=True)
    ap.add_argument("--uniflow-path", default=os.path.join(IMAGE_SCRIPTS, "UniFlow"),
                    help="UniFlow repo root containing the `uniflow/` package.")
    ap.add_argument("--config-path", default=DEFAULT_MODEL_DIR, help="dir with config.json")
    ap.add_argument("--ckpt-path", default=os.path.join(DEFAULT_MODEL_DIR, "model.safetensors"))
    ap.add_argument("--feature", default="latent_mean", choices=["latent_mean", "patch_mean", "cls"])
    ap.add_argument("--image-size", type=int, default=448, help="UniFlow native resolution.")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=5)
    ap.add_argument("--nb_knn", nargs="+", type=int, default=[10, 20, 100, 200])
    ap.add_argument("--temperature", type=float, default=0.07)
    args = ap.parse_args()

    output_dir = args.output_dir or (
        DEFAULT_OUTPUT if args.feature == "latent_mean" else f"{DEFAULT_OUTPUT}_{args.feature}"
    )

    distributed.enable(overwrite=True)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output=output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    # Reuse TokBench's loader (package wiring, safetensors key handling all live there).
    if IMAGE_SCRIPTS not in sys.path:
        sys.path.insert(0, IMAGE_SCRIPTS)
    from uniflow_rec import load_uniflow

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    uniflow, config = load_uniflow(
        SimpleNamespace(uniflow_path=args.uniflow_path, config_path=args.config_path, ckpt_path=args.ckpt_path),
        device="cuda", dtype=dtype,
    )
    uniflow.eval()
    n_params = sum(p.numel() for p in uniflow.parameters())
    logger.info(f"Loaded UniFlow  params={n_params / 1e6:.1f}M  feature={args.feature}  "
                f"ckpt={args.ckpt_path}")

    model = UniFlowFeature(uniflow, feature=args.feature).cuda().eval()

    # UniFlow's own preprocessing: resize to native res, center-crop, ImageNet-normalize.
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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
