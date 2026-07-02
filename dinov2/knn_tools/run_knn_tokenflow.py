#!/usr/bin/env python
"""k-NN eval for TokenFlow (dual-codebook semantic+pixel tokenizer) on the DINOv2
ImageNet-1k layout / protocol.

TokenFlow's `encode(x)` returns the quantized latent `quant:[B, code_dim, H, W]`,
whose first `semantic_code_dim` (=32) channels are the CLIP-aligned *semantic* code
(the rest is the VQGAN/pixel code -- see `decode`'s torch.split). For k-NN we tap the
tokenizer's own semantic code and global-average-pool it -> [B, 32]. Pass
`--feature teacher` to instead pool the raw CLIP ViT-B/16 teacher patch tokens
([B, 768]) as a CLIP upper-bound reference.

Either way the [B, D] vector goes to DINOv2's `eval_knn_with_model`, which
L2-normalizes it and runs k-NN.

The model loads CLIP ViT-B/16 from the *relative* path ./clip_model/ViT-B-16.pt
(SHA-checked; downloads from OpenAI if missing). We pre-symlink the in-repo copy there.

Single A100-80G, conda env `dino` (needs `einops`). Run under torchrun:

    REPO=/cache/ma-user/VTBenchLab/dinov2
    DATA=/cache/ma-user/VTBenchLab/data/imagenet1k
    cd $REPO
    PYTHONPATH=. torchrun --nproc_per_node=1 knn_tools/run_knn_tokenflow.py \
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
DEFAULT_CKPT = os.path.join(MODELZOO, "TokenFlow", "tokenflow_clipb_32k_enhanced.pt")
DEFAULT_CLIP = os.path.join(IMAGE_SCRIPTS, "clip_model", "ViT-B-16.pt")
DEFAULT_OUTPUT = "/cache/ma-user/VTBenchLab/outputs/rec_vae_knn/tokenflow"


class TokenFlowFeature(nn.Module):
    """Wraps TokenFlow so forward(images:[B,3,H,W]) -> features:[B,D]."""

    def __init__(self, vq_model, feature):
        super().__init__()
        self.vq_model = vq_model
        self.feature = feature
        self.semantic_code_dim = vq_model.semantic_code_dim

    def forward(self, images):
        if self.feature == "teacher":
            # Raw CLIP ViT-B/16 teacher patch tokens -> GAP -> [B, 768] (upper bound).
            tokens = self.vq_model.encoder_vqkd(images, return_patch_tokens=True)
            return tokens.float().mean(dim=1)
        # TokenFlow's own quantized semantic code: first semantic_code_dim channels.
        quant, _, _ = self.vq_model.encode(images)
        sem = quant[:, : self.semantic_code_dim]          # [B, 32, H, W]
        return sem.float().mean(dim=(2, 3))               # GAP -> [B, 32]


def ensure_clip_model(clip_ckpt):
    """TokenFlow hard-codes download_root='./clip_model' (relative to CWD) and looks
    for ViT-B-16.pt there. Symlink the in-repo copy so build never hits the network."""
    target = os.path.abspath(os.path.join("clip_model", "ViT-B-16.pt"))
    if os.path.exists(target):
        return
    if not os.path.exists(clip_ckpt):
        raise FileNotFoundError(f"Missing CLIP ViT-B/16 checkpoint: {clip_ckpt}")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    os.symlink(os.path.abspath(clip_ckpt), target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--train-dataset", dest="train_dataset_str", required=True)
    ap.add_argument("--val-dataset", dest="val_dataset_str", required=True)
    ap.add_argument("--ckpt-path", default=DEFAULT_CKPT)
    ap.add_argument("--tokenflow-path", default=os.path.join(IMAGE_SCRIPTS, "TokenFlow"))
    ap.add_argument("--clip-ckpt", default=DEFAULT_CLIP)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--feature", choices=["sem_quant", "teacher"], default="sem_quant",
                    help="sem_quant = TokenFlow's quantized semantic code [B,32]; "
                         "teacher = raw CLIP ViT-B/16 patch tokens [B,768].")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=5)
    ap.add_argument("--nb_knn", nargs="+", type=int, default=[10, 20, 100, 200])
    ap.add_argument("--temperature", type=float, default=0.07)
    args = ap.parse_args()

    distributed.enable(overwrite=True)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    if IMAGE_SCRIPTS not in sys.path:
        sys.path.insert(0, IMAGE_SCRIPTS)  # so `import TokenFlow` + resize_rec resolve
    ensure_clip_model(args.clip_ckpt)

    from tokenflow_rec import load_tokenflow

    vq_model = load_tokenflow(
        SimpleNamespace(
            tokenflow_path=args.tokenflow_path,
            ckpt_path=args.ckpt_path,
            padding_size=args.resolution,
        ),
        device="cuda",
    )
    vq_model.eval()
    n_params = sum(p.numel() for p in vq_model.parameters())
    logger.info(f"Loaded TokenFlow  res={args.resolution}  feature={args.feature}  "
                f"params={n_params / 1e6:.1f}M  ckpt={args.ckpt_path}")

    model = TokenFlowFeature(vq_model, args.feature).cuda().eval()

    # Match the rec preprocessing: resize/center-crop to res + [-1,1] normalization.
    transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
