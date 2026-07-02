#!/usr/bin/env python
"""k-NN eval for TokLIP (tokenized-CLIP) on the DINOv2 ImageNet-1k layout / protocol.

NOTE: TokBench's `toklip_*_rec.py` only loads a LlamaGen VQ for *pixel* reconstruction
and ignores the TokLIP_*.pt semantic weights. For k-NN we load the actual TokLIP
encoder (open_clip / SigLIP2-SO400M visual tower). `model.encode_image(pixels)` runs
the internal VQ tokenization + transformer and returns a global [B, 1152] feature, which
we hand to DINOv2's `eval_knn_with_model` (it L2-normalizes [B, D] and runs k-NN).

The toklip ViT internally builds a LlamaGen VQ and loads `vq_ds16_t2i.pt` from the
*relative* path ./tokenizer/pretrained_models/ (downloading from HF if missing). We
pre-symlink the modelzoo copy there so it never hits the network.

Single A100-80G, conda env `dino`. Run under torchrun (one variant per invocation):

    REPO=/cache/ma-user/VTBenchLab/dinov2
    DATA=/cache/ma-user/VTBenchLab/data/imagenet1k
    cd $REPO
    PYTHONPATH=. torchrun --nproc_per_node=1 knn_tools/run_knn_toklip.py --variant s \
        --train-dataset "ImageNet:split=TRAIN:root=$DATA:extra=$DATA/extra" \
        --val-dataset   "ImageNet:split=VAL:root=$DATA:extra=$DATA/extra"
"""
import argparse
import logging
import os
import sys

import torch
from torch import nn

import dinov2.distributed as distributed
from dinov2.eval.knn import eval_knn_with_model
from dinov2.eval.metrics import AccuracyAveraging
from dinov2.logging import setup_logging

MODELZOO = "/cache/ma-user/VTBenchLab/TokBench/tokenizer_modelzoo"
TOKLIP_SRC = "/cache/ma-user/VTBenchLab/TokBench/tokenzier_vae_scripts/image_scripts/TokLIP/src"
DEFAULT_VQ_CKPT = os.path.join(MODELZOO, "TokLIP", "vq_ds16_t2i.pt")
OUTPUT_ROOT = "/cache/ma-user/VTBenchLab/outputs/rec_vae_knn"

VARIANTS = {
    "s": {
        "cfg": "ViT-SO400M-16-SigLIP2-256-toklip",
        "image_size": 256,
        "ckpt": os.path.join(MODELZOO, "TokLIP", "TokLIP_S_256.pt"),
        "name": "toklip_s_256",
    },
    "l": {
        "cfg": "ViT-SO400M-16-SigLIP2-384-toklip",
        "image_size": 384,
        "ckpt": os.path.join(MODELZOO, "TokLIP", "TokLIP_L_384.pt"),
        "name": "toklip_l_384",
    },
}


class TokLIPFeature(nn.Module):
    """Wraps TokLIP so forward(images:[B,3,H,W]) -> features:[B,1152]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        # encode_image returns the un-normalized global image feature; the k-NN
        # harness L2-normalizes [B, D] itself.
        return self.model.encode_image(images).float()


def ensure_vq_ckpt(vq_ckpt):
    """The toklip ViT hard-codes ./tokenizer/pretrained_models/vq_ds16_t2i.pt
    (relative to CWD). Symlink the modelzoo copy there so build never downloads."""
    target = os.path.abspath(os.path.join("tokenizer", "pretrained_models", "vq_ds16_t2i.pt"))
    if os.path.exists(target):
        return
    if not os.path.exists(vq_ckpt):
        raise FileNotFoundError(f"Missing TokLIP VQ checkpoint: {vq_ckpt}")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    os.symlink(os.path.abspath(vq_ckpt), target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="s",
                    help="s = TokLIP-S/256, l = TokLIP-L/384")
    ap.add_argument("--output-dir", default=None, help="default: rec_vae_knn/<variant name>")
    ap.add_argument("--train-dataset", dest="train_dataset_str", required=True)
    ap.add_argument("--val-dataset", dest="val_dataset_str", required=True)
    ap.add_argument("--ckpt-path", default=None, help="override TokLIP_*.pt path")
    ap.add_argument("--vq-ckpt", default=DEFAULT_VQ_CKPT)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=5)
    ap.add_argument("--nb_knn", nargs="+", type=int, default=[10, 20, 100, 200])
    ap.add_argument("--temperature", type=float, default=0.07)
    args = ap.parse_args()

    spec = VARIANTS[args.variant]
    output_dir = args.output_dir or os.path.join(OUTPUT_ROOT, spec["name"])
    ckpt = args.ckpt_path or spec["ckpt"]

    distributed.enable(overwrite=True)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output=output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    if TOKLIP_SRC not in sys.path:
        sys.path.insert(0, TOKLIP_SRC)  # so `open_clip` / `timm_local` resolve to TokLIP's
    ensure_vq_ckpt(args.vq_ckpt)

    from create_toklip import create_toklip

    model, _, preprocess_val = create_toklip(
        model=spec["cfg"], image_size=spec["image_size"], model_path=ckpt, device="cuda",
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded TokLIP-{args.variant.upper()} ({spec['cfg']}, {spec['image_size']}px)  "
                f"params={n_params / 1e6:.1f}M  ckpt={ckpt}")

    feat_model = TokLIPFeature(model).cuda().eval()

    eval_knn_with_model(
        model=feat_model,
        output_dir=output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        nb_knn=tuple(args.nb_knn),
        temperature=args.temperature,
        autocast_dtype=torch.float,
        accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
        transform=preprocess_val,  # TokLIP's own SigLIP2 resize + normalization
        gather_on_cpu=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_per_class_list=[-1],
        n_tries=1,
    )


if __name__ == "__main__":
    main()
