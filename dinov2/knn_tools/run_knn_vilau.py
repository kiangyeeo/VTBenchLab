#!/usr/bin/env python
"""k-NN eval for VILA-U's vision tokenizer (RQ-VAE + SigLIP) on the DINOv2
ImageNet-1k layout / protocol.

VILA-U tokenizes with a SigLIP vision tower whose penultimate layer is residual-
vector-quantized. `rqvaesiglip.encode_image(pixels)` returns (code, z_q) where
z_q:[B,H,W,C] is the quantized SigLIP feature map (C=1024 for the 256px / 1152 for
the 384px model). We global-average-pool z_q over H,W -> [B,C] and hand that to
DINOv2's `eval_knn_with_model` (it L2-normalizes [B, D] and runs k-NN).

We reuse TokBench's `load_vilau_tokenizer`, including its CLIPImageProcessor, as the
eval transform so preprocessing matches exactly how the tokenizer was benchmarked.

Single A100-80G, conda env `dino` (needs `transformers`). Run under torchrun:

    REPO=/cache/ma-user/VTBenchLab/dinov2
    DATA=/cache/ma-user/VTBenchLab/data/imagenet1k
    cd $REPO
    PYTHONPATH=. torchrun --nproc_per_node=1 knn_tools/run_knn_vilau.py \
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

import dinov2.distributed as distributed
from dinov2.eval.knn import eval_knn_with_model
from dinov2.eval.metrics import AccuracyAveraging
from dinov2.logging import setup_logging

IMAGE_SCRIPTS = "/cache/ma-user/VTBenchLab/TokBench/tokenzier_vae_scripts/image_scripts"
MODELZOO = "/cache/ma-user/VTBenchLab/TokBench/tokenizer_modelzoo"
DEFAULT_MODEL_DIR = os.path.join(MODELZOO, "VILA-U", "vila-u-7b-256")
DEFAULT_SIGLIP_CONFIG = os.path.join(MODELZOO, "VILA-U", "siglip-large-patch16-256")
DEFAULT_OUTPUT = "/cache/ma-user/VTBenchLab/outputs/rec_vae_knn/vilau_7b_256"


class VilaUFeature(nn.Module):
    """Wraps VILA-U's rqvaesiglip so forward(images:[B,3,H,W]) -> features:[B,C]."""

    def __init__(self, model, dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, images):
        # encode_image -> (code, z_q); z_q:[B,H,W,C] is the quantized SigLIP map.
        _code, z_q = self.model.encode_image(images.to(self.dtype))
        return z_q.float().mean(dim=(1, 2))  # GAP over H,W -> [B, C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--train-dataset", dest="train_dataset_str", required=True)
    ap.add_argument("--val-dataset", dest="val_dataset_str", required=True)
    ap.add_argument("--model-path", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--siglip-config-path", default=DEFAULT_SIGLIP_CONFIG)
    ap.add_argument("--vilau-path", default=os.path.join(IMAGE_SCRIPTS, "vila-u"))
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=5)
    ap.add_argument("--nb_knn", nargs="+", type=int, default=[10, 20, 100, 200])
    ap.add_argument("--temperature", type=float, default=0.07)
    args = ap.parse_args()

    distributed.enable(overwrite=True)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    if IMAGE_SCRIPTS not in sys.path:
        sys.path.insert(0, IMAGE_SCRIPTS)
    from vilau_rec import load_vilau_tokenizer

    tok = load_vilau_tokenizer(
        SimpleNamespace(
            vilau_path=args.vilau_path,
            model_path=args.model_path,
            siglip_config_path=args.siglip_config_path,
            dtype=args.dtype,
        ),
        device="cuda",
    )
    n_params = sum(p.numel() for p in tok.model.parameters())
    logger.info(f"Loaded VILA-U rqvaesiglip  image_size={tok.image_size}  "
                f"params={n_params / 1e6:.1f}M  dtype={args.dtype}")

    model = VilaUFeature(tok.model, tok.dtype).cuda().eval()

    # Reuse VILA-U's own CLIPImageProcessor (resize+normalize to image_size, mean/std 0.5).
    image_processor = tok.image_processor

    def transform(pil_img):
        return image_processor.preprocess(pil_img, return_tensors="pt")["pixel_values"][0]

    autocast_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                      "float32": torch.float}[args.dtype]

    eval_knn_with_model(
        model=model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        nb_knn=tuple(args.nb_knn),
        temperature=args.temperature,
        autocast_dtype=autocast_dtype,
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
