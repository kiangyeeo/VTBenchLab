#!/usr/bin/env python
"""Linear-probing eval for LlamaGen VQ tokenizers (VQ-8 / VQ-16).

LlamaGen is a reconstruction tokenizer, so it has no semantic CLS or CLIP head.
This adapter exposes the spatial tokenizer features to DINOv2's linear protocol:

  - `enc`:   encoder feature map [B, 256, h, w] -> patch tokens [B, h*w, 256]
  - `quant`: quantized code embedding [B, 8, h, w] -> patch tokens [B, h*w, 8]

For both modes, the class token is the spatial mean of the patch tokens. DINOv2
then trains its usual grid of frozen-backbone linear heads.
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
OUTPUT_ROOT = "/cache/ma-user/VTBenchLab/outputs/vae_linear_probing"
DEFAULT_LRS = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]
PM1_MEAN = (0.5, 0.5, 0.5)
PM1_STD = (0.5, 0.5, 0.5)

VARIANTS = {
    "vq16": {
        "model_name": "llamagen_vq16",
        "ckpt": os.path.join(MODELZOO, "LlamaGen", "vq_ds16_c2i.pt"),
    },
    "vq8": {
        "model_name": "llamagen_vq8",
        "ckpt": os.path.join(MODELZOO, "LlamaGen", "vq_ds8_c2i.pt"),
    },
}


class LlamaGenLinearAdapter(nn.Module):
    """Expose DINOv2-compatible intermediate layers for LlamaGen spatial tokens."""

    def __init__(self, vq_model, feature="enc"):
        super().__init__()
        self.vq_model = vq_model
        self.feature = feature

    def _feature_map(self, images):
        if self.feature == "quant":
            quant, _, _ = self.vq_model.encode(images)
            return quant.float()
        return self.vq_model.encoder(images).float()

    def get_intermediate_layers(self, images, n, return_class_token=False):
        fmap = self._feature_map(images)
        patch_tokens = fmap.flatten(2).transpose(1, 2).contiguous()
        class_token = patch_tokens.mean(dim=1)

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
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="vq8")
    ap.add_argument("--output-dir", default=None, help="default: outputs/vae_linear_probing/llamagen_<variant>")
    ap.add_argument("--train-dataset", dest="train_dataset_str", required=True)
    ap.add_argument("--val-dataset", dest="val_dataset_str", required=True)
    ap.add_argument("--test-datasets", dest="test_dataset_strs", nargs="+", default=None)
    ap.add_argument("--ckpt-path", default=None)
    ap.add_argument("--llamagen-path", default=os.path.join(IMAGE_SCRIPTS, "LlamaGen"))
    ap.add_argument(
        "--feature",
        choices=["enc", "quant"],
        default="enc",
        help="enc probes the 256-d pre-quant encoder map; quant probes the 8-d code embedding.",
    )
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--eval-resize-size", type=int, default=256)
    ap.add_argument("--hflip-prob", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--epoch-length", type=int, default=1250)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--eval-period-iterations", type=int, default=1250)
    ap.add_argument("--save-checkpoint-frequency", type=int, default=20)
    ap.add_argument("--learning-rates", nargs="+", type=float, default=DEFAULT_LRS)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    spec = VARIANTS[args.variant]
    output_dir = args.output_dir or os.path.join(OUTPUT_ROOT, spec["model_name"])

    distributed.enable(overwrite=True)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output=output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    if IMAGE_SCRIPTS not in sys.path:
        sys.path.insert(0, IMAGE_SCRIPTS)
    from llamagen_rec import load_llamagen_vq

    ckpt_path = args.ckpt_path or spec["ckpt"]
    vq_model = load_llamagen_vq(
        SimpleNamespace(
            llamagen_path=args.llamagen_path,
            ckpt_path=ckpt_path,
            model_name=spec["model_name"],
        ),
        device="cuda",
    )
    vq_model.eval().requires_grad_(False)
    n_params = sum(p.numel() for p in vq_model.parameters())
    logger.info(
        f"Loaded LlamaGen {args.variant} params={n_params / 1e6:.1f}M feature={args.feature} ckpt={ckpt_path}"
    )

    model = LlamaGenLinearAdapter(vq_model, feature=args.feature).cuda().eval()
    train_transform, eval_transform = _build_transforms(args.image_size, args.eval_resize_size, args.hflip_prob)

    run_eval_linear(
        model=model,
        output_dir=output_dir,
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
