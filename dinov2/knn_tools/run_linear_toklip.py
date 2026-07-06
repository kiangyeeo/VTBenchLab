#!/usr/bin/env python
"""Linear-probing eval for TokLIP visual features.

By default this probes TokLIP's semantic ViT tokens, which is the right surface
for measuring classification-oriented semantic ability. A `zq` mode is kept for
the older reconstruction/generation-aligned VQ latent probe.
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
DEFAULT_VQ_CKPT = os.path.join(MODELZOO, "TokLIP", "vq_ds16_t2i.pt")
DEFAULT_LRS = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]
PM1_MEAN = (0.5, 0.5, 0.5)
PM1_STD = (0.5, 0.5, 0.5)

VARIANTS = {
    "s": {
        "model_name": "toklip_s",
        "output_name": "toklip_s_semantic_256",
        "image_size": 256,
        "ckpt": os.path.join(MODELZOO, "TokLIP", "TokLIP_S_256.pt"),
    },
    "l": {
        "model_name": "toklip_l",
        "output_name": "toklip_l_semantic_384",
        "image_size": 384,
        "ckpt": os.path.join(MODELZOO, "TokLIP", "TokLIP_L_384.pt"),
    },
}


class TokLIPLinearAdapter(nn.Module):
    """Expose DINOv2-compatible intermediate layers for TokLIP features."""

    def __init__(self, model, feature, encode_semantic_tokens=None):
        super().__init__()
        self.model = model
        self.feature = feature
        self.encode_semantic_tokens = encode_semantic_tokens

    def _tokens(self, images):
        if self.feature == "zq":
            quant, _, _ = self.model.encode(images)
            return quant.float().flatten(2).transpose(1, 2).contiguous()

        tokens = self.encode_semantic_tokens(self.model, images)
        return tokens.float().contiguous()

    def get_intermediate_layers(self, images, n, return_class_token=False):
        patch_tokens = self._tokens(images)
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
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="s", help="s = TokLIP-S/256, l = TokLIP-L/384")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--train-dataset", dest="train_dataset_str", required=True)
    ap.add_argument("--val-dataset", dest="val_dataset_str", required=True)
    ap.add_argument("--test-datasets", dest="test_dataset_strs", nargs="+", default=None)
    ap.add_argument("--toklip-path", default=os.path.join(IMAGE_SCRIPTS, "TokLIP"))
    ap.add_argument("--ckpt-path", default=None, help="TokLIP_*.pt path; checked for variant bookkeeping.")
    ap.add_argument("--vq-ckpt-path", default=DEFAULT_VQ_CKPT)
    ap.add_argument(
        "--feature",
        default="semantic",
        choices=["semantic", "zq"],
        help="semantic probes TokLIP ViT semantic tokens; zq probes the low-level VQ latent.",
    )
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--eval-resize-size", type=int, default=None)
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

    spec = VARIANTS[args.variant]
    if args.output_dir is None:
        output_name = spec["output_name"] if args.feature == "semantic" else spec["output_name"].replace("semantic", "zq")
        output_dir = os.path.join(OUTPUT_ROOT, output_name)
    else:
        output_dir = args.output_dir
    ckpt_path = args.ckpt_path or spec["ckpt"]
    image_size = args.image_size or spec["image_size"]
    eval_resize_size = args.eval_resize_size or image_size

    distributed.enable(overwrite=True)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output=output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    if IMAGE_SCRIPTS not in sys.path:
        sys.path.insert(0, IMAGE_SCRIPTS)
    from toklip_rec_common import encode_toklip_semantic_tokens, load_toklip_semantic_model, load_toklip_vq

    common_args = SimpleNamespace(
        toklip_path=args.toklip_path,
        toklip_ckpt_path=ckpt_path,
        vq_ckpt_path=args.vq_ckpt_path,
        model_name=spec["model_name"],
        toklip_model_config=None,
    )
    if args.feature == "semantic":
        toklip_model = load_toklip_semantic_model(common_args, device="cuda")
        encode_fn = encode_toklip_semantic_tokens
    else:
        toklip_model = load_toklip_vq(common_args, device="cuda")
        encode_fn = None

    toklip_model.eval().requires_grad_(False)
    n_params = sum(p.numel() for p in toklip_model.parameters())
    logger.info(
        f"Loaded TokLIP-{args.variant.upper()} feature={args.feature} params={n_params / 1e6:.1f}M "
        f"image_size={image_size} ckpt={ckpt_path} vq_ckpt={args.vq_ckpt_path}"
    )

    model = TokLIPLinearAdapter(toklip_model, feature=args.feature, encode_semantic_tokens=encode_fn).cuda().eval()
    train_transform, eval_transform = _build_transforms(image_size, eval_resize_size, args.hflip_prob)

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
