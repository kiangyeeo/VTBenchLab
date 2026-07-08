#!/usr/bin/env python
"""Linear-probing eval for MetaCLIP timm backbones.

This keeps MetaCLIP separate from TokLIP: MetaCLIP is a continuous CLIP visual
encoder, while TokLIP is a tokenized-CLIP model with its own tokenizer path.
"""
import argparse
import logging
import os

import timm
import torch
import torch.nn as nn
from timm.data import create_transform, resolve_model_data_config
from timm.models import load_checkpoint

import dinov2.distributed as distributed
from dinov2.eval.linear import run_eval_linear
from dinov2.eval.metrics import MetricType
from dinov2.logging import setup_logging


DEFAULT_MODEL = "vit_base_patch16_clip_224.metaclip_2pt5b"
DEFAULT_OUTPUT = "/cache/ma-user/VTBenchLab/outputs/vae_linear_probing/metaclip_b16_2pt5b"
DEFAULT_LRS = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]


class TimmIntermediateAdapter(nn.Module):
    """Expose DINOv2-compatible get_intermediate_layers() on a timm ViT."""

    def __init__(self, timm_model):
        super().__init__()
        self.model = timm_model

    def get_intermediate_layers(self, x, n, return_class_token=False):
        outs = self.model.get_intermediate_layers(x, n, return_prefix_tokens=True, norm=True)
        if return_class_token:
            return tuple((patch, prefix[:, 0]) for patch, prefix in outs)
        return tuple(patch for patch, _ in outs)


def _find_checkpoint(path):
    if not path:
        return None
    if os.path.isfile(path):
        return path
    candidates = [
        "model.safetensors",
        "pytorch_model.bin",
        "open_clip_pytorch_model.bin",
        "checkpoint.pth",
    ]
    for name in candidates:
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"No supported checkpoint file found under {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--train-dataset", dest="train_dataset_str", required=True)
    ap.add_argument("--val-dataset", dest="val_dataset_str", required=True)
    ap.add_argument("--test-datasets", dest="test_dataset_strs", nargs="+", default=None)
    ap.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional local checkpoint file or HF local-dir. If omitted, timm downloads via HF cache.",
    )
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

    checkpoint_path = _find_checkpoint(args.checkpoint_path)
    timm_model = timm.create_model(args.model, pretrained=checkpoint_path is None)
    if checkpoint_path is not None:
        load_checkpoint(timm_model, checkpoint_path, strict=True)

    data_config = resolve_model_data_config(timm_model)
    train_transform = create_transform(**data_config, is_training=True)
    eval_transform = create_transform(**data_config, is_training=False)

    model = TimmIntermediateAdapter(timm_model).cuda().eval()
    n_params = sum(p.numel() for p in timm_model.parameters())
    logger.info(
        f"Loaded MetaCLIP/timm model '{args.model}' params={n_params / 1e6:.1f}M "
        f"checkpoint={checkpoint_path or 'hf'} data_config={data_config}"
    )

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
