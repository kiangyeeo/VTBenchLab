#!/usr/bin/env python
"""Linear-probing eval for VILA-U visual features.

Default feature is the VILA-U SigLIP semantic latent at the penultimate layer,
matching the "semantic" TokBench variant discussed for MLLM-aligned probing.
The adapter exposes the spatial latent map as DINOv2-style patch tokens and
uses their mean as the class token.
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
DEFAULT_MODEL_DIR = os.path.join(MODELZOO, "VILA-U", "vila-u-7b-256")
DEFAULT_SIGLIP_CONFIG = os.path.join(MODELZOO, "VILA-U", "siglip-large-patch16-256")
DEFAULT_LRS = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]
PM1_MEAN = (0.5, 0.5, 0.5)
PM1_STD = (0.5, 0.5, 0.5)


def encode_semantic_latent(model, image, layer_name):
    vision_model = model.siglip_model.vision_model
    hidden_states = vision_model.embeddings(image)

    attention_mask = None
    output_attentions = None
    layers = vision_model.encoder.layers
    target_idx = len(layers) - 2 if layer_name == "penultimate" else len(layers) - 1

    for i, encoder_layer in enumerate(layers):
        if vision_model.encoder.gradient_checkpointing and vision_model.encoder.training:
            layer_outputs = vision_model.encoder._gradient_checkpointing_func(
                encoder_layer.__call__,
                hidden_states,
                attention_mask,
                output_attentions,
            )
        else:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
            )
        hidden_states = layer_outputs[0]
        if i == target_idx:
            batch_size, seq_len, channels = hidden_states.shape
            side = int(seq_len**0.5)
            if side * side != seq_len:
                raise ValueError(f"VILA-U semantic latent sequence is not square: seq_len={seq_len}")
            return hidden_states.reshape(batch_size, side, side, channels)

    raise RuntimeError(f"Failed to extract VILA-U semantic latent from {layer_name} layer")


class VilaULinearAdapter(nn.Module):
    """Expose DINOv2-compatible intermediate layers for VILA-U spatial features."""

    def __init__(self, model, dtype, feature="semantic", semantic_layer="penultimate"):
        super().__init__()
        self.model = model
        self.dtype = dtype
        self.feature = feature
        self.semantic_layer = semantic_layer

    def _feature_map(self, images):
        if self.feature == "quantized":
            _code, z_q = self.model.encode_image(images.to(self.dtype))
            return z_q.float()

        return encode_semantic_latent(self.model, images.to(self.dtype), self.semantic_layer).float()

    def get_intermediate_layers(self, images, n, return_class_token=False):
        fmap = self._feature_map(images)
        patch_tokens = fmap.reshape(fmap.shape[0], -1, fmap.shape[-1]).contiguous()
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
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--train-dataset", dest="train_dataset_str", required=True)
    ap.add_argument("--val-dataset", dest="val_dataset_str", required=True)
    ap.add_argument("--test-datasets", dest="test_dataset_strs", nargs="+", default=None)
    ap.add_argument("--model-path", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--siglip-config-path", default=DEFAULT_SIGLIP_CONFIG)
    ap.add_argument("--vilau-path", default=os.path.join(IMAGE_SCRIPTS, "vila-u"))
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument(
        "--feature",
        default="semantic",
        choices=["semantic", "quantized"],
        help="semantic probes the pre-quant SigLIP latent; quantized probes VILA-U z_q.",
    )
    ap.add_argument("--semantic-layer", default="penultimate", choices=["penultimate", "last"])
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

    if args.output_dir is None:
        suffix = args.semantic_layer if args.feature == "semantic" else "zq"
        args.output_dir = os.path.join(OUTPUT_ROOT, f"vilau_7b_256_{args.feature}_{suffix}")

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
    tok.model.eval().requires_grad_(False)
    n_params = sum(p.numel() for p in tok.model.parameters())
    logger.info(
        f"Loaded VILA-U params={n_params / 1e6:.1f}M feature={args.feature} "
        f"semantic_layer={args.semantic_layer} image_size={tok.image_size} dtype={args.dtype}"
    )

    model = VilaULinearAdapter(
        tok.model,
        tok.dtype,
        feature=args.feature,
        semantic_layer=args.semantic_layer,
    ).cuda().eval()
    train_transform, eval_transform = _build_transforms(args.image_size, args.eval_resize_size, args.hflip_prob)
    autocast_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float}[args.dtype]

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
        autocast_dtype=autocast_dtype,
        resume=not args.no_resume,
        val_metric_type=MetricType.MEAN_ACCURACY,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


if __name__ == "__main__":
    main()
