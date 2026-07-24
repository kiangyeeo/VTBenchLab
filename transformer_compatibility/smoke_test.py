#!/usr/bin/env python
"""CPU/CUDA smoke test for both MC1 sequence extractors and all readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parent
WORKSPACE = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from extractors import build_tokenizer
from readouts import build_readout, trainable_parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "imagenet_mc1_protocol.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.expanduser().resolve().open() as handle:
        config = json.load(handle)
    device = torch.device(args.device)
    image_size = int(config["dataset"]["image_size"])
    num_classes = int(config["dataset"]["num_classes"])

    results = []
    for model_name, model_config in config["models"].items():
        tokenizer = build_tokenizer(model_config, config["surface"], WORKSPACE)
        tokenizer.to(device).eval()
        images = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
        with torch.no_grad():
            tokens = tokenizer(images)
        expected_shape = (
            args.batch_size,
            int(model_config["token_count"]),
            int(model_config["input_dim"]),
        )
        if tuple(tokens.shape) != expected_shape:
            raise AssertionError(f"Expected {expected_shape}, got {tuple(tokens.shape)}")

        for readout_name in ("gap_linear", "gap_mlp", "transformer"):
            readout, metadata = build_readout(
                name=readout_name,
                input_dim=int(model_config["input_dim"]),
                num_classes=num_classes,
                readout_configs=config["readouts"],
            )
            readout.to(device).train()
            logits = readout(tokens)
            if tuple(logits.shape) != (args.batch_size, num_classes):
                raise AssertionError(
                    f"Expected logits {(args.batch_size, num_classes)}, "
                    f"got {tuple(logits.shape)}"
                )
            loss = logits.square().mean()
            loss.backward()
            if not any(
                parameter.grad is not None
                for parameter in readout.parameters()
                if parameter.requires_grad
            ):
                raise AssertionError(f"No gradients for {model_name}/{readout_name}")
            results.append(
                {
                    "model": model_name,
                    "readout": readout_name,
                    "token_shape": list(tokens.shape),
                    "logit_shape": list(logits.shape),
                    "trainable_parameters": trainable_parameter_count(readout),
                    "metadata": metadata,
                }
            )
            del readout, logits, loss
        del tokenizer, tokens, images

    print(json.dumps({"status": "ok", "checks": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

