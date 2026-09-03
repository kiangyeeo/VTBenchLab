from __future__ import annotations

import argparse
import gc

import torch
from PIL import Image

from lar.model_adapters import load_spatial_bundle

from .utils import choose_names, load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="One-image spatial-token adapter smoke test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizers", nargs="+", default=["all"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    names = choose_names(args.tokenizers, config["tokenizers"])
    image_root = resolve_path(config, config["data"]["coco_train_images"])
    image_path = next(iter(sorted(image_root.glob("*.jpg"))), None)
    if image_path is None:
        raise FileNotFoundError(f"No JPG images under {image_root}")
    image = Image.open(image_path).convert("RGB")
    device = torch.device(args.device)
    for name in names:
        spec = config["tokenizers"][name]
        bundle = load_spatial_bundle(spec["loader_name"], device)
        pixels = bundle.eval_transform(image).unsqueeze(0).to(device)
        with torch.inference_mode(), bundle.autocast_context():
            tokens = bundle.encoder(pixels)
        if tokens.ndim != 3 or tokens.shape[0] != 1 or not torch.isfinite(tokens).all():
            raise RuntimeError(f"Invalid output for {name}: {tuple(tokens.shape)}")
        print(
            f"{name}: shape={tuple(tokens.shape)} dtype={tokens.dtype} "
            f"range=[{float(tokens.min()):.4g},{float(tokens.max()):.4g}]",
            flush=True,
        )
        del bundle, pixels, tokens
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
