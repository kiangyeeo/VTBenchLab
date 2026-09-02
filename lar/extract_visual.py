#!/usr/bin/env python
"""Extract deterministic patch-mean tokenizer features for LAR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from .data import WORKSPACE, image_path, write_lines, write_manifests
    from .model_adapters import load_patch_bundle
except ImportError:  # Direct execution: python lar/extract_visual.py
    from data import WORKSPACE, image_path, write_lines, write_manifests
    from model_adapters import load_patch_bundle


class ManifestImages(Dataset):
    def __init__(self, identifiers: list[str], image_set: str, transform):
        self.identifiers = identifiers
        self.image_set = image_set
        self.transform = transform

    def __len__(self) -> int:
        return len(self.identifiers)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = image_path(self.image_set, self.identifiers[index])
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-config", type=Path, default=WORKSPACE / "lar" / "configs" / "models.yaml")
    parser.add_argument("--models", nargs="+", default=None, help="Configured names; defaults to enabled models")
    parser.add_argument("--image-set", choices=("coco4618", "coco5000", "in1k10k"), required=True)
    parser.add_argument("--output-root", type=Path, default=WORKSPACE / "lar" / "features")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only; writes a .limitN stem")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip only complete FP32 arrays whose ID sidecar exactly matches this manifest.",
    )
    return parser.parse_args()


def load_specs(path: Path, selected: list[str] | None) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    specs = payload.get("models", [])
    by_name = {row["name"]: row for row in specs}
    names = selected or [row["name"] for row in specs if row.get("enabled", False)]
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise ValueError(f"Models not found in {path}: {missing}")
    return [by_name[name] for name in names]


def extract_one(spec: dict, identifiers: list[str], args: argparse.Namespace) -> Path:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    protocol_value = spec.get("probing_protocol")
    protocol = None if not protocol_value else WORKSPACE / protocol_value
    if protocol is not None:
        if not protocol.is_file():
            raise FileNotFoundError(f"Selected probing run does not exist: {protocol}")
        with protocol.open("r", encoding="utf-8") as handle:
            protocol_payload = json.load(handle)
        if protocol_payload.get("model") != spec["loader_name"]:
            raise RuntimeError(
                f"Protocol model {protocol_payload.get('model')} != loader {spec['loader_name']}"
            )

    bundle = load_patch_bundle(spec["loader_name"], device)
    dataset = ManifestImages(identifiers, args.image_set, bundle.eval_transform)
    batch_size = int(args.batch_size or spec.get("batch_size", 64))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    suffix = "" if args.limit is None else f".limit{len(identifiers)}"
    stem = f"{spec['name']}__{args.image_set}{suffix}"
    output_path = args.output_root / f"{stem}.npy"
    ids_path = args.output_root / f"{stem}.ids.txt"
    metadata_path = args.output_root / f"{stem}.meta.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite")

    mmap = None
    offset = 0
    token_count = None
    for images in loader:
        images = images.to(device, non_blocking=True)
        with torch.inference_mode(), bundle.autocast_context():
            features = bundle.encoder(images).float()
        if features.ndim != 2 or not torch.isfinite(features).all():
            raise RuntimeError(f"Invalid features from {spec['name']}: {tuple(features.shape)}")
        array = features.cpu().numpy().astype(np.float32, copy=False)
        if mmap is None:
            mmap = np.lib.format.open_memmap(
                output_path, mode="w+", dtype=np.float32, shape=(len(dataset), array.shape[1])
            )
        mmap[offset : offset + len(array)] = array
        offset += len(array)
        observed_tokens = bundle.encoder.last_n_tokens
        if observed_tokens is not None:
            if token_count is not None and token_count != observed_tokens:
                raise RuntimeError("Tokenizer returned a non-constant number of patch tokens")
            token_count = observed_tokens
        print(f"{spec['name']}: {offset}/{len(dataset)}", flush=True)
    if mmap is None or offset != len(dataset):
        raise RuntimeError(f"Incomplete extraction: wrote {offset}/{len(dataset)} rows")
    mmap.flush()
    del mmap
    write_lines(ids_path, identifiers)
    metadata = {
        "name": spec["name"],
        "loader_name": spec["loader_name"],
        "image_set": args.image_set,
        "N": len(dataset),
        "d": int(np.load(output_path, mmap_mode="r").shape[1]),
        "n_tokens": token_count,
        "pooling": bundle.representation,
        "transform": bundle.transform_description,
        "checkpoint_paths": bundle.checkpoint_paths,
        "probing_protocol": None if protocol is None else str(protocol),
        "seed": args.seed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_path


def complete_existing_path(spec: dict, identifiers: list[str], args: argparse.Namespace) -> Path | None:
    suffix = "" if args.limit is None else f".limit{len(identifiers)}"
    path = args.output_root / f"{spec['name']}__{args.image_set}{suffix}.npy"
    if not path.exists():
        return None
    ids_path = path.with_suffix(".ids.txt")
    metadata_path = path.with_suffix(".meta.json")
    if not ids_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"Incomplete existing feature sidecars for {path}")
    if ids_path.read_text(encoding="utf-8").splitlines() != identifiers:
        raise RuntimeError(f"Existing feature IDs do not match the requested manifest: {path}")
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2 or array.shape[0] != len(identifiers) or array.dtype != np.float32:
        raise RuntimeError(f"Invalid existing feature array: {path}, shape={array.shape}, dtype={array.dtype}")
    return path


def main() -> None:
    args = parse_args()
    if args.overwrite and args.skip_existing:
        raise ValueError("--overwrite and --skip-existing are mutually exclusive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    torch.manual_seed(args.seed)
    manifests = write_manifests(seed=args.seed)
    identifiers = manifests[args.image_set].read_text(encoding="utf-8").splitlines()
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        identifiers = identifiers[: args.limit]
    specs = load_specs(args.models_config, args.models)
    specs = specs[args.shard_index :: args.num_shards]
    print(
        f"shard {args.shard_index}/{args.num_shards}: {len(specs)} model(s)",
        flush=True,
    )
    for spec in specs:
        if args.skip_existing:
            existing = complete_existing_path(spec, identifiers, args)
            if existing is not None:
                print(f"SKIP complete {existing}", flush=True)
                continue
        print(extract_one(spec, identifiers, args))


if __name__ == "__main__":
    main()
