from __future__ import annotations

import argparse
import gc
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .data import ImageResolver, load_examples
from .utils import (
    atomic_write_json,
    canonical_hash,
    choose_names,
    device_from_config,
    load_config,
    resolve_path,
)


class FinalNormPatchEncoder(nn.Module):
    """Return final normalized timm ViT patch tokens without prefix tokens."""

    def __init__(self, model: nn.Module, expected_tokens: int, expected_dim: int) -> None:
        super().__init__()
        self.model = model
        self.expected_tokens = expected_tokens
        self.expected_dim = expected_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model.get_intermediate_layers(
            images,
            n=1,
            return_prefix_tokens=True,
            norm=True,
        )
        final_output = outputs[-1]
        if isinstance(final_output, tuple):
            tokens, _prefix_tokens = final_output
        else:
            # Prefix-free ViTs (including timm SigLIP 2) return only patches
            # even when return_prefix_tokens=True is requested.
            tokens = final_output
        expected = (images.shape[0], self.expected_tokens, self.expected_dim)
        if tuple(tokens.shape) != expected:
            raise RuntimeError(
                f"Tokenizer returned {tuple(tokens.shape)}; expected {expected}"
            )
        if not bool(torch.isfinite(tokens).all().item()):
            raise RuntimeError("Tokenizer returned non-finite patch tokens")
        return tokens


class RAEv2PatchEncoder(nn.Module):
    """Return the normalized K=23 spatial latent consumed by RAE v2."""

    def __init__(
        self,
        model: nn.Module,
        latent_mean: torch.Tensor,
        latent_var: torch.Tensor,
        expected_tokens: int,
        expected_dim: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.model = model
        self.expected_tokens = expected_tokens
        self.expected_dim = expected_dim
        self.eps = float(eps)
        self.register_buffer(
            "pixel_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer("latent_mean", latent_mean.unsqueeze(0).float())
        self.register_buffer("latent_var", latent_var.unsqueeze(0).float())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized_images = (images.float() - self.pixel_mean) / self.pixel_std
        outputs = self.model.get_intermediate_layers(
            normalized_images.to(dtype=next(self.model.parameters()).dtype),
            n=tuple(range(1, 24)),
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        tokens = torch.stack(outputs, dim=0).mean(dim=0)
        tokens = tokens + outputs[-1].mean(dim=1, keepdim=True)
        batch_size = tokens.shape[0]
        latent = tokens.transpose(1, 2).reshape(
            batch_size, self.expected_dim, 16, 16
        ).float()
        expected_latent = (batch_size, self.expected_dim, 16, 16)
        if tuple(latent.shape) != expected_latent:
            raise RuntimeError(
                f"RAE v2 returned latent {tuple(latent.shape)}; expected {expected_latent}"
            )
        latent = (latent - self.latent_mean) / torch.sqrt(self.latent_var + self.eps)
        normalized_tokens = latent.flatten(2).transpose(1, 2).contiguous()
        expected = (batch_size, self.expected_tokens, self.expected_dim)
        if tuple(normalized_tokens.shape) != expected:
            raise RuntimeError(
                f"RAE v2 returned {tuple(normalized_tokens.shape)}; expected {expected}"
            )
        if not bool(torch.isfinite(normalized_tokens).all().item()):
            raise RuntimeError("RAE v2 returned non-finite patch tokens")
        return normalized_tokens


def _checkpoint_file(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if checkpoint.is_file():
        return checkpoint
    if checkpoint.is_dir():
        for filename in ("model.safetensors", "pytorch_model.bin", "checkpoint.pth"):
            candidate = checkpoint / filename
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No supported checkpoint found at {checkpoint}")


def _load_raev2_encoder(spec: dict[str, Any], device: torch.device) -> RAEv2PatchEncoder:
    checkpoint = _checkpoint_file(spec["checkpoint"])
    stats_path = Path(spec["stats"]).expanduser().resolve()
    source_path = Path(spec["source_path"]).expanduser().resolve()
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    if not (source_path / "hubconf.py").is_file():
        raise FileNotFoundError(f"Missing local DINOv3 hubconf.py under {source_path}")

    model = torch.hub.load(
        str(source_path),
        "dinov3_vitl16",
        source="local",
        trust_repo=True,
        skip_validation=True,
        weights=str(checkpoint),
    )
    if int(model.embed_dim) != 1024 or len(model.blocks) != 24:
        raise RuntimeError(
            f"Unexpected RAE v2 backbone: dim={model.embed_dim}, depth={len(model.blocks)}"
        )
    model.norm = nn.LayerNorm(1024, elementwise_affine=False)
    model = model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    latent_mean = stats.get("mean") if isinstance(stats, dict) else None
    latent_var = stats.get("var") if isinstance(stats, dict) else None
    expected_stats = (int(spec["hidden_dim"]), 16, 16)
    if (
        not torch.is_tensor(latent_mean)
        or not torch.is_tensor(latent_var)
        or tuple(latent_mean.shape) != expected_stats
        or tuple(latent_var.shape) != expected_stats
    ):
        mean_shape = None if not torch.is_tensor(latent_mean) else tuple(latent_mean.shape)
        var_shape = None if not torch.is_tensor(latent_var) else tuple(latent_var.shape)
        raise RuntimeError(
            f"RAE v2 stats mismatch: expected {expected_stats}, "
            f"got mean={mean_shape}, var={var_shape}"
        )
    if not bool(torch.isfinite(latent_mean).all() and torch.isfinite(latent_var).all()):
        raise RuntimeError("RAE v2 normalization statistics contain non-finite values")
    if not bool(torch.all(latent_var >= 0)):
        raise RuntimeError("RAE v2 normalization variance contains negative values")
    return RAEv2PatchEncoder(
        model,
        latent_mean,
        latent_var,
        expected_tokens=int(spec["token_count"]),
        expected_dim=int(spec["hidden_dim"]),
    ).to(device).eval().requires_grad_(False)


def _load_encoder(spec: dict[str, Any], device: torch.device) -> nn.Module:
    import timm
    from timm.models import load_checkpoint
    from timm.models.eva import checkpoint_filter_fn

    checkpoint = _checkpoint_file(spec["checkpoint"])
    loader = str(spec.get("loader", "perception"))
    if loader == "raev2":
        return _load_raev2_encoder(spec, device)
    create_kwargs: dict[str, Any] = {
        "pretrained": False,
        "num_classes": int(spec.get("num_classes", 0)),
        "dtype": torch.bfloat16,
    }
    config_path = Path(spec["checkpoint"]).resolve() / "config.json"
    if loader == "timm" and config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            local_config = json.load(handle)
        pretrained_cfg = local_config.get("pretrained_cfg")
        if isinstance(pretrained_cfg, dict):
            create_kwargs["pretrained_cfg"] = pretrained_cfg
    model = timm.create_model(
        spec["architecture"],
        **create_kwargs,
    )
    filter_fn = checkpoint_filter_fn if loader == "perception" else None
    load_checkpoint(model, str(checkpoint), strict=True, filter_fn=filter_fn)
    encoder = FinalNormPatchEncoder(
        model,
        expected_tokens=int(spec["token_count"]),
        expected_dim=int(spec["hidden_dim"]),
    )
    return encoder.to(device).eval().requires_grad_(False)


def _transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def _preprocess(spec: dict[str, Any], model: nn.Module):
    loader = str(spec.get("loader", "perception"))
    if loader == "perception":
        return _transform(int(spec["image_size"]))
    if loader == "raev2":
        image_size = int(spec["image_size"])
        return transforms.Compose(
            [
                transforms.Resize(
                    image_size,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
    from timm.data import create_transform, resolve_model_data_config

    data_config = resolve_model_data_config(model)
    resolved_size = int(data_config["input_size"][-1])
    expected_size = int(spec["image_size"])
    if resolved_size != expected_size:
        raise RuntimeError(
            f"Resolved input size {resolved_size} does not match configured {expected_size}"
        )
    return create_transform(**data_config, is_training=False)


def extract_one(config: dict[str, Any], name: str, force: bool = False) -> Path:
    device = device_from_config(config)
    raw_spec = config["tokenizers"][name]
    spec = dict(raw_spec)
    loader = str(spec.get("loader", "perception"))
    if loader != "registry":
        spec["checkpoint"] = str(resolve_path(config, spec["checkpoint"]))
        for key in ("stats", "source_path"):
            if key in spec:
                spec[key] = str(resolve_path(config, spec[key]))
    artifact_root = resolve_path(config, config["runtime"]["artifact_root"])
    output_dir = artifact_root / "tokens" / name
    metadata_path = output_dir / "cache.json"
    examples = load_examples(config)
    manifest_meta = json.loads(
        (artifact_root / "manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    identity = {
        "schema_version": 2,
        "tokenizer": name,
        "spec": spec,
        "records_sha256": manifest_meta["records_sha256"],
        "feature_dtype": config["runtime"]["feature_dtype"],
        "record_count": len(examples),
    }
    if loader != "registry":
        checkpoint = _checkpoint_file(spec["checkpoint"])
        identity.update(
            {
                "checkpoint_file": str(checkpoint),
                "checkpoint_size": checkpoint.stat().st_size,
                "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
            }
        )
    if loader == "raev2":
        stats_path = Path(spec["stats"])
        identity["stats_size"] = stats_path.stat().st_size
        identity["stats_mtime_ns"] = stats_path.stat().st_mtime_ns
    fingerprint = canonical_hash(identity)
    if metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint and metadata.get("complete"):
            print(f"Reusing complete token cache: {output_dir}")
            return output_dir
        raise RuntimeError(
            f"Existing token cache {output_dir} does not match this run; use --force"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: list[str] = []
    transform_description = ""
    representation = "final normalized last-block patch tokens after discarding prefix tokens"
    if loader == "registry":
        from lar.model_adapters import load_spatial_bundle

        bundle = load_spatial_bundle(str(spec["loader_name"]), device)
        encoder = bundle.encoder
        preprocess = bundle.eval_transform
        autocast_context = bundle.autocast_context
        checkpoint_paths = [str(path) for path in bundle.checkpoint_paths]
        transform_description = str(bundle.transform_description)
        representation = str(bundle.representation)
    else:
        encoder = _load_encoder(spec, device)
        preprocess = _preprocess(spec, encoder.model)
        autocast_context = lambda: torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        )
        checkpoint_paths = [str(identity["checkpoint_file"])]
    resolver = ImageResolver()
    batch_size = int(spec.get("extract_batch_size", 1))
    shard_size = int(config["runtime"]["token_shard_size"])
    if shard_size % batch_size:
        raise ValueError("token_shard_size must be divisible by extract_batch_size")
    feature_dtype = getattr(torch, str(config["runtime"]["feature_dtype"]))
    index: dict[str, dict[str, int | str]] = {}
    token_shape: tuple[int, int] | None = None

    for shard_start in range(0, len(examples), shard_size):
        shard_examples = examples[shard_start : shard_start + shard_size]
        batches = []
        for batch_start in range(0, len(shard_examples), batch_size):
            batch_examples = shard_examples[batch_start : batch_start + batch_size]
            images = torch.stack([preprocess(resolver.load(example)) for example in batch_examples])
            images = images.to(
                device=device,
                dtype=None if loader == "registry" else torch.bfloat16,
                non_blocking=True,
            )
            with torch.inference_mode(), autocast_context():
                tokens = encoder(images)
            if tokens.ndim != 3 or tokens.shape[0] != len(batch_examples):
                raise RuntimeError(
                    f"{name} returned {tuple(tokens.shape)}; expected [B,T,D]"
                )
            current_shape = (int(tokens.shape[1]), int(tokens.shape[2]))
            if token_shape is None:
                token_shape = current_shape
                configured_shape = (
                    spec.get("token_count"), spec.get("hidden_dim")
                )
                if all(value is not None for value in configured_shape) and current_shape != (
                    int(configured_shape[0]), int(configured_shape[1])
                ):
                    raise RuntimeError(
                        f"{name} returned T,D={current_shape}; configured {configured_shape}"
                    )
            elif current_shape != token_shape:
                raise RuntimeError(
                    f"{name} token shape changed from {token_shape} to {current_shape}"
                )
            batches.append(tokens.to(device="cpu", dtype=feature_dtype))
        shard_tokens = torch.cat(batches, dim=0).contiguous()
        shard_number = shard_start // shard_size
        shard_name = f"shard_{shard_number:05d}.safetensors"
        temporary = shard_dir / f"{shard_name}.tmp"
        destination = shard_dir / shard_name
        save_file({"tokens": shard_tokens}, temporary)
        os.replace(temporary, destination)
        for offset, example in enumerate(shard_examples):
            index[example.record_id] = {"shard": shard_name, "offset": offset}
        print(
            f"{name}: cached {min(shard_start + len(shard_examples), len(examples))}/"
            f"{len(examples)} records"
        )

    if token_shape is None:
        raise RuntimeError("Cannot cache an empty manifest")
    atomic_write_json(output_dir / "index.json", index)
    atomic_write_json(
        metadata_path,
        {
            **identity,
            "fingerprint": fingerprint,
            "complete": True,
            "shape": [len(examples), token_shape[0], token_shape[1]],
            "representation": representation,
            "transform_description": transform_description,
            "checkpoint_paths": checkpoint_paths,
        },
    )
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir


class TokenCache:
    def __init__(self, root: str | Path, max_cached_shards: int = 2) -> None:
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        self.metadata = json.loads((self.root / "cache.json").read_text(encoding="utf-8"))
        self._load_shard = lru_cache(maxsize=max_cached_shards)(self._load_shard_uncached)

    def _load_shard_uncached(self, name: str) -> torch.Tensor:
        return load_file(self.root / "shards" / name, device="cpu")["tokens"]

    def get(self, record_id: str) -> torch.Tensor:
        location = self.index[record_id]
        return self._load_shard(str(location["shard"]))[int(location["offset"])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache PE patch tokens for the fixed manifest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizers", nargs="+", default=["all"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    names = choose_names(args.tokenizers, config["tokenizers"])
    for name in names:
        extract_one(config, name, force=args.force)


if __name__ == "__main__":
    main()
