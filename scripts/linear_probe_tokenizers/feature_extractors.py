"""Frozen feature extractors for the tokenizer linear-probing baseline.

Each extractor returns exactly one preselected visual representation shaped
``[batch, feature_dim]``.  Pooling performed here is part of that representation;
the evaluator does not perform DINOv2's multi-block or CLS/patch readout search.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable

import torch
from torch import nn

from dinov2.data.transforms import make_classification_eval_transform, make_classification_train_transform


PM1_MEAN = (0.5, 0.5, 0.5)
PM1_STD = (0.5, 0.5, 0.5)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IDENTITY_MEAN = (0.0, 0.0, 0.0)
IDENTITY_STD = (1.0, 1.0, 1.0)

MC1_SPECS = {
    "mc1_b32_224_400m": (
        "vit_base_patch32_clip_224.metaclip_400m",
        "mc1_b32_224_400m_checkpoint",
        "final normalized CLS before the MetaCLIP 768-to-512 projection",
    ),
    "mc1_b16_224_400m": (
        "vit_base_patch16_clip_224.metaclip_400m",
        "mc1_b16_224_400m_checkpoint",
        "final normalized CLS before the MetaCLIP 768-to-512 projection",
    ),
    "mc1_l14_224_400m": (
        "vit_large_patch14_clip_224.metaclip_400m",
        "mc1_l14_224_400m_checkpoint",
        "final normalized CLS before the MetaCLIP 1024-to-768 projection",
    ),
    "mc1_b32_224_2.5b": (
        "vit_base_patch32_clip_224.metaclip_2pt5b",
        "mc1_b32_224_2_5b_checkpoint",
        "final normalized CLS before the MetaCLIP 768-to-512 projection",
    ),
    "mc1_b16_224_2.5b": (
        "vit_base_patch16_clip_224.metaclip_2pt5b",
        "mc1_b16_224_2_5b_checkpoint",
        "final normalized CLS before the MetaCLIP 768-to-512 projection",
    ),
    "mc1_l14_224_2.5b": (
        "vit_large_patch14_clip_224.metaclip_2pt5b",
        "mc1_l14_224_2_5b_checkpoint",
        "final normalized CLS before the MetaCLIP 1024-to-768 projection",
    ),
    "mc1_h14_224_2.5b": (
        "vit_huge_patch14_clip_224.metaclip_2pt5b",
        "mc1_h14_224_2_5b_checkpoint",
        "final normalized CLS before the MetaCLIP 1280-to-1024 projection",
    ),
    "mc1_g14_224_2.5b": (
        "vit_gigantic_patch14_clip_224.metaclip_2pt5b",
        "mc1_g14_224_2_5b_checkpoint",
        "final normalized CLS before the MetaCLIP 1664-to-1280 projection",
    ),
    "mc1_h14_224_v1.2": (
        "vit_huge_patch14_clip_224.metaclip_altogether",
        "mc1_h14_224_v1_2_checkpoint",
        "final normalized CLS before the MetaCLIP 1280-to-1024 projection",
    ),
}

MC2_TIMM_SPECS = {
    "mc2_h14_378": (
        "vit_huge_patch14_clip_378.metaclip2_worldwide",
        "mc2_h14_378_checkpoint",
        "final normalized CLS before the MetaCLIP 2 1280-to-1024 projection",
    ),
    "mc2_g14_224": (
        "vit_gigantic_patch14_clip_224.metaclip2_worldwide",
        "mc2_g14_224_checkpoint",
        "final normalized CLS before the MetaCLIP 2 1664-to-1280 projection",
    ),
    "mc2_g14_378": (
        "vit_gigantic_patch14_clip_378.metaclip2_worldwide",
        "mc2_g14_378_checkpoint",
        "final normalized CLS before the MetaCLIP 2 1664-to-1280 projection",
    ),
}

# checkpoint arg, image size, patch size, width, depth, projection output dim
MC2_DISTILLED_SPECS = {
    "mc2_s16_224": ("mc2_s16_224_checkpoint", 224, 16, 384, 12, 384),
    "mc2_s16_384": ("mc2_s16_384_checkpoint", 384, 16, 384, 12, 384),
    "mc2_s16_224_mt5": ("mc2_s16_224_mt5_checkpoint", 224, 16, 384, 12, 384),
    "mc2_m16_224": ("mc2_m16_224_checkpoint", 224, 16, 512, 12, 512),
    "mc2_m16_384": ("mc2_m16_384_checkpoint", 384, 16, 512, 12, 512),
    "mc2_m16_224_mt5": ("mc2_m16_224_mt5_checkpoint", 224, 16, 512, 12, 512),
    "mc2_b32_224": ("mc2_b32_224_checkpoint", 224, 32, 768, 12, 512),
    "mc2_b32_384": ("mc2_b32_384_checkpoint", 384, 32, 768, 12, 512),
    "mc2_b32_224_mt5": ("mc2_b32_224_mt5_checkpoint", 224, 32, 768, 12, 512),
    "mc2_b16_224": ("mc2_b16_224_checkpoint", 224, 16, 768, 12, 512),
    "mc2_b16_384": ("mc2_b16_384_checkpoint", 384, 16, 768, 12, 512),
    "mc2_l14_224": ("mc2_l14_224_checkpoint", 224, 14, 1024, 24, 768),
}

SIGLIP2_B_SPECS = {
    "siglip2_b32_256": (
        "vit_base_patch32_siglip_256",
        "siglip2_b32_256_model_path",
        256,
    ),
    "siglip2_b16_224": (
        "vit_base_patch16_siglip_224",
        "siglip2_b16_224_model_path",
        224,
    ),
    "siglip2_b16_256": (
        "vit_base_patch16_siglip_256",
        "siglip2_b16_256_model_path",
        256,
    ),
    "siglip2_b16_384": (
        "vit_base_patch16_siglip_384",
        "siglip2_b16_384_model_path",
        384,
    ),
    "siglip2_b16_512": (
        "vit_base_patch16_siglip_512",
        "siglip2_b16_512_model_path",
        512,
    ),
}

RAEV2_SPECS = {
    "dinov3": {
        "stats": "stage1/imagenet/dinov3l-k1/stats.pt",
        "representation": (
            "spatial mean of the normalized RAEv2 DINOv3-L/16 K=1 tokenizer latent"
        ),
    },
    "raev2": {
        "stats": "stage1/imagenet/dinov3l-k23/stats.pt",
        "representation": (
            "spatial mean of the normalized RAEv2 DINOv3-L/16 K=23 "
            "multi-layer tokenizer latent"
        ),
    },
    "ijepa": {
        "stats": "stage1/imagenet/jepa-h-k1/stats.pt",
        "representation": (
            "spatial mean of the normalized RAEv2 I-JEPA-H/14 K=1 tokenizer latent"
        ),
    },
}

DINOV3_L_CHECKPOINT = (
    "encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
IJEPA_H_CHECKPOINT = "encoders/ijepa/ijepa_vith.pth"
RAEV2_K23_LAYERS = tuple(range(1, 24))


@dataclass
class FeatureBundle:
    encoder: nn.Module
    train_transform: Callable
    eval_transform: Callable
    autocast_context: Callable
    representation: str
    transform_description: str
    backbone_precision: str
    checkpoint_paths: list[str]


class MetaCLIPEncoder(nn.Module):
    """Final normalized MetaCLIP CLS, before the vision projection head."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model.get_intermediate_layers(
            images,
            n=1,
            return_prefix_tokens=True,
            norm=True,
        )
        _patch_tokens, prefix_tokens = outputs[-1]
        if prefix_tokens.ndim != 3 or prefix_tokens.shape[1] < 1:
            raise RuntimeError(f"Unexpected MetaCLIP prefix-token shape: {tuple(prefix_tokens.shape)}")
        return prefix_tokens[:, 0].float()


class MetaCLIP2DistilledEncoder(nn.Module):
    """Final normalized CLS from an OpenAI-style MetaCLIP 2 visual tower."""

    def __init__(self, visual: nn.Module):
        super().__init__()
        if visual.proj is not None:
            raise ValueError("MetaCLIP 2 visual projection must be disabled for linear probing")
        self.visual = visual

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual(images)
        if features.ndim != 2:
            raise RuntimeError(f"Unexpected MetaCLIP 2 CLS shape: {tuple(features.shape)}")
        return features.float()


class OpenAIClipEncoder(nn.Module):
    """Final post-LayerNorm OpenAI CLIP CLS, before visual_projection."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=images, return_dict=True)
        if outputs.pooler_output is None or outputs.pooler_output.ndim != 2:
            shape = None if outputs.pooler_output is None else tuple(outputs.pooler_output.shape)
            raise RuntimeError(f"Unexpected OpenAI CLIP pooled-output shape: {shape}")
        return outputs.pooler_output.float()


class SigLIP2MAPEncoder(nn.Module):
    """Native SigLIP 2 MAP output returned directly by ``model(images)``."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        if features.ndim != 2:
            raise RuntimeError(f"Unexpected SigLIP 2 MAP shape: {tuple(features.shape)}")
        return features.float()


class RAEv2LatentEncoder(nn.Module):
    """Pool the exact normalized spatial latent consumed by an RAEv2 decoder."""

    def __init__(
        self,
        model: nn.Module,
        variant: str,
        latent_mean: torch.Tensor,
        latent_var: torch.Tensor,
        eps: float = 1e-5,
    ):
        super().__init__()
        if variant not in RAEV2_SPECS:
            raise ValueError(f"Unsupported RAEv2 tokenizer variant: {variant}")
        self.model = model
        self.variant = variant
        self.eps = float(eps)
        self.register_buffer(
            "pixel_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer("latent_mean", latent_mean.unsqueeze(0).float())
        self.register_buffer("latent_var", latent_var.unsqueeze(0).float())

    def _encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.pixel_mean) / self.pixel_std
        if self.variant == "ijepa":
            # Match RAEv2's JEPAEncoder: native RAE input is 256, while the
            # I-JEPA backbone itself receives 224x224.
            images = nn.functional.interpolate(images, 224, mode="bicubic")
            return self.model(images)

        # DINOv3's RAEv2 transform is a square resize to the native 256 input.
        if images.shape[-2:] != (256, 256):
            images = nn.functional.interpolate(
                images,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        if self.variant == "dinov3":
            return self.model.forward_features(images)["x_norm_patchtokens"]

        outputs = self.model.get_intermediate_layers(
            images,
            n=RAEV2_K23_LAYERS,
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
        return patch_tokens + outputs[-1].mean(dim=1, keepdim=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self._encode_tokens(images)
        if tokens.ndim != 3:
            raise RuntimeError(
                f"Unexpected {self.variant} tokenizer-token shape: {tuple(tokens.shape)}"
            )
        batch_size, token_count, feature_dim = tokens.shape
        grid_size = math.isqrt(token_count)
        if grid_size * grid_size != token_count:
            raise RuntimeError(
                f"{self.variant} returned a non-square token grid: {token_count}"
            )
        latent = tokens.transpose(1, 2).reshape(
            batch_size,
            feature_dim,
            grid_size,
            grid_size,
        )
        expected_shape = tuple(self.latent_mean.shape[1:])
        if tuple(latent.shape[1:]) != expected_shape:
            raise RuntimeError(
                f"{self.variant} latent shape mismatch: expected {expected_shape}, "
                f"got {tuple(latent.shape[1:])}"
            )
        latent = (latent - self.latent_mean) / torch.sqrt(self.latent_var + self.eps)
        return latent.mean(dim=(2, 3)).float()


class TokLIPEncoder(nn.Module):
    """Mean of TokLIP's final normalized semantic tokens."""

    def __init__(self, trunk: nn.Module, encode_tokens: Callable):
        super().__init__()
        self.trunk = trunk
        self.encode_tokens = encode_tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_tokens(self.trunk, images)
        return tokens.mean(dim=1).float()


class UniTokEncoder(nn.Module):
    """Mean-pooled quantized UniTok tokens after fc_norm, before projection."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.model.encoder(images).float()
        tokens = self.model.quant_proj(tokens)
        indices = self.model.quantizer.f_to_idx(tokens)
        tokens = self.model.quantizer.idx_to_f(indices)
        tokens = self.model.post_quant_proj(tokens)
        return self.model.fc_norm(tokens.mean(dim=1)).float()


def _encode_vilau_penultimate(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    vision_model = model.siglip_model.vision_model
    hidden_states = vision_model.embeddings(images)
    target_index = len(vision_model.encoder.layers) - 2

    for index, encoder_layer in enumerate(vision_model.encoder.layers):
        layer_outputs = encoder_layer(
            hidden_states,
            None,
            output_attentions=None,
        )
        hidden_states = layer_outputs[0]
        if index == target_index:
            return hidden_states.mean(dim=1).float()

    raise RuntimeError("Failed to extract VILA-U penultimate SigLIP tokens")


class VilaUEncoder(nn.Module):
    """Mean of the penultimate SigLIP tokens used by the VILA-U tokenizer."""

    def __init__(self, model: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return _encode_vilau_penultimate(self.model, images.to(self.dtype))


def _pm1_transforms(image_size: int):
    train_transform = make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=0.5,
        mean=PM1_MEAN,
        std=PM1_STD,
    )
    eval_transform = make_classification_eval_transform(
        resize_size=image_size,
        crop_size=image_size,
        mean=PM1_MEAN,
        std=PM1_STD,
    )
    return train_transform, eval_transform


def _resolve_checkpoint(path: str) -> str:
    checkpoint = Path(path).expanduser().resolve()
    if checkpoint.is_file():
        return str(checkpoint)
    if checkpoint.is_dir():
        for name in ("model.safetensors", "pytorch_model.bin", "open_clip_pytorch_model.bin", "checkpoint.pth"):
            candidate = checkpoint / name
            if candidate.is_file():
                return str(candidate)
    raise FileNotFoundError(f"No supported checkpoint found at {checkpoint}")


def _load_metaclip(
    model_name: str,
    checkpoint_path: str,
    representation: str,
    device: torch.device,
) -> FeatureBundle:
    import timm
    from timm.data import create_transform, resolve_model_data_config
    from timm.models import load_checkpoint

    checkpoint = _resolve_checkpoint(checkpoint_path)
    model = timm.create_model(model_name, pretrained=False)
    load_checkpoint(model, checkpoint, strict=True)
    data_config = resolve_model_data_config(model)
    image_size = int(data_config["input_size"][-1])
    mean = tuple(float(value) for value in data_config["mean"])
    std = tuple(float(value) for value in data_config["std"])

    # Deliberately use the DINO classification augmentation here.  timm's
    # training transform adds ColorJitter, which would make this model's probe
    # use a different augmentation protocol from the other tokenizers.
    train_transform = make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=0.5,
        mean=mean,
        std=std,
    )
    eval_transform = create_transform(**data_config, is_training=False)
    encoder = MetaCLIPEncoder(model).to(device).eval().requires_grad_(False)
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=nullcontext,
        representation=representation,
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=timm deterministic native transform, mean={mean}, std={std}, no ColorJitter"
        ),
        backbone_precision="float32",
        checkpoint_paths=[checkpoint],
    )


def _load_metaclip2_distilled(
    checkpoint_path: str,
    image_size: int,
    patch_size: int,
    width: int,
    depth: int,
    projection_dim: int,
    device: torch.device,
) -> FeatureBundle:
    clip_root = Path(__file__).resolve().parents[2] / "CLIP"
    if str(clip_root) not in sys.path:
        sys.path.insert(0, str(clip_root))
    from clip.model import VisionTransformer

    checkpoint = _resolve_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError(f"Expected a MetaCLIP 2 state_dict checkpoint at {checkpoint}")
    state_dict = payload["state_dict"]
    visual_state = {
        key.removeprefix("visual."): value
        for key, value in state_dict.items()
        if key.startswith("visual.")
    }
    required_keys = {"conv1.weight", "positional_embedding", "proj"}
    missing_required = sorted(required_keys - visual_state.keys())
    if missing_required:
        raise RuntimeError(f"Missing MetaCLIP 2 visual keys: {missing_required}")

    actual_width = int(visual_state["conv1.weight"].shape[0])
    actual_patch_size = int(visual_state["conv1.weight"].shape[-1])
    patch_tokens = int(visual_state["positional_embedding"].shape[0]) - 1
    grid_size = math.isqrt(patch_tokens)
    if grid_size * grid_size != patch_tokens:
        raise RuntimeError(f"Non-square MetaCLIP 2 positional grid: {patch_tokens} tokens")
    actual_image_size = grid_size * actual_patch_size
    actual_depth = len(
        {
            key.split(".")[2]
            for key in visual_state
            if key.startswith("transformer.resblocks.") and key.endswith(".attn.in_proj_weight")
        }
    )
    actual_projection_shape = tuple(visual_state["proj"].shape)
    expected = (image_size, patch_size, width, depth, (width, projection_dim))
    actual = (
        actual_image_size,
        actual_patch_size,
        actual_width,
        actual_depth,
        actual_projection_shape,
    )
    if actual != expected:
        raise RuntimeError(f"MetaCLIP 2 visual architecture mismatch: expected={expected}, actual={actual}")

    visual = VisionTransformer(
        input_resolution=image_size,
        patch_size=patch_size,
        width=width,
        layers=depth,
        heads=width // 64,
        output_dim=projection_dim,
    )
    # The distilled MetaCLIP 2 model configs use standard GELU rather than
    # the QuickGELU variants used by MetaCLIP 1.
    for block in visual.transformer.resblocks:
        block.mlp.gelu = nn.GELU()
    visual.load_state_dict(visual_state, strict=True)
    visual.proj = None
    del payload, state_dict, visual_state

    encoder = MetaCLIP2DistilledEncoder(visual).to(device).eval().requires_grad_(False)
    train_transform = make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=0.5,
        mean=CLIP_MEAN,
        std=CLIP_STD,
    )
    eval_transform = make_classification_eval_transform(
        resize_size=image_size,
        crop_size=image_size,
        mean=CLIP_MEAN,
        std=CLIP_STD,
    )
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=nullcontext,
        representation=(
            f"final normalized CLS before the MetaCLIP 2 {width}-to-{projection_dim} projection"
        ),
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=Resize({image_size})+CenterCrop({image_size}), "
            f"mean={CLIP_MEAN}, std={CLIP_STD}"
        ),
        backbone_precision="float32",
        checkpoint_paths=[checkpoint],
    )


def _load_siglip2_map(
    model_path: str,
    expected_architecture: str,
    expected_image_size: int,
    device: torch.device,
) -> FeatureBundle:
    import timm
    from timm.data import create_transform, resolve_model_data_config
    from timm.models import load_checkpoint

    model_dir = Path(model_path).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing SigLIP 2 model directory: {model_dir}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing SigLIP 2 config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    expected_config = {
        "architecture": expected_architecture,
        "num_classes": 0,
        "num_features": 768,
        "global_pool": "map",
    }
    actual_config = {key: config.get(key) for key in expected_config}
    if actual_config != expected_config:
        raise RuntimeError(
            f"SigLIP 2 model config mismatch: expected={expected_config}, actual={actual_config}"
        )
    pretrained_cfg = config.get("pretrained_cfg")
    if not isinstance(pretrained_cfg, dict):
        raise RuntimeError(f"Missing pretrained_cfg object in {config_path}")
    input_size = pretrained_cfg.get("input_size")
    if input_size != [3, expected_image_size, expected_image_size]:
        raise RuntimeError(
            f"SigLIP 2 input-size mismatch: expected "
            f"{[3, expected_image_size, expected_image_size]}, got {input_size}"
        )

    checkpoint = _resolve_checkpoint(str(model_dir))
    model = timm.create_model(
        expected_architecture,
        pretrained=False,
        pretrained_cfg=pretrained_cfg,
        num_classes=0,
        global_pool="map",
    )
    load_checkpoint(model, checkpoint, strict=True)
    if getattr(model, "attn_pool", None) is None:
        raise RuntimeError("SigLIP 2 model does not expose the expected MAP attention pool")

    data_config = resolve_model_data_config(model)
    image_size = int(data_config["input_size"][-1])
    if image_size != expected_image_size:
        raise RuntimeError(
            f"Resolved SigLIP 2 input size {image_size} does not match {expected_image_size}"
        )
    mean = tuple(float(value) for value in data_config["mean"])
    std = tuple(float(value) for value in data_config["std"])
    train_transform = make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=0.5,
        mean=mean,
        std=std,
    )
    eval_transform = create_transform(**data_config, is_training=False)
    encoder = SigLIP2MAPEncoder(model).to(device).eval().requires_grad_(False)
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=nullcontext,
        representation=(
            "native SigLIP 2 model(images) [B,D] output after MAP attention pooling"
        ),
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=timm deterministic native transform, mean={mean}, std={std}, no ColorJitter"
        ),
        backbone_precision="float32",
        checkpoint_paths=[checkpoint, str(config_path)],
    )


def _load_dinov3_l(
    dinov3_path: Path,
    checkpoint: Path,
    device: torch.device,
) -> nn.Module:
    if not (dinov3_path / "hubconf.py").is_file():
        raise FileNotFoundError(f"Missing local DINOv3 repository: {dinov3_path}")
    model = torch.hub.load(
        str(dinov3_path),
        "dinov3_vitl16",
        source="local",
        trust_repo=True,
        skip_validation=True,
        weights=str(checkpoint),
    )
    if int(model.embed_dim) != 1024 or len(model.blocks) != 24:
        raise RuntimeError(
            "Unexpected DINOv3-L architecture: "
            f"embed_dim={model.embed_dim}, depth={len(model.blocks)}"
        )
    # This is part of RAEv2's released representation: its DINOv3 wrapper
    # deliberately removes the final norm's affine parameters.
    model.norm = nn.LayerNorm(1024, elementwise_affine=False)
    return model.to(device).eval().requires_grad_(False)


def _load_ijepa_h(
    raev2_path: Path,
    checkpoint: Path,
    device: torch.device,
) -> nn.Module:
    raev2_src = raev2_path / "src"
    if not (raev2_src / "encoders" / "models" / "jepa.py").is_file():
        raise FileNotFoundError(f"Missing RAEv2 I-JEPA source: {raev2_src}")
    if str(raev2_src) not in sys.path:
        sys.path.insert(0, str(raev2_src))
    from encoders.models.jepa import vit_huge

    model = vit_huge(img_size=[224, 224], patch_size=14)
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    encoder_state = payload.get("encoder") if isinstance(payload, dict) else None
    if not isinstance(encoder_state, dict):
        raise RuntimeError(f"Missing I-JEPA encoder state_dict in {checkpoint}")
    if not encoder_state or any(not key.startswith("module.") for key in encoder_state):
        raise RuntimeError(f"Unexpected I-JEPA encoder key layout in {checkpoint}")
    encoder_state = {
        key.removeprefix("module."): value
        for key, value in encoder_state.items()
    }
    model.load_state_dict(encoder_state, strict=True)
    del payload, encoder_state
    if int(model.embed_dim) != 1280 or len(model.blocks) != 32:
        raise RuntimeError(
            "Unexpected I-JEPA-H architecture: "
            f"embed_dim={model.embed_dim}, depth={len(model.blocks)}"
        )
    return model.to(device).eval().requires_grad_(False)


def _load_raev2_variant(args, device: torch.device, variant: str) -> FeatureBundle:
    spec = RAEV2_SPECS[variant]
    model_root = Path(args.raev2_model_root).expanduser().resolve()
    raev2_path = Path(args.raev2_path).expanduser().resolve()
    dinov3_path = Path(args.dinov3_path).expanduser().resolve()
    stats_path = model_root / spec["stats"]
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing {variant} normalization statistics: {stats_path}")

    if variant in {"dinov3", "raev2"}:
        checkpoint = model_root / DINOV3_L_CHECKPOINT
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing DINOv3-L checkpoint: {checkpoint}")
        model = _load_dinov3_l(dinov3_path, checkpoint, device)
        expected_feature_dim = 1024
    else:
        checkpoint = model_root / IJEPA_H_CHECKPOINT
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing I-JEPA-H checkpoint: {checkpoint}")
        model = _load_ijepa_h(raev2_path, checkpoint, device)
        expected_feature_dim = 1280

    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    latent_mean = stats.get("mean") if isinstance(stats, dict) else None
    latent_var = stats.get("var") if isinstance(stats, dict) else None
    expected_stats_shape = (expected_feature_dim, 16, 16)
    if (
        not torch.is_tensor(latent_mean)
        or not torch.is_tensor(latent_var)
        or tuple(latent_mean.shape) != expected_stats_shape
        or tuple(latent_var.shape) != expected_stats_shape
    ):
        mean_shape = None if not torch.is_tensor(latent_mean) else tuple(latent_mean.shape)
        var_shape = None if not torch.is_tensor(latent_var) else tuple(latent_var.shape)
        raise RuntimeError(
            f"{variant} normalization-stat shape mismatch: expected {expected_stats_shape}, "
            f"got mean={mean_shape}, var={var_shape}"
        )
    if not torch.isfinite(latent_mean).all() or not torch.isfinite(latent_var).all():
        raise RuntimeError(f"{variant} normalization statistics contain non-finite values")
    if not torch.all(latent_var >= 0):
        raise RuntimeError(f"{variant} normalization variance contains negative values")

    encoder = RAEv2LatentEncoder(
        model,
        variant,
        latent_mean,
        latent_var,
    ).to(device).eval().requires_grad_(False)
    train_transform = make_classification_train_transform(
        crop_size=256,
        hflip_prob=0.5,
        mean=IDENTITY_MEAN,
        std=IDENTITY_STD,
    )
    eval_transform = make_classification_eval_transform(
        resize_size=256,
        crop_size=256,
        mean=IDENTITY_MEAN,
        std=IDENTITY_STD,
    )
    source_paths = [str(raev2_path)]
    if variant in {"dinov3", "raev2"}:
        source_paths.append(str(dinov3_path))
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=nullcontext,
        representation=spec["representation"],
        transform_description=(
            "train=RandomResizedCrop(256)+HorizontalFlip(0.5), "
            "eval=Resize(256)+CenterCrop(256), transform output in [0,1]; "
            f"encoder-internal normalization mean={IMAGENET_MEAN}, std={IMAGENET_STD}"
        ),
        backbone_precision="float32",
        checkpoint_paths=[str(checkpoint), str(stats_path), *source_paths],
    )


def _processor_square_size(value, name: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        if "shortest_edge" in value:
            return int(value["shortest_edge"])
        height = value.get("height")
        width = value.get("width")
        if height is not None and width is not None and int(height) == int(width):
            return int(height)
    raise ValueError(f"Expected a square CLIP image-processor {name}, got {value!r}")


def _load_clip_openai_l14(args, device: torch.device) -> FeatureBundle:
    from transformers import CLIPImageProcessor, CLIPVisionModel

    model_path = Path(args.clip_openai_model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Missing OpenAI CLIP model directory: {model_path}")
    checkpoint = _resolve_checkpoint(str(model_path))
    model, loading_info = CLIPVisionModel.from_pretrained(
        str(model_path),
        local_files_only=True,
        output_loading_info=True,
    )
    unexpected_vision_keys = [
        key for key in loading_info["unexpected_keys"] if key.startswith("vision_model.")
    ]
    if (
        loading_info["missing_keys"]
        or loading_info["mismatched_keys"]
        or loading_info["error_msgs"]
        or unexpected_vision_keys
    ):
        raise RuntimeError(f"Failed to strictly load the OpenAI CLIP vision tower: {loading_info}")
    processor = CLIPImageProcessor.from_pretrained(str(model_path), local_files_only=True)

    image_size = int(model.config.image_size)
    crop_size = _processor_square_size(processor.crop_size, "crop_size")
    resize_size = _processor_square_size(processor.size, "size")
    if crop_size != image_size:
        raise ValueError(
            f"CLIP processor crop size {crop_size} does not match model image size {image_size}"
        )
    mean = tuple(float(value) for value in processor.image_mean)
    std = tuple(float(value) for value in processor.image_std)
    train_transform = make_classification_train_transform(
        crop_size=image_size,
        hflip_prob=0.5,
        mean=mean,
        std=std,
    )
    eval_transform = make_classification_eval_transform(
        resize_size=resize_size,
        crop_size=image_size,
        mean=mean,
        std=std,
    )
    encoder = OpenAIClipEncoder(model).to(device).eval().requires_grad_(False)
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=nullcontext,
        representation="final post-LayerNorm CLS before the OpenAI CLIP 1024-to-768 visual projection",
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=Resize({resize_size})+CenterCrop({image_size}), mean={mean}, std={std}"
        ),
        backbone_precision="float32",
        checkpoint_paths=[checkpoint],
    )


def _load_toklip(args, device: torch.device, variant: str) -> FeatureBundle:
    if str(args.image_scripts) not in sys.path:
        sys.path.insert(0, str(args.image_scripts))
    from toklip_rec_common import encode_toklip_semantic_tokens, load_toklip_semantic_model

    is_small = variant == "s"
    image_size = 256 if is_small else 384
    model_name = "toklip_s" if is_small else "toklip_l"
    checkpoint = args.toklip_s_checkpoint if is_small else args.toklip_l_checkpoint
    checkpoint = str(Path(checkpoint).expanduser().resolve())
    vq_checkpoint = str(Path(args.toklip_vq_checkpoint).expanduser().resolve())
    trunk = load_toklip_semantic_model(
        SimpleNamespace(
            toklip_path=str(Path(args.toklip_path).expanduser().resolve()),
            toklip_ckpt_path=checkpoint,
            vq_ckpt_path=vq_checkpoint,
            model_name=model_name,
            toklip_model_config=None,
        ),
        device=str(device),
    )
    encoder = TokLIPEncoder(trunk, encode_toklip_semantic_tokens)
    encoder = encoder.to(device).eval().requires_grad_(False)
    train_transform, eval_transform = _pm1_transforms(image_size)
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=nullcontext,
        representation="mean of final normalized TokLIP semantic tokens; forward_head is not used",
        transform_description=(
            f"train=RandomResizedCrop({image_size})+HorizontalFlip(0.5), "
            f"eval=Resize({image_size})+CenterCrop({image_size}), mean={PM1_MEAN}, std={PM1_STD}"
        ),
        backbone_precision="float32",
        checkpoint_paths=[checkpoint, vq_checkpoint],
    )


def _load_unitok(args, device: torch.device) -> FeatureBundle:
    if str(args.image_scripts) not in sys.path:
        sys.path.insert(0, str(args.image_scripts))
    from unitok_vae_rec import load_unitok

    checkpoint = str(Path(args.unitok_checkpoint).expanduser().resolve())
    model, _preprocess = load_unitok(
        SimpleNamespace(
            unitok_path=str(Path(args.unitok_path).expanduser().resolve()),
            ckpt_path=checkpoint,
        )
    )
    encoder = UniTokEncoder(model).to(device).eval().requires_grad_(False)
    train_transform, eval_transform = _pm1_transforms(256)
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=nullcontext,
        representation=(
            "quantize/dequantize -> post_quant_proj -> token mean -> fc_norm, before UniTok projection"
        ),
        transform_description=(
            f"train=RandomResizedCrop(256)+HorizontalFlip(0.5), "
            f"eval=Resize(256)+CenterCrop(256), mean={PM1_MEAN}, std={PM1_STD}"
        ),
        backbone_precision="float32",
        checkpoint_paths=[checkpoint],
    )


def _load_vilau(args, device: torch.device) -> FeatureBundle:
    if str(args.image_scripts) not in sys.path:
        sys.path.insert(0, str(args.image_scripts))
    from vilau_rec import load_vilau_tokenizer

    model_path = str(Path(args.vilau_model_path).expanduser().resolve())
    siglip_config_path = str(Path(args.vilau_siglip_config).expanduser().resolve())
    tokenizer = load_vilau_tokenizer(
        SimpleNamespace(
            vilau_path=str(Path(args.vilau_path).expanduser().resolve()),
            model_path=model_path,
            siglip_config_path=siglip_config_path,
            dtype="bfloat16",
        ),
        device=str(device),
    )
    encoder = VilaUEncoder(tokenizer.model, tokenizer.dtype)
    encoder = encoder.to(device).eval().requires_grad_(False)
    train_transform, eval_transform = _pm1_transforms(tokenizer.image_size)
    return FeatureBundle(
        encoder=encoder,
        train_transform=train_transform,
        eval_transform=eval_transform,
        autocast_context=partial(torch.autocast, device_type="cuda", dtype=torch.bfloat16),
        representation="mean of the penultimate VILA-U SigLIP encoder-block tokens",
        transform_description=(
            f"train=RandomResizedCrop({tokenizer.image_size})+HorizontalFlip(0.5), "
            f"eval=Resize({tokenizer.image_size})+CenterCrop({tokenizer.image_size}), "
            f"mean={PM1_MEAN}, std={PM1_STD}"
        ),
        backbone_precision="bfloat16",
        checkpoint_paths=[model_path, siglip_config_path],
    )


def load_feature_bundle(model_name: str, args, device: torch.device) -> FeatureBundle:
    if model_name == "metaclip":
        return _load_metaclip(
            args.metaclip_model,
            args.metaclip_checkpoint,
            "final normalized CLS before the MetaCLIP 768-to-512 projection",
            device,
        )
    if model_name == "clip_openai__l14":
        return _load_clip_openai_l14(args, device)
    if model_name == "clip_meta__l14":
        return _load_metaclip(
            args.clip_meta_model,
            args.clip_meta_checkpoint,
            "final normalized CLS before the MetaCLIP 1024-to-768 projection",
            device,
        )
    if model_name in MC1_SPECS:
        timm_model_name, checkpoint_arg, representation = MC1_SPECS[model_name]
        return _load_metaclip(
            timm_model_name,
            getattr(args, checkpoint_arg),
            representation,
            device,
        )
    if model_name in MC2_TIMM_SPECS:
        timm_model_name, checkpoint_arg, representation = MC2_TIMM_SPECS[model_name]
        return _load_metaclip(
            timm_model_name,
            getattr(args, checkpoint_arg),
            representation,
            device,
        )
    if model_name in MC2_DISTILLED_SPECS:
        checkpoint_arg, image_size, patch_size, width, depth, projection_dim = (
            MC2_DISTILLED_SPECS[model_name]
        )
        return _load_metaclip2_distilled(
            getattr(args, checkpoint_arg),
            image_size,
            patch_size,
            width,
            depth,
            projection_dim,
            device,
        )
    if model_name in SIGLIP2_B_SPECS:
        architecture, model_path_arg, image_size = SIGLIP2_B_SPECS[model_name]
        return _load_siglip2_map(
            getattr(args, model_path_arg),
            architecture,
            image_size,
            device,
        )
    if model_name in RAEV2_SPECS:
        return _load_raev2_variant(args, device, model_name)
    if model_name == "toklip_s":
        return _load_toklip(args, device, "s")
    if model_name == "toklip_l":
        return _load_toklip(args, device, "l")
    if model_name == "unitok":
        return _load_unitok(args, device)
    if model_name == "vilau":
        return _load_vilau(args, device)
    raise ValueError(f"Unsupported tokenizer model: {model_name}")
