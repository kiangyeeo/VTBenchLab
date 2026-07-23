"""Frozen feature extractors for the tokenizer linear-probing baseline.

Each extractor returns exactly one preselected visual representation shaped
``[batch, feature_dim]``.  Pooling performed here is part of that representation;
the evaluator does not perform DINOv2's multi-block or CLS/patch readout search.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable

import torch
from torch import nn

from dinov2.data.transforms import make_classification_eval_transform, make_classification_train_transform


PM1_MEAN = (0.5, 0.5, 0.5)
PM1_STD = (0.5, 0.5, 0.5)

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
    if model_name == "toklip_s":
        return _load_toklip(args, device, "s")
    if model_name == "toklip_l":
        return _load_toklip(args, device, "l")
    if model_name == "unitok":
        return _load_unitok(args, device)
    if model_name == "vilau":
        return _load_vilau(args, device)
    raise ValueError(f"Unsupported tokenizer model: {model_name}")
