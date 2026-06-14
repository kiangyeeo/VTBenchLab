from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms

from .common import REPO_ROOT, load_model_config, workspace_path


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_image_transform(crop_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class FrozenEncoder(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        *,
        alias: str,
        metadata: Dict[str, Any],
        image_transform,
        encode_kind: str,
        video_aggregation: str = "mean",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.module = module.eval()
        self.alias = alias
        self.metadata = metadata
        self.image_transform = image_transform
        self.encode_kind = encode_kind
        self.video_aggregation = video_aggregation
        self.normalize = normalize
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if self.encode_kind == "openclip":
            feats = self.module.encode_image(images)
        elif self.encode_kind == "dinov2_linear":
            outputs = self.module.get_intermediate_layers(images, n=4, return_class_token=True)
            feats = torch.cat(
                [outputs[0][1], outputs[1][1], outputs[2][1], outputs[3][1], outputs[3][0].mean(dim=1)],
                dim=1,
            )
        elif self.encode_kind == "forward_features":
            out = self.module.forward_features(images)
            if isinstance(out, dict):
                cls = out.get("x_norm_clstoken")
                patch = out.get("x_norm_patchtokens")
                feats = torch.cat([cls, patch.mean(dim=1)], dim=1) if patch is not None else cls
            elif isinstance(out, torch.Tensor) and out.ndim == 3:
                feats = out[:, 0]
            else:
                feats = out
        elif self.encode_kind == "transformers":
            out = self.module(pixel_values=images)
            tokens = getattr(out, "last_hidden_state", out[0])
            feats = tokens[:, 0]
        else:
            out = self.module(images)
            if isinstance(out, tuple):
                out = out[0]
            feats = out[:, 0] if isinstance(out, torch.Tensor) and out.ndim == 3 else out
        feats = feats.float()
        return F.normalize(feats, dim=-1) if self.normalize else feats

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim == 5:
            bsz, frames = batch.shape[:2]
            feats = self.encode_images(batch.flatten(0, 1))
            feats = feats.view(bsz, frames, -1)
            if self.video_aggregation == "concat":
                feats = feats.reshape(bsz, frames * feats.shape[-1])
                return F.normalize(feats, dim=-1) if self.normalize else feats
            return feats.mean(dim=1)
        return self.encode_images(batch)


def _load_dinov2(spec: Dict[str, Any], allow_download: bool) -> Tuple[torch.nn.Module, str]:
    sys.path.insert(0, str(REPO_ROOT))
    from dinov2.hub import backbones

    fn = getattr(backbones, spec["hub_name"])
    checkpoint = workspace_path(spec["checkpoint"])
    if checkpoint.exists():
        model = fn(weights=str(checkpoint))
        return model, str(checkpoint)
    if allow_download:
        model = fn(pretrained=True)
        return model, spec.get("official_url", "dinov2_default_url")
    raise FileNotFoundError(f"Missing DINOv2 checkpoint: {checkpoint}")


def _load_ibot(spec: Dict[str, Any]) -> Tuple[torch.nn.Module, str]:
    from SAIL.model.ibot import vit_large

    checkpoint = workspace_path(spec["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing iBOT checkpoint: {checkpoint}")
    model = vit_large()
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("state_dict", state)
    state = state.get("teacher", state)
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model, str(checkpoint)


def _load_torchhub(spec: Dict[str, Any], model_root: Path, allow_download: bool) -> Tuple[torch.nn.Module, str]:
    if not allow_download:
        raise FileNotFoundError(
            f"{spec['hub_name']} is a torchhub model. Run the download script once or pass --allow-download."
        )
    torch.hub.set_dir(str(model_root / "torchhub"))
    model = torch.hub.load(spec["repo"], spec["hub_name"])
    return model, f"torchhub://{spec['repo']}/{spec['hub_name']}"


def _load_transformers(spec: Dict[str, Any], allow_download: bool) -> Tuple[torch.nn.Module, str]:
    from transformers import AutoModel

    local_dir = workspace_path(spec["checkpoint"])
    source = str(local_dir) if local_dir.exists() else spec["hf_id"]
    if not local_dir.exists() and not allow_download:
        raise FileNotFoundError(f"Missing Hugging Face model directory: {local_dir}")
    model = AutoModel.from_pretrained(source, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    return model, source


def _load_openclip(spec: Dict[str, Any], allow_download: bool):
    import open_clip

    model_name = spec["openclip_model"]
    checkpoint = workspace_path(spec["checkpoint"]) if spec.get("checkpoint") else None
    if checkpoint and checkpoint.exists():
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=str(checkpoint))
        return model, preprocess, str(checkpoint)
    if spec.get("checkpoint_required"):
        raise FileNotFoundError(
            f"Missing required OpenCLIP paper-baseline checkpoint: {checkpoint}. "
            "This alias intentionally does not fall back to ViT-g-14/laion2b_s12b_b42k."
        )
    if not allow_download:
        raise FileNotFoundError(f"Missing OpenCLIP weights for {model_name}; pass --allow-download to use open_clip.")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=spec.get("pretrained"))
    return model, preprocess, f"open_clip://{model_name}/{spec.get('pretrained')}"


def build_encoder(
    alias: str,
    *,
    config_path: Optional[str] = None,
    model_root: str | os.PathLike = "checkpoints/dinov2_baseline",
    device: str = "cuda",
    allow_download: bool = False,
    video_aggregation: str = "mean",
    normalize: bool = True,
) -> FrozenEncoder:
    cfg = load_model_config(config_path)
    specs = cfg["models"]
    if alias not in specs:
        raise KeyError(f"Unknown model alias '{alias}'. Available: {', '.join(sorted(specs))}")
    spec = specs[alias]
    model_root = workspace_path(model_root)
    kind = spec["kind"]
    preprocess = default_image_transform()
    source = ""

    if kind == "dinov2":
        module, source = _load_dinov2(spec, allow_download)
        encode_kind = "dinov2_linear"
    elif kind == "ibot":
        module, source = _load_ibot(spec)
        encode_kind = "forward_features"
    elif kind == "torchhub":
        module, source = _load_torchhub(spec, model_root, allow_download)
        encode_kind = "forward_features"
    elif kind == "transformers":
        module, source = _load_transformers(spec, allow_download)
        encode_kind = "transformers"
    elif kind == "openclip":
        module, preprocess, source = _load_openclip(spec, allow_download)
        encode_kind = "openclip"
    else:
        raise ValueError(f"Unsupported model kind: {kind}")

    module = module.to(device).eval()
    if device.startswith("cuda") and kind != "transformers":
        module = module.half()
    metadata = dict(spec)
    metadata["checkpoint_source"] = source
    return FrozenEncoder(
        module,
        alias=alias,
        metadata=metadata,
        image_transform=preprocess,
        encode_kind=encode_kind,
        video_aggregation=video_aggregation,
        normalize=normalize,
    )
