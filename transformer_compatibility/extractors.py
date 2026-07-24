"""Frozen sequence extractors used by the compatibility probes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


SUPPORTED_CHECKPOINT_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "open_clip_pytorch_model.bin",
    "checkpoint.pth",
)


def resolve_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if checkpoint.is_file():
        return checkpoint
    if checkpoint.is_dir():
        for filename in SUPPORTED_CHECKPOINT_FILENAMES:
            candidate = checkpoint / filename
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"No supported checkpoint found at {checkpoint}")


@dataclass(frozen=True)
class TokenSurface:
    name: str
    input_dim: int
    token_count: int
    grid_shape: tuple[int, int]
    checkpoint: str
    timm_model: str


class FrozenMetaClipPatchTokenizer(nn.Module):
    """Return the final normalized MetaCLIP patch tokens before projection."""

    def __init__(
        self,
        *,
        timm_model: str,
        checkpoint: str | Path,
        input_dim: int,
        token_count: int,
        grid_shape: tuple[int, int],
        surface_name: str,
    ) -> None:
        super().__init__()
        import timm
        from timm.models import load_checkpoint

        checkpoint_path = resolve_checkpoint(checkpoint)
        model = timm.create_model(timm_model, pretrained=False)
        load_checkpoint(model, str(checkpoint_path), strict=True)
        model.eval().requires_grad_(False)

        self.model = model
        self.surface = TokenSurface(
            name=surface_name,
            input_dim=int(input_dim),
            token_count=int(token_count),
            grid_shape=tuple(int(value) for value in grid_shape),
            checkpoint=str(checkpoint_path),
            timm_model=timm_model,
        )

    def train(self, mode: bool = True) -> "FrozenMetaClipPatchTokenizer":
        # The tokenizer remains frozen and deterministic when the readout trains.
        super().train(False)
        self.model.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model.get_intermediate_layers(
            images,
            n=1,
            return_prefix_tokens=True,
            norm=True,
        )
        patch_tokens, prefix_tokens = outputs[-1]
        expected = (images.shape[0], self.surface.token_count, self.surface.input_dim)
        if tuple(patch_tokens.shape) != expected:
            raise RuntimeError(
                f"Expected patch-token shape {expected}, got {tuple(patch_tokens.shape)}"
            )
        if prefix_tokens.ndim != 3 or prefix_tokens.shape[1] < 1:
            raise RuntimeError(
                f"Expected at least one prefix token, got {tuple(prefix_tokens.shape)}"
            )
        return patch_tokens


def build_tokenizer(
    model_config: dict[str, Any],
    surface_config: dict[str, Any],
    workspace: Path,
) -> FrozenMetaClipPatchTokenizer:
    checkpoint = Path(model_config["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = workspace / checkpoint
    return FrozenMetaClipPatchTokenizer(
        timm_model=model_config["timm_model"],
        checkpoint=checkpoint,
        input_dim=int(model_config["input_dim"]),
        token_count=int(model_config["token_count"]),
        grid_shape=tuple(model_config["grid_shape"]),
        surface_name=surface_config["name"],
    )
