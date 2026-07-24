from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from .base import SequenceTokenizerAdapter


class Siglip2Adapter(SequenceTokenizerAdapter):
    """Final-normalized SigLIP2 patch tokens before attention pooling."""

    adapter_name = "siglip2"

    @property
    def checkpoint_file(self) -> Path:
        checkpoint = self.checkpoint_path
        if checkpoint.is_file():
            return checkpoint
        if checkpoint.is_dir():
            for filename in ("model.safetensors", "pytorch_model.bin"):
                candidate = checkpoint / filename
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(f"No supported SigLIP2 checkpoint found at {checkpoint}")

    def load(
        self,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "Siglip2Adapter":
        import timm
        from timm.models import load_checkpoint

        model_name = str(self.tokenizer_config["model_name"])
        resolved_device = torch.device(device)
        model = timm.create_model(model_name, pretrained=False)
        load_checkpoint(model, str(self.checkpoint_file), strict=True)
        model = model.to(device=resolved_device, dtype=dtype)
        model.eval().requires_grad_(False)
        self._model = model
        self._device = resolved_device
        self._dtype = dtype
        return self

    def _encode_values(self, images: Tensor) -> Tensor:
        assert self._model is not None
        outputs = self._model.get_intermediate_layers(
            images,
            n=1,
            return_prefix_tokens=False,
            norm=True,
        )
        if not outputs:
            raise RuntimeError("SigLIP2 returned no intermediate layers")
        patch_tokens = outputs[-1]
        if isinstance(patch_tokens, tuple):
            patch_tokens = patch_tokens[0]
        if patch_tokens.ndim != 3:
            raise RuntimeError(
                f"Unexpected SigLIP2 patch-token shape: {tuple(patch_tokens.shape)}"
            )
        return patch_tokens
