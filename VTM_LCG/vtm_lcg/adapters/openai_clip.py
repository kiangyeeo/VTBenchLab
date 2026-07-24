from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from .base import SequenceTokenizerAdapter


class OpenAIClipL14Adapter(SequenceTokenizerAdapter):
    """Final-normalized OpenAI CLIP ViT-L/14 patch tokens."""

    adapter_name = "openai_clip"

    @property
    def checkpoint_file(self) -> Path:
        checkpoint = self.checkpoint_path
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"OpenAI CLIP checkpoint directory is missing: {checkpoint}")
        for filename in ("model.safetensors", "pytorch_model.bin"):
            candidate = checkpoint / filename
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No supported OpenAI CLIP weight file in {checkpoint}")

    def load(
        self,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "OpenAIClipL14Adapter":
        from transformers import CLIPVisionModel

        resolved_device = torch.device(device)
        model, loading_info = CLIPVisionModel.from_pretrained(
            str(self.checkpoint_path),
            local_files_only=True,
            output_loading_info=True,
        )
        unexpected_vision_keys = [
            key
            for key in loading_info["unexpected_keys"]
            if key.startswith("vision_model.")
        ]
        if (
            loading_info["missing_keys"]
            or loading_info["mismatched_keys"]
            or loading_info["error_msgs"]
            or unexpected_vision_keys
        ):
            raise RuntimeError(
                "Failed to strictly load the OpenAI CLIP vision tower: "
                f"{loading_info}"
            )
        model = model.to(device=resolved_device, dtype=dtype)
        model.eval().requires_grad_(False)
        self._model = model
        self._device = resolved_device
        self._dtype = dtype
        return self

    def _encode_values(self, images: Tensor) -> Tensor:
        assert self._model is not None
        outputs = self._model(pixel_values=images, return_dict=True)
        hidden = outputs.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[1] < 2:
            raise RuntimeError(f"Unexpected OpenAI CLIP hidden shape: {tuple(hidden.shape)}")
        patch_tokens = hidden[:, 1:, :]
        return self._model.vision_model.post_layernorm(patch_tokens)

