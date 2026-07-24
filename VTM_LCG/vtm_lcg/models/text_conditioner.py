from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class FrozenClipTextConditioner:
    """Shared frozen OpenAI CLIP text tower used by every visual tokenizer."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        from transformers import CLIPTextModel, CLIPTokenizerFast

        self.checkpoint = checkpoint.resolve()
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        model, loading_info = CLIPTextModel.from_pretrained(
            str(self.checkpoint),
            local_files_only=True,
            output_loading_info=True,
        )
        unexpected_text_keys = [
            key
            for key in loading_info["unexpected_keys"]
            if key.startswith("text_model.")
        ]
        if (
            loading_info["missing_keys"]
            or loading_info["mismatched_keys"]
            or loading_info["error_msgs"]
            or unexpected_text_keys
        ):
            raise RuntimeError(
                f"Failed to strictly load the shared CLIP text tower: {loading_info}"
            )
        model = model.to(device=device, dtype=dtype)
        model.eval().requires_grad_(False)
        self.model: nn.Module = model
        self.tokenizer = CLIPTokenizerFast.from_pretrained(
            str(self.checkpoint),
            local_files_only=True,
        )

    @torch.no_grad()
    def encode(self, captions: list[str]) -> tuple[Tensor, Tensor]:
        tokenized = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        attention_mask = tokenized["attention_mask"].to(
            device=self.device,
            dtype=torch.bool,
        )
        model_inputs = {
            key: value.to(self.device)
            for key, value in tokenized.items()
        }
        outputs = self.model(**model_inputs, return_dict=True)
        embeddings = outputs.last_hidden_state
        if not bool(torch.isfinite(embeddings).all().item()):
            raise RuntimeError("Shared CLIP text tower produced non-finite embeddings")
        return embeddings, attention_mask
