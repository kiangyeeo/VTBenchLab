from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from vtm_lcg.schemas import TokenBatch, row_major_coords


class SequenceTokenizerAdapter(ABC):
    """Common interface for frozen dense visual-token sequence extractors."""

    adapter_name = "base"

    def __init__(
        self,
        tokenizer_config: Mapping[str, Any],
        preprocess_config: Mapping[str, Any],
    ) -> None:
        self.tokenizer_config = dict(tokenizer_config)
        tokenizer_preprocess = self.tokenizer_config.get("preprocess", {})
        if not isinstance(tokenizer_preprocess, Mapping):
            raise ValueError("tokenizer preprocess override must be a mapping")
        self.preprocess_config = {
            **dict(preprocess_config),
            **dict(tokenizer_preprocess),
        }
        self.tokenizer_id = str(self.tokenizer_config["id"])
        self.surface = str(self.tokenizer_config["surface"])
        expected_grid = self.tokenizer_config["expected_grid"]
        self.grid_shape = (int(expected_grid[0]), int(expected_grid[1]))
        self.expected_hidden_dim = int(self.tokenizer_config["expected_hidden_dim"])
        input_size = int(self.preprocess_config["input_size"])
        self.input_size = (input_size, input_size)
        self._model: nn.Module | None = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None
        self._preprocess = self._build_preprocess()

    def _build_preprocess(self):
        interpolation_name = str(
            self.preprocess_config.get("interpolation", "bicubic")
        ).upper()
        try:
            interpolation = InterpolationMode[interpolation_name]
        except KeyError as error:
            raise ValueError(f"Unsupported interpolation: {interpolation_name}") from error
        resize_size = int(self.preprocess_config["resize_size"])
        input_size = int(self.preprocess_config["input_size"])
        mean = tuple(float(value) for value in self.preprocess_config["mean"])
        std = tuple(float(value) for value in self.preprocess_config["std"])
        return transforms.Compose(
            [
                transforms.Resize(
                    resize_size,
                    interpolation=interpolation,
                    antialias=True,
                ),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    @property
    def preprocess(self):
        return self._preprocess

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.tokenizer_config["checkpoint"]).resolve()

    @property
    @abstractmethod
    def checkpoint_file(self) -> Path:
        """The exact model-weight file used for cache identity."""

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "tokenizer_id": self.tokenizer_id,
                "adapter": self.adapter_name,
                "checkpoint": str(self.checkpoint_path),
                "checkpoint_file": str(self.checkpoint_file),
                "input_resolution": self.input_size[0],
                "patch_size": self.input_size[0] // self.grid_shape[0],
                "token_count": self.grid_shape[0] * self.grid_shape[1],
                "hidden_dim": self.expected_hidden_dim,
                "representation_surface": self.surface,
                "grid_shape": list(self.grid_shape),
                "preprocess": dict(self.preprocess_config),
            }
        )

    @abstractmethod
    def load(
        self,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "SequenceTokenizerAdapter":
        """Load and freeze the local vision tower."""

    @abstractmethod
    def _encode_values(self, images: Tensor) -> Tensor:
        """Return patch-token values with shape [B,N,D], excluding special tokens."""

    def encode(self, images: Tensor) -> TokenBatch:
        if self._model is None or self._device is None or self._dtype is None:
            raise RuntimeError(f"Adapter {self.tokenizer_id} must be loaded before encode()")
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must have shape [B,3,H,W], got {tuple(images.shape)}")
        if tuple(images.shape[-2:]) != self.input_size:
            raise ValueError(
                f"Expected image size {self.input_size}, got {tuple(images.shape[-2:])}"
            )

        model_inputs = images.to(
            device=self._device,
            dtype=self._dtype,
            non_blocking=True,
        )
        values = self._encode_values(model_inputs)
        expected_shape = (
            images.shape[0],
            self.grid_shape[0] * self.grid_shape[1],
            self.expected_hidden_dim,
        )
        if tuple(values.shape) != expected_shape:
            raise RuntimeError(
                f"{self.tokenizer_id} returned {tuple(values.shape)}; expected {expected_shape}"
            )
        if not bool(torch.isfinite(values).all().item()):
            raise RuntimeError(f"{self.tokenizer_id} produced non-finite visual tokens")

        batch_size, token_count, _ = values.shape
        mask = torch.ones(
            (batch_size, token_count),
            dtype=torch.bool,
            device=values.device,
        )
        special_mask = torch.zeros_like(mask)
        coords = row_major_coords(self.grid_shape, device=values.device)
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1)
        return TokenBatch(
            values=values,
            mask=mask,
            coords=coords,
            special_mask=special_mask,
            grid_shape=self.grid_shape,
            surface=self.surface,
            input_size=self.input_size,
            metadata=self.metadata,
        )
