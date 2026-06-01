from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class VisualTokenizerWrapper(nn.Module, ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor [B, 3, H, W]
        Returns:
            tokens: Tensor [B, N, D], continuous visual token embeddings
        """
        raise NotImplementedError
