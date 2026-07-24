from __future__ import annotations

import torch
from torch import Tensor, nn

from vtm_lcg.models.masked_predictor import build_2d_sincos_position_embedding


class CrossViewResidualPredictor(nn.Module):
    """Predict aligned target-view residuals from visible source-view residuals."""

    def __init__(
        self,
        *,
        visual_input_dim: int,
        model_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        grid_shape: tuple[int, int],
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.visual_input_dim = visual_input_dim
        self.model_dim = model_dim
        self.grid_shape = grid_shape
        self.token_count = grid_shape[0] * grid_shape[1]
        self.input_projection = nn.Linear(visual_input_dim, model_dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, model_dim))
        position_embedding = build_2d_sincos_position_embedding(
            grid_shape,
            model_dim,
        )
        self.register_buffer(
            "position_embedding",
            position_embedding.unsqueeze(0),
            persistent=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Linear(model_dim, visual_input_dim)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        source_residuals: Tensor,
        masked_positions: Tensor,
    ) -> Tensor:
        if source_residuals.ndim != 3:
            raise ValueError("source_residuals must have shape [B,N,D]")
        batch_size, token_count, hidden_dim = source_residuals.shape
        expected = (self.token_count, self.visual_input_dim)
        if (token_count, hidden_dim) != expected:
            raise ValueError(
                f"Expected source shape [B,{expected[0]},{expected[1]}], "
                f"got {tuple(source_residuals.shape)}"
            )
        if tuple(masked_positions.shape) != (batch_size, token_count):
            raise ValueError("masked_positions must have shape [B,N]")
        if masked_positions.dtype is not torch.bool:
            raise ValueError("masked_positions must be bool")

        hidden = self.input_projection(source_residuals)
        mask_tokens = self.mask_token.expand(batch_size, token_count, -1)
        hidden = torch.where(
            masked_positions.unsqueeze(-1),
            mask_tokens,
            hidden,
        )
        hidden = hidden + self.position_embedding.to(hidden.dtype)
        return self.output_projection(self.encoder(hidden))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
