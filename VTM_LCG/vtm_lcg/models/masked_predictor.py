from __future__ import annotations

import torch
from torch import Tensor, nn


def _sincos_1d(positions: Tensor, embedding_dim: int) -> Tensor:
    if embedding_dim % 2 != 0:
        raise ValueError(f"1D sin/cos embedding dimension must be even: {embedding_dim}")
    half = embedding_dim // 2
    frequencies = torch.arange(half, dtype=torch.float64, device=positions.device)
    frequencies = frequencies / max(half, 1)
    frequencies = 1.0 / (10000.0**frequencies)
    angles = positions.to(torch.float64).unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat((angles.sin(), angles.cos()), dim=1).to(torch.float32)


def build_2d_sincos_position_embedding(
    grid_shape: tuple[int, int],
    embedding_dim: int,
) -> Tensor:
    if embedding_dim % 4 != 0:
        raise ValueError(
            f"2D sin/cos embedding dimension must be divisible by four: {embedding_dim}"
        )
    rows, columns = grid_shape
    row_ids = torch.arange(rows, dtype=torch.float32)
    column_ids = torch.arange(columns, dtype=torch.float32)
    row_grid, column_grid = torch.meshgrid(row_ids, column_ids, indexing="ij")
    half = embedding_dim // 2
    row_embedding = _sincos_1d(row_grid.reshape(-1), half)
    column_embedding = _sincos_1d(column_grid.reshape(-1), half)
    return torch.cat((row_embedding, column_embedding), dim=1)


class MaskedVisualPredictor(nn.Module):
    """Tiny Transformer that predicts masked normalized visual tokens."""

    def __init__(
        self,
        *,
        visual_input_dim: int,
        text_input_dim: int,
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
        self.text_input_dim = text_input_dim
        self.model_dim = model_dim
        self.grid_shape = grid_shape
        self.token_count = grid_shape[0] * grid_shape[1]

        self.visual_projection = nn.Linear(visual_input_dim, model_dim)
        self.text_projection = nn.Linear(text_input_dim, model_dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, model_dim))
        self.modality_embedding = nn.Parameter(torch.empty(2, 1, model_dim))
        position_embedding = build_2d_sincos_position_embedding(
            grid_shape,
            model_dim,
        )
        self.register_buffer(
            "visual_position_embedding",
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
        nn.init.normal_(self.modality_embedding, std=0.02)

    def forward(
        self,
        visual_values: Tensor,
        masked_positions: Tensor,
        *,
        text_embeddings: Tensor | None = None,
        text_attention_mask: Tensor | None = None,
        hide_all_visual: bool = False,
    ) -> Tensor:
        if visual_values.ndim != 3:
            raise ValueError("visual_values must have shape [B,N,D]")
        batch_size, token_count, hidden_dim = visual_values.shape
        if token_count != self.token_count or hidden_dim != self.visual_input_dim:
            raise ValueError(
                f"Expected visual shape [B,{self.token_count},{self.visual_input_dim}], "
                f"got {tuple(visual_values.shape)}"
            )
        if tuple(masked_positions.shape) != (batch_size, token_count):
            raise ValueError("masked_positions must have shape [B,N]")
        if masked_positions.dtype is not torch.bool:
            raise ValueError("masked_positions must be bool")

        visual_hidden = self.visual_projection(visual_values)
        replacement_mask = (
            torch.ones_like(masked_positions) if hide_all_visual else masked_positions
        )
        mask_tokens = self.mask_token.expand(batch_size, token_count, -1)
        visual_hidden = torch.where(
            replacement_mask.unsqueeze(-1),
            mask_tokens,
            visual_hidden,
        )
        visual_hidden = (
            visual_hidden
            + self.visual_position_embedding.to(visual_hidden.dtype)
            + self.modality_embedding[0]
        )
        sequence_parts = [visual_hidden]
        padding_parts = [
            torch.zeros(
                batch_size,
                token_count,
                dtype=torch.bool,
                device=visual_values.device,
            )
        ]

        if text_embeddings is not None:
            if text_embeddings.ndim != 3 or text_embeddings.shape[0] != batch_size:
                raise ValueError("text_embeddings must have shape [B,T,D_text]")
            if text_embeddings.shape[2] != self.text_input_dim:
                raise ValueError(
                    f"Expected text dim {self.text_input_dim}, "
                    f"got {text_embeddings.shape[2]}"
                )
            if text_attention_mask is None:
                raise ValueError("text_attention_mask is required with text_embeddings")
            if tuple(text_attention_mask.shape) != tuple(text_embeddings.shape[:2]):
                raise ValueError("text_attention_mask must have shape [B,T]")
            if text_attention_mask.dtype is not torch.bool:
                raise ValueError("text_attention_mask must be bool")
            text_hidden = (
                self.text_projection(text_embeddings) + self.modality_embedding[1]
            )
            sequence_parts.append(text_hidden)
            padding_parts.append(~text_attention_mask)
        elif text_attention_mask is not None:
            raise ValueError("text_attention_mask was provided without text_embeddings")

        sequence = torch.cat(sequence_parts, dim=1)
        key_padding_mask = torch.cat(padding_parts, dim=1)
        encoded = self.encoder(
            sequence,
            src_key_padding_mask=key_padding_mask,
        )
        return self.output_projection(encoded[:, :token_count])

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

