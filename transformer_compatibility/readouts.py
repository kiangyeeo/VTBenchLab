"""Controlled ImageNet readouts for frozen visual-token sequences."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class GapLinearReadout(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.classifier(tokens.mean(dim=1))


class GapMLPReadout(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.mlp(tokens.mean(dim=1))


class LowRankProjector(nn.Module):
    def __init__(self, input_dim: int, rank: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, rank, bias=False),
            nn.GELU(),
            nn.Linear(rank, output_dim, bias=False),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class TokenTransformerReadout(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        projector_rank: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        ffn_dim = int(round(hidden_dim * ffn_ratio))
        self.projector = LowRankProjector(input_dim, projector_rank, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.input_norm(self.projector(tokens))
        cls = self.cls_token.expand(projected.shape[0], -1, -1)
        encoded = self.transformer(torch.cat((cls, projected), dim=1))
        return self.classifier(self.output_norm(encoded[:, 0]))


def _matched_mlp_hidden_dim(
    *, input_dim: int, num_classes: int, target_parameters: int
) -> int:
    # Two biased Linear layers:
    # input_dim * hidden + hidden + hidden * classes + classes.
    denominator = input_dim + num_classes + 1
    hidden = max(1, round((target_parameters - num_classes) / denominator))
    candidates = range(max(1, hidden - 4), hidden + 5)

    def parameter_count(width: int) -> int:
        return input_dim * width + width + width * num_classes + num_classes

    return min(candidates, key=lambda width: abs(parameter_count(width) - target_parameters))


def _build_transformer(
    *, input_dim: int, num_classes: int, config: dict[str, Any]
) -> TokenTransformerReadout:
    return TokenTransformerReadout(
        input_dim=input_dim,
        num_classes=num_classes,
        projector_rank=int(config["projector_rank"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        ffn_ratio=float(config["ffn_ratio"]),
        dropout=float(config["dropout"]),
    )


def build_readout(
    *,
    name: str,
    input_dim: int,
    num_classes: int,
    readout_configs: dict[str, Any],
) -> tuple[nn.Module, dict[str, Any]]:
    if name == "gap_linear":
        module = GapLinearReadout(input_dim, num_classes)
        metadata: dict[str, Any] = {}
    elif name == "transformer":
        module = _build_transformer(
            input_dim=input_dim,
            num_classes=num_classes,
            config=readout_configs["transformer"],
        )
        metadata = dict(readout_configs["transformer"])
    elif name == "gap_mlp":
        reference = _build_transformer(
            input_dim=input_dim,
            num_classes=num_classes,
            config=readout_configs["transformer"],
        )
        target_parameters = trainable_parameter_count(reference)
        hidden_dim = _matched_mlp_hidden_dim(
            input_dim=input_dim,
            num_classes=num_classes,
            target_parameters=target_parameters,
        )
        del reference
        module = GapMLPReadout(input_dim, hidden_dim, num_classes)
        metadata = {
            "hidden_dim": hidden_dim,
            "target_transformer_parameters": target_parameters,
        }
    else:
        choices = ", ".join(sorted(("gap_linear", "gap_mlp", "transformer")))
        raise ValueError(f"Unknown readout '{name}'. Available: {choices}")

    metadata["trainable_parameters"] = trainable_parameter_count(module)
    return module, metadata
