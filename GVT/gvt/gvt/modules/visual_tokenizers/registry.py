from functools import partial

import torch
import torch.nn as nn

try:
    from apex.normalization import FusedLayerNorm
except ImportError:
    FusedLayerNorm = nn.LayerNorm

from gvt.modules.visual_modules.eva import EVAVisionTransformer
from .base import VisualTokenizerWrapper


def _load_encoder_state_dict(model, state_dict, filter_name="decoder"):
    updated_keys = []
    missing_keys = []
    unexpected_keys = list(state_dict.keys())

    for name, _ in model.named_parameters():
        if name in state_dict.keys():
            updated_keys.append(name)
            unexpected_keys.remove(name)
        else:
            missing_keys.append(name)

    if missing_keys:
        print("missing keys:", [k for k in missing_keys if filter_name not in k])
    if unexpected_keys:
        print("unexpected keys:", [k for k in unexpected_keys if filter_name not in k])

    model.load_state_dict(state_dict, strict=False)


class GVTTokenizerWrapper(VisualTokenizerWrapper):
    def __init__(self, config):
        super().__init__()
        self.visual_encoder = EVAVisionTransformer(
            img_size=224,
            patch_size=14,
            depth=24,
            mlp_ratio=2.6667,
            num_heads=16,
            embed_dim=1024,
            drop_path_rate=0,
            xattn=True,
            qkv_bias=True,
            norm_layer=partial(FusedLayerNorm, eps=1e-6),
            rope=True,
            pt_hw_seq_len=16,
            intp_freq=True,
            naiveswiglu=True,
            subln=True,
        )
        self._output_dim = 1024

        visual_tokenizer_path = config.get("visual_tokenizer_path", "")
        if visual_tokenizer_path:
            params = torch.load(visual_tokenizer_path, map_location="cpu")["model"]
            new_state_dict = {}
            for key, value in params.items():
                if "visual" in key and "head" not in key:
                    new_key = key.replace("visual.", "")
                    new_state_dict[new_key] = value
            _load_encoder_state_dict(self.visual_encoder, new_state_dict)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual_encoder.forward_features(images, return_all_features=True)


class _StubTokenizerWrapper(VisualTokenizerWrapper):
    tokenizer_name = "stub"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._output_dim = int(config.get("visual_tokenizer_dim", 1024))

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"visual_tokenizer_type='{self.tokenizer_name}' is a placeholder. "
            "Implement this wrapper so it returns continuous tokens [B, N, D]."
        )


class CLIPTokenizerWrapper(_StubTokenizerWrapper):
    tokenizer_name = "clip"


class DINOv2TokenizerWrapper(_StubTokenizerWrapper):
    tokenizer_name = "dinov2"


class CustomTokenizerWrapper(_StubTokenizerWrapper):
    tokenizer_name = "custom"


TOKENIZER_REGISTRY = {
    "gvt": GVTTokenizerWrapper,
    "clip": CLIPTokenizerWrapper,
    "dinov2": DINOv2TokenizerWrapper,
    "custom": CustomTokenizerWrapper,
}


def build_visual_tokenizer(config):
    tokenizer_type = config.get("visual_tokenizer_type", "gvt").lower()
    if tokenizer_type not in TOKENIZER_REGISTRY:
        choices = ", ".join(sorted(TOKENIZER_REGISTRY))
        raise ValueError(f"Unknown visual_tokenizer_type='{tokenizer_type}'. Available: {choices}")
    return TOKENIZER_REGISTRY[tokenizer_type](config)
