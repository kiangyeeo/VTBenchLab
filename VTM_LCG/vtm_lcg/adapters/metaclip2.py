from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .base import SequenceTokenizerAdapter


class MetaClip2Adapter(SequenceTokenizerAdapter):
    """Final-normalized MetaCLIP 2 patch tokens before visual projection."""

    adapter_name = "metaclip2"

    @property
    def checkpoint_file(self) -> Path:
        checkpoint = self.checkpoint_path
        if checkpoint.is_file():
            return checkpoint
        if checkpoint.is_dir():
            candidates = sorted(checkpoint.glob("*.pt"))
            if len(candidates) == 1:
                return candidates[0]
        raise FileNotFoundError(
            f"No unambiguous MetaCLIP 2 checkpoint found at {checkpoint}"
        )

    def _visual_state_and_spec(self) -> tuple[dict[str, Tensor], dict[str, int]]:
        payload: Any = torch.load(
            self.checkpoint_file,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(payload, dict) or not isinstance(
            payload.get("state_dict"), dict
        ):
            raise RuntimeError(
                f"Expected a MetaCLIP 2 state_dict checkpoint at {self.checkpoint_file}"
            )
        state_dict = payload["state_dict"]
        visual_state = {
            key.removeprefix("visual."): value
            for key, value in state_dict.items()
            if key.startswith("visual.")
        }
        required = {"conv1.weight", "positional_embedding", "proj"}
        missing = sorted(required - set(visual_state))
        if missing:
            raise RuntimeError(f"Missing MetaCLIP 2 visual keys: {missing}")

        width = int(visual_state["conv1.weight"].shape[0])
        patch_size = int(visual_state["conv1.weight"].shape[-1])
        patch_tokens = int(visual_state["positional_embedding"].shape[0]) - 1
        grid_size = math.isqrt(patch_tokens)
        if grid_size * grid_size != patch_tokens:
            raise RuntimeError(
                f"Non-square MetaCLIP 2 positional grid: {patch_tokens} tokens"
            )
        depth = len(
            {
                key.split(".")[2]
                for key in visual_state
                if key.startswith("transformer.resblocks.")
                and key.endswith(".attn.in_proj_weight")
            }
        )
        projection_dim = int(visual_state["proj"].shape[1])
        return visual_state, {
            "width": width,
            "patch_size": patch_size,
            "grid_size": grid_size,
            "image_size": grid_size * patch_size,
            "depth": depth,
            "projection_dim": projection_dim,
        }

    def load(
        self,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "MetaClip2Adapter":
        clip_root = Path(__file__).resolve().parents[3] / "CLIP"
        if str(clip_root) not in sys.path:
            sys.path.insert(0, str(clip_root))
        from clip.model import VisionTransformer

        visual_state, actual = self._visual_state_and_spec()
        expected = {
            "image_size": self.input_size[0],
            "patch_size": self.input_size[0] // self.grid_shape[0],
            "grid_size": self.grid_shape[0],
            "width": self.expected_hidden_dim,
            "depth": int(self.tokenizer_config["depth"]),
            "projection_dim": int(self.tokenizer_config["projection_dim"]),
        }
        if actual != expected:
            raise RuntimeError(
                f"MetaCLIP 2 visual architecture mismatch: "
                f"expected={expected}, actual={actual}"
            )

        visual = VisionTransformer(
            input_resolution=actual["image_size"],
            patch_size=actual["patch_size"],
            width=actual["width"],
            layers=actual["depth"],
            heads=actual["width"] // 64,
            output_dim=actual["projection_dim"],
        )
        for block in visual.transformer.resblocks:
            block.mlp.gelu = nn.GELU()
        visual.load_state_dict(visual_state, strict=True)
        resolved_device = torch.device(device)
        visual = visual.to(device=resolved_device, dtype=dtype)
        # OpenAI CLIP's custom LayerNorm always evaluates its input in FP32.
        # Keep the affine parameters in FP32 as well; downcasting the whole
        # module makes CUDA mixed-precision calls fail with Float/BFloat16
        # parameter mismatches.
        for module in visual.modules():
            if isinstance(module, nn.LayerNorm):
                module.float()
        visual.eval().requires_grad_(False)
        self._model = visual
        self._device = resolved_device
        self._dtype = dtype
        return self

    def _encode_values(self, images: Tensor) -> Tensor:
        assert self._model is not None
        visual = self._model
        hidden = visual.conv1(images)
        hidden = hidden.reshape(hidden.shape[0], hidden.shape[1], -1)
        hidden = hidden.permute(0, 2, 1)
        class_tokens = visual.class_embedding.to(hidden.dtype).expand(
            hidden.shape[0], 1, -1
        )
        hidden = torch.cat((class_tokens, hidden), dim=1)
        hidden = visual.ln_pre(hidden + visual.positional_embedding.to(hidden.dtype))
        hidden = visual.transformer(hidden.permute(1, 0, 2)).permute(1, 0, 2)
        return visual.ln_post(hidden[:, 1:, :])
