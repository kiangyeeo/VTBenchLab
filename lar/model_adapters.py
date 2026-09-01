"""Patch-mean adapters for the tokenizer loaders used by the probing code."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


WORKSPACE = Path(__file__).resolve().parents[1]
PROBE_SCRIPTS = WORKSPACE / "scripts" / "linear_probe_tokenizers"
DINO_ROOT = WORKSPACE / "dinov2"
if str(PROBE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PROBE_SCRIPTS))
if str(DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(DINO_ROOT))

import feature_extractors as fe  # noqa: E402


class PatchMeanEncoder(nn.Module):
    """Return only spatial/patch-token means and remember the token count."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.last_n_tokens: int | None = None

    def _finish(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] < 1:
            raise RuntimeError(f"Expected patch tokens [B,T,D], got {tuple(tokens.shape)}")
        self.last_n_tokens = int(tokens.shape[1])
        return tokens.mean(dim=1).float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder

        if isinstance(encoder, fe.HFClsPatchEncoder):
            tokens = encoder.model(pixel_values=images, return_dict=True).last_hidden_state
            return self._finish(tokens[:, encoder.prefix_token_count :])

        if isinstance(encoder, fe.HFClsEncoder):
            tokens = encoder.model(pixel_values=images, return_dict=True).last_hidden_state
            return self._finish(tokens[:, 1:])

        if isinstance(encoder, fe.MetaCLIPEncoder):
            patches, _prefix = encoder.model.get_intermediate_layers(
                images, n=1, return_prefix_tokens=True, norm=True
            )[-1]
            return self._finish(patches)

        if isinstance(encoder, fe.SigLIP2MAPEncoder):
            output = encoder.model.forward_features(images)
            tokens = output
            if isinstance(output, dict):
                tokens = output.get("x_norm_patchtokens")
                if tokens is None:
                    tokens = output.get("x")
            if not torch.is_tensor(tokens):
                raise RuntimeError("SigLIP2 forward_features did not return token features")
            prefix_count = int(getattr(encoder.model, "num_prefix_tokens", 0) or 0)
            return self._finish(tokens[:, prefix_count:])

        if isinstance(encoder, fe.PerceptionEncoderReadout):
            tokens = encoder.model.forward_features(images)
            prefix_count = int(getattr(encoder.model, "num_prefix_tokens", 0) or 0)
            return self._finish(tokens[:, prefix_count:])

        if isinstance(encoder, fe.EUPEViTEncoder):
            outputs = encoder.model.forward_features(images)
            return self._finish(outputs["x_norm_patchtokens"])

        if isinstance(encoder, fe.PixioEncoder):
            tokens = encoder.encoder(encoder.embeddings(images))
            if encoder.readout == "post-ln":
                tokens = encoder.layernorm(tokens)
            return self._finish(tokens[:, encoder.n_cls_tokens :])

        if isinstance(encoder, fe.TokLIPEncoder):
            return self._finish(encoder.encode_tokens(encoder.trunk, images))

        # These tokenizer wrappers already mean-pool their native spatial tokens.
        already_patch_mean = (
            fe.RAEv2LatentEncoder,
            fe.UniTokEncoder,
            fe.UniARBSQEncoder,
            fe.VQGANEncoder,
            fe.VilaUEncoder,
        )
        if isinstance(encoder, already_patch_mean):
            features = encoder(images)
            configured = getattr(encoder, "tokens_per_image", None)
            self.last_n_tokens = None if configured is None else int(configured)
            return features.float()

        raise NotImplementedError(
            f"No patch-mean LAR adapter for {type(encoder).__module__}.{type(encoder).__name__}"
        )


def _loader_args() -> SimpleNamespace:
    model_zoo = WORKSPACE / "TokBench" / "tokenizer_modelzoo"
    continuous = model_zoo / "continuous"
    image_scripts = WORKSPACE / "TokBench" / "tokenzier_vae_scripts" / "image_scripts"
    values: dict[str, object] = {
        "continuous_model_root": str(continuous),
        "image_scripts": image_scripts,
        "metaclip_model": "vit_base_patch16_clip_224.metaclip_2pt5b",
        "metaclip_checkpoint": str(
            model_zoo / "MetaCLIP" / "vit_base_patch16_clip_224.metaclip_2pt5b"
        ),
        "clip_openai_model_path": str(continuous / "clip_openai__l14"),
        "clip_meta_model": "vit_large_patch14_clip_224.metaclip_2pt5b",
        "clip_meta_checkpoint": str(continuous / "mc1_l14_224_2.5b"),
        "dinov3_path": str(image_scripts / "dinov3"),
        "pixio_readout": "post-ln",
        "toklip_path": str(image_scripts / "TokLIP"),
        "toklip_s_checkpoint": str(model_zoo / "TokLIP" / "TokLIP_S_256.pt"),
        "toklip_l_checkpoint": str(model_zoo / "TokLIP" / "TokLIP_L_384.pt"),
        "toklip_vq_checkpoint": str(model_zoo / "TokLIP" / "vq_ds16_t2i.pt"),
        "unitok_path": str(image_scripts / "UniTok"),
        "unitok_checkpoint": str(model_zoo / "unitok_20250227" / "unitok_tokenizer.pth"),
    }
    for model_name, (_architecture, path_arg, _size, _dim) in fe.SIGLIP2_SPECS.items():
        values[path_arg] = str(continuous / model_name)
    for model_name, (_architecture, checkpoint_arg, _representation) in fe.MC1_SPECS.items():
        values[checkpoint_arg] = str(continuous / model_name)
    return SimpleNamespace(**values)


def load_patch_bundle(loader_name: str, device: torch.device):
    """Reuse all checkpoint/transform defaults from the linear-probe loader."""
    bundle = fe.load_feature_bundle(loader_name, _loader_args(), device)
    if device.type != "cuda":
        bundle.autocast_context = nullcontext
    bundle.encoder = PatchMeanEncoder(bundle.encoder).to(device).eval().requires_grad_(False)
    bundle.representation = "mean of final spatial patch tokens; all prefix/CLS tokens excluded"
    return bundle
