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


def _ensure_transformers_torch_compatibility() -> None:
    """Keep optional Transformers FP8 imports working with PyTorch 2.6.

    Transformers 5.10 imports its fine-grained FP8 integration while lazily
    resolving ordinary vision classes such as CLIPImageProcessor.  That module
    references ``torch.float8_e8m0fnu``, which is absent from the workspace's
    PyTorch 2.6 even though none of the LAR visual encoders use that FP8 dtype.
    Install the same inert dtype alias before any feature loader can import
    Transformers.
    """
    if not hasattr(torch, "float8_e8m0fnu"):
        fallback = getattr(torch, "float8_e4m3fn", None)
        if fallback is None:
            raise RuntimeError(
                "This Transformers version expects an FP8 dtype symbol unavailable "
                f"in PyTorch {torch.__version__}; install a compatible PyTorch/Transformers pair"
            )
        torch.float8_e8m0fnu = fallback


_ensure_transformers_torch_compatibility()

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

        if isinstance(encoder, fe.HFConvNeXtGlobalEncoder):
            outputs = encoder.model(pixel_values=images, return_dict=True)
            spatial = outputs.last_hidden_state
            if spatial.ndim != 4:
                raise RuntimeError(f"Unexpected ConvNeXt map shape: {tuple(spatial.shape)}")
            self.last_n_tokens = int(spatial.shape[-2] * spatial.shape[-1])
            return outputs.pooler_output.float()

        if isinstance(encoder, fe.MetaCLIPEncoder):
            patches, _prefix = encoder.model.get_intermediate_layers(
                images, n=1, return_prefix_tokens=True, norm=True
            )[-1]
            return self._finish(patches)

        if isinstance(encoder, fe.MetaCLIP2DistilledEncoder):
            visual = encoder.visual
            x = visual.conv1(images)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = visual.class_embedding.to(x.dtype)
            cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat((cls, x), dim=1)
            x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype))
            x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
            return self._finish(visual.ln_post(x[:, 1:]))

        if isinstance(encoder, fe.OpenAIClipEncoder):
            tokens = encoder.model(pixel_values=images, return_dict=True).last_hidden_state
            # Transformers <=4.x wrapped the vision tower under
            # ``model.vision_model``.  In 5.x CLIPVisionModel is the tower
            # itself, so post_layernorm lives directly on ``model``.
            vision_tower = getattr(encoder.model, "vision_model", encoder.model)
            post_layernorm = getattr(vision_tower, "post_layernorm", None)
            if post_layernorm is None:
                raise RuntimeError(
                    f"Unsupported CLIP vision layout: {type(encoder.model).__module__}."
                    f"{type(encoder.model).__name__} has no post_layernorm"
                )
            normalized = post_layernorm(tokens[:, 1:])
            return self._finish(normalized)

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

        if isinstance(encoder, fe.EUPEConvNeXtEncoder):
            self.last_n_tokens = int((images.shape[-2] // 32) * (images.shape[-1] // 32))
            return encoder(images).float()

        if isinstance(encoder, fe.PixioEncoder):
            tokens = encoder.encoder(encoder.embeddings(images))
            if encoder.readout == "post-ln":
                tokens = encoder.layernorm(tokens)
            return self._finish(tokens[:, encoder.n_cls_tokens :])

        if isinstance(encoder, fe.TokLIPEncoder):
            return self._finish(encoder.encode_tokens(encoder.trunk, images))

        if isinstance(encoder, fe.UniTokEncoder):
            tokens = encoder.model.encoder(images).float()
            tokens = encoder.model.quant_proj(tokens)
            indices = encoder.model.quantizer.f_to_idx(tokens)
            tokens = encoder.model.quantizer.idx_to_f(indices)
            tokens = encoder.model.post_quant_proj(tokens)
            self.last_n_tokens = int(tokens.shape[1])
            return encoder.model.fc_norm(tokens.mean(dim=1)).float()

        if isinstance(encoder, fe.VQGANEncoder):
            latent = encoder.quant_conv(encoder.encoder(images))
            quantized, _embedding_loss, _info = encoder.quantize(latent)
            self.last_n_tokens = int(quantized.shape[2] * quantized.shape[3])
            return quantized.mean(dim=(2, 3)).float()

        if isinstance(encoder, fe.RAEv2LatentEncoder):
            self.last_n_tokens = int(encoder.latent_mean.shape[-2] * encoder.latent_mean.shape[-1])
            return encoder(images).float()

        if isinstance(encoder, fe.UniARBSQEncoder):
            self.last_n_tokens = int(
                (encoder.image_size // encoder.patch_size) ** 2 // encoder.merge_size**2
            )
            return encoder(images).float()

        if isinstance(encoder, fe.VilaUEncoder):
            config = encoder.model.siglip_model.vision_model.config
            patch_size = int(config.patch_size)
            self.last_n_tokens = int(
                (images.shape[-2] // patch_size) * (images.shape[-1] // patch_size)
            )
            return encoder(images).float()

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
        "raev2_model_root": str(model_zoo / "RAEv2-models"),
        "raev2_path": str(image_scripts / "RAEv2"),
        "pixio_readout": "post-ln",
        "toklip_path": str(image_scripts / "TokLIP"),
        "toklip_s_checkpoint": str(model_zoo / "TokLIP" / "TokLIP_S_256.pt"),
        "toklip_l_checkpoint": str(model_zoo / "TokLIP" / "TokLIP_L_384.pt"),
        "toklip_vq_checkpoint": str(model_zoo / "TokLIP" / "vq_ds16_t2i.pt"),
        "unitok_path": str(image_scripts / "UniTok"),
        "unitok_checkpoint": str(model_zoo / "unitok_20250227" / "unitok_tokenizer.pth"),
        "uniar_path": str(image_scripts / "UniAR"),
        "uniar_checkpoint": str(model_zoo / "uniar_bsq" / "bsq_encoder"),
        "uniar_image_size": 256,
        "vilau_path": str(image_scripts / "vila-u"),
        "vilau_model_path": str(model_zoo / "VILA-U" / "vila-u-7b-256"),
        "vilau_siglip_config": str(model_zoo / "VILA-U" / "siglip-large-patch16-256"),
        "vqgan_path": str(image_scripts / "taming-transformers"),
        "vqgan_config": str(
            model_zoo / "taming_vqgan_imagenet_f16_16384" / "model.yaml"
        ),
        "vqgan_checkpoint": str(
            model_zoo / "taming_vqgan_imagenet_f16_16384" / "last.ckpt"
        ),
    }
    for model_name, (_architecture, path_arg, _size, _dim) in fe.SIGLIP2_SPECS.items():
        values[path_arg] = str(continuous / model_name)
    for model_name, (_architecture, checkpoint_arg, _representation) in fe.MC1_SPECS.items():
        values[checkpoint_arg] = str(continuous / model_name)
    for model_name, (_architecture, checkpoint_arg, _representation) in fe.MC2_TIMM_SPECS.items():
        values[checkpoint_arg] = str(continuous / model_name)
    mc2_filenames = {
        "mc2_s16_224": "metaclip2_s16_224px_worldwide.pt",
        "mc2_s16_384": "metaclip2_s16_384px_worldwide.pt",
        "mc2_s16_224_mt5": "metaclip2_s16_224px_mt5_worldwide.pt",
        "mc2_m16_224": "metaclip2_m16_224px_worldwide.pt",
        "mc2_m16_384": "metaclip2_m16_384px_worldwide.pt",
        "mc2_m16_224_mt5": "metaclip2_m16_224px_mt5_worldwide.pt",
        "mc2_b32_224": "metaclip2_b32_224px_worldwide.pt",
        "mc2_b32_384": "metaclip2_b32_384px_worldwide.pt",
        "mc2_b32_224_mt5": "metaclip2_b32_224px_mt5_worldwide.pt",
        "mc2_b16_224": "metaclip2_b16_224px_worldwide.pt",
        "mc2_b16_384": "metaclip2_b16_384px_worldwide.pt",
        "mc2_l14_224": "metaclip2_l14_224px_worldwide.pt",
    }
    for model_name, (checkpoint_arg, *_rest) in fe.MC2_DISTILLED_SPECS.items():
        values[checkpoint_arg] = str(continuous / model_name / mc2_filenames[model_name])
    return SimpleNamespace(**values)


def load_patch_bundle(loader_name: str, device: torch.device):
    """Reuse all checkpoint/transform defaults from the linear-probe loader."""
    bundle = fe.load_feature_bundle(loader_name, _loader_args(), device)
    if device.type != "cuda":
        bundle.autocast_context = nullcontext
    bundle.encoder = PatchMeanEncoder(bundle.encoder).to(device).eval().requires_grad_(False)
    bundle.representation = "mean of final spatial patch tokens; all prefix/CLS tokens excluded"
    return bundle
