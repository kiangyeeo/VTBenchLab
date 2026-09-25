#!/usr/bin/env python3
"""Build the epoch-1 linear-probe/FLOPs table requested for the paper models.

FLOPs use the common vision-model convention where one multiply-accumulate is
one operation.  The count covers one visual-encoder forward pass through the
representation used by the probe, but not image preprocessing or the linear
classifier.  LayerNorm, activation, softmax, residual adds, and pooling adds
are omitted, matching the usual quoted ViT MAC/FLOPs numbers (for example,
about 17.6 G for ViT-B/16 at 224 px).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_ROOT = REPO_ROOT / "outputs/vae_linear_probing_dinov2_single_paperlr"
BN_ROOT = REPO_ROOT / "outputs/vae_linear_probing_dinov2_single_paperlr_bn"
OUTPUT = Path(__file__).with_name("first_epoch_flops.csv")


@dataclass(frozen=True)
class Model:
    rank: int
    label: str
    experiment: str
    source_root: Path
    image_size: int
    patch_size: int
    width: int
    depth: int
    mlp_hidden: int
    prefix_tokens: int = 1
    map_pool: bool = False
    notes: str = ""


def vit_macs(model: Model) -> int:
    """Analytic dense-ViT MAC count through the requested probe readout."""
    patches_per_side = model.image_size // model.patch_size
    patch_tokens = patches_per_side**2
    tokens = patch_tokens + model.prefix_tokens
    d = model.width

    # Strided-convolution patch embedding.
    total = patch_tokens * d * 3 * model.patch_size**2

    # Per block: QKV and output projections, attention QK/AV matmuls, MLP.
    block = (
        4 * tokens * d**2
        + 2 * tokens**2 * d
        + 2 * tokens * d * model.mlp_hidden
    )
    total += model.depth * block

    # SigLIP 2 and PE-Core use timm's one-query AttentionPoolLatent.  It has
    # Q and KV projections, QK/AV matmuls, an output projection, and an MLP.
    if model.map_pool:
        pool = (
            d**2
            + 2 * tokens * d**2
            + 2 * tokens * d
            + d**2
            + 2 * d * model.mlp_hidden
        )
        total += pool
    return total


def m(
    rank: int,
    label: str,
    experiment: str,
    image_size: int,
    patch_size: int,
    width: int,
    depth: int,
    mlp_ratio: float = 4.0,
    *,
    prefix_tokens: int = 1,
    map_pool: bool = False,
    source_root: Path = MAIN_ROOT,
    notes: str = "",
) -> Model:
    return Model(
        rank,
        label,
        experiment,
        source_root,
        image_size,
        patch_size,
        width,
        depth,
        int(width * mlp_ratio),
        prefix_tokens,
        map_pool,
        notes,
    )


MODELS = [
    m(1, "SigLIP2 So400m/14 (384)", "siglip2_sm14_384", 378, 14, 1152, 27, 3.7362, prefix_tokens=0, map_pool=True, notes="checkpoint name says 384; actual native crop is 378"),
    m(2, "SigLIP2 ViT-G/16 (384)", "siglip2_g16_384", 384, 16, 1536, 40, prefix_tokens=0, map_pool=True),
    m(3, "SigLIP2 ViT-L/16 (384)", "siglip2_l16_384", 384, 16, 1024, 24, prefix_tokens=0, map_pool=True),
    m(4, "SigLIP2 So400m/16 (512)", "siglip2_sm16_512", 512, 16, 1152, 27, 3.7362, prefix_tokens=0, map_pool=True),
    m(5, "SigLIP2 So400m/16 (384)", "siglip2_sm16_384", 384, 16, 1152, 27, 3.7362, prefix_tokens=0, map_pool=True),
    m(6, "SigLIP2 ViT-G/16 (256)", "siglip2_g16_256", 256, 16, 1536, 40, prefix_tokens=0, map_pool=True),
    m(7, "MetaCLIP 2 ViT-G/14 (378)", "mc2_g14_378", 378, 14, 1664, 48, 64 / 13),
    m(8, "SigLIP2 So400m/14 (224)", "siglip2_sm14_224", 224, 14, 1152, 27, 3.7362, prefix_tokens=0, map_pool=True),
    m(9, "PE-Core-G/14 (448)", "pe_core_g14_448", 448, 14, 1536, 50, 8960 / 1536, prefix_tokens=0, map_pool=True),
    m(10, "SigLIP2 So400m/16 (256)", "siglip2_sm16_256", 256, 16, 1152, 27, 3.7362, prefix_tokens=0, map_pool=True),
    m(11, "SigLIP2 ViT-L/16 (256)", "siglip2_l16_256", 256, 16, 1024, 24, prefix_tokens=0, map_pool=True),
    m(12, "MetaCLIP ViT-G/14 (2.5B, 224)", "mc1_g14_224_2.5b", 224, 14, 1664, 48, 64 / 13),
    m(13, "MetaCLIP 2 ViT-G/14 (224)", "mc2_g14_224", 224, 14, 1664, 48, 64 / 13),
    m(14, "SigLIP2 ViT-B/16 (512)", "siglip2_b16_512", 512, 16, 768, 12, prefix_tokens=0, map_pool=True),
    m(15, "MetaCLIP ViT-H/14 (v1.2, 224)", "mc1_h14_224_v1.2", 224, 14, 1280, 32),
    m(16, "OpenAI CLIP ViT-L/14 (224)", "clip_openai__l14", 224, 14, 1024, 24),
    m(17, "MetaCLIP ViT-H/14 (2.5B, 224)", "mc1_h14_224_2.5b", 224, 14, 1280, 32),
    m(18, "MetaCLIP ViT-L/14 (2.5B, 224)", "mc1_l14_224_2.5b", 224, 14, 1024, 24),
    m(19, "SigLIP2 ViT-B/16 (256)", "siglip2_b16_256", 256, 16, 768, 12, prefix_tokens=0, map_pool=True),
    m(20, "SigLIP2 ViT-B/16 (224)", "siglip2_b16_224", 224, 16, 768, 12, prefix_tokens=0, map_pool=True),
    m(21, "MetaCLIP 2 ViT-L/14 (224)", "mc2_l14_224", 224, 14, 1024, 24),
    m(22, "PE-Lang-L/14 (448)", "pe_lang_l14_448", 448, 14, 1024, 23),
    m(23, "PE-Core-B/16 (224)", "pe_core_b16_224", 224, 16, 768, 12, map_pool=True),
    m(24, "MetaCLIP ViT-B/16 (400M, 224)", "mc1_b16_224_400m", 224, 16, 768, 12),
    m(25, "MetaCLIP ViT-B/16 (2.5B, 224)", "mc1_b16_224_2.5b", 224, 16, 768, 12),
    m(26, "Pixio ViT-L/16", "pixio_vitl16", 256, 16, 1024, 24, prefix_tokens=8),
    m(27, "DINOv2 ViT-G/14", "dinov2_giant", 224, 14, 1536, 40),
    m(28, "DINOv2 ViT-L/14", "dinov2_large", 224, 14, 1024, 24),
    m(29, "Pixio ViT-B/16", "pixio_vitb16", 256, 16, 768, 12, prefix_tokens=8),
    m(30, "EUPE ViT-B", "eupe_vit_b", 256, 16, 768, 12, prefix_tokens=5),
    m(31, "EUPE ViT-S", "eupe_vit_s", 256, 16, 384, 12, prefix_tokens=5),
    m(32, "MetaCLIP 2 ViT-S/16 (224)", "mc2_s16_224", 224, 16, 384, 12),
    m(33, "DINOv2 ViT-B/14", "dinov2_base", 224, 14, 768, 12),
    m(34, "DINOv2 ViT-S/14", "dinov2_small", 224, 14, 384, 12),
    m(35, "Pixio ViT-H/16", "pixio_vith16", 256, 16, 1280, 32, prefix_tokens=8),
    m(36, "I-JEPA ViT-H/14", "ijepa", 224, 14, 1280, 32, prefix_tokens=0, notes="probe preprocesses at 256 then resizes encoder input to 224"),
    m(37, "DINO ViT-B/16", "dinov1_vitb16", 224, 16, 768, 12),
    m(38, "Web-SSL MAE 3B (224)", "webssl_mae3b_full2b_224_cls", 224, 14, 3072, 26, source_root=BN_ROOT, notes="BN protocol, CLS readout, as requested"),
    m(39, "EUPE ViT-T", "eupe_vit_t", 256, 16, 192, 12, prefix_tokens=5),
    m(40, "DINO ViT-S/8", "dinov1_vits8", 224, 8, 384, 12),
    m(41, "DINO ViT-B/8", "dinov1_vitb8", 224, 8, 768, 12),
    m(42, "DINO ViT-S/16", "dinov1_vits16", 224, 16, 384, 12),
]


def first_epoch(model: Model) -> tuple[dict | None, Path]:
    path = model.source_root / model.experiment / "metrics_history.jsonl"
    if not path.is_file():
        return None, path
    with path.open(encoding="utf-8") as handle:
        first_line = handle.readline()
    return json.loads(first_line), path


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "vision_encoder",
        "experiment",
        "epoch1_top1_pct",
        "epoch1_top5_pct",
        "epoch1_effective_lr",
        "epoch1_iteration",
        "flops_g_mac1",
        "flops_g_fma2",
        "encoder_input_px",
        "status",
        "metrics_path",
        "notes",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in MODELS:
            epoch, metrics_path = first_epoch(model)
            macs = vit_macs(model)
            best = None if epoch is None else epoch["best_classifier"]
            writer.writerow(
                {
                    "rank": model.rank,
                    "vision_encoder": model.label,
                    "experiment": model.experiment,
                    "epoch1_top1_pct": "" if best is None else f'{100 * best["accuracy"]:.3f}',
                    "epoch1_top5_pct": "" if best is None else f'{100 * best["top5_accuracy"]:.3f}',
                    "epoch1_effective_lr": "" if best is None else best["effective_lr"],
                    "epoch1_iteration": "" if epoch is None else epoch["iteration"],
                    "flops_g_mac1": f"{macs / 1e9:.3f}",
                    "flops_g_fma2": f"{2 * macs / 1e9:.3f}",
                    "encoder_input_px": model.image_size,
                    "status": "missing epoch-1 metrics" if epoch is None else "ok",
                    "metrics_path": metrics_path.relative_to(REPO_ROOT),
                    "notes": model.notes,
                }
            )
    print(OUTPUT)


if __name__ == "__main__":
    main()
