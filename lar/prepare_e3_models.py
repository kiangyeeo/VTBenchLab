#!/usr/bin/env python
"""Build the E3 visual-extraction config from the evaluation target table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

try:
    from .data import WORKSPACE
    from .model_adapters import fe
except ImportError:  # Direct execution
    from data import WORKSPACE
    from model_adapters import fe


ALIASES = {
    "i-jepa": "ijepa",
    "i_jepa": "ijepa",
    "unitok_attn": "unitok",
    "toklip_s_256": "toklip_s",
    "toklip_l_384": "toklip_l",
    "toklip_s_semantic_256": "toklip_s",
    "toklip_l_semantic_384": "toklip_l",
    "vilau_256": "vilau",
    "vilau_7b_256_semantic_penultimate": "vilau",
    "uniar_bsq_final27_256": "uniar_bsq",
    "vqgan_imagenet_f16_16384": "vqgan",
    "metaclip_b16_2pt5b": "metaclip",
    "dino_vitb16": "dinov1_vitb16",
    "dino_vitb8": "dinov1_vitb8",
    "dino_vits16": "dinov1_vits16",
    "dino_vits8": "dinov1_vits8",
    "dinov3_vitl16": "dinov3_vitl16_lvd1689m",
    "dinov3_vitb16": "dinov3_vitb16_lvd1689m",
    "dinov3_vits16": "dinov3_vits16_lvd1689m",
}
EXPECTED_FAMILIES = {
    "siglip2", "mc1", "mc2", "pe_core", "pe_lang", "dinov2", "dinov3", "dino",
    "webssl_mae", "pixio", "ijepa", "eupe", "raev2", "clip", "toklip", "unitok",
    "vilau", "uniar",
}


def supported_loaders() -> set[str]:
    names = {
        "metaclip", "clip_openai__l14", "clip_meta__l14",
        "toklip_s", "toklip_l", "unitok", "uniar_bsq", "vilau", "vqgan",
    }
    for attribute in (
        "MC1_SPECS", "MC2_TIMM_SPECS", "MC2_DISTILLED_SPECS", "SIGLIP2_SPECS",
        "RAEV2_SPECS", "PE_SPECS", "DINO_VIT_SPECS", "DINOV3_CONVNEXT_SPECS",
        "WEBSSL_DINO_SPECS", "WEBSSL_MAE_SPECS", "PIXIO_SPECS", "EUPE_SPECS",
    ):
        names.update(getattr(fe, attribute))
    return names


def canonical_loader(name: str) -> str:
    stripped = name.strip()
    lowered = stripped.lower()
    if lowered in ALIASES:
        return ALIASES[lowered]
    for suffix in ("_pre_ln_bn", "_mae_bn", "_cls"):
        if lowered.endswith(suffix):
            return stripped[: -len(suffix)]
    return stripped


def infer_family(name: str, loader: str) -> str:
    lowered = name.lower()
    if loader in {"clip_openai__l14", "clip_meta__l14", "metaclip"}:
        return "clip"
    for prefix, family in (
        ("siglip2", "siglip2"), ("mc1", "mc1"), ("mc2", "mc2"),
        ("pe_core", "pe_core"), ("pe_lang", "pe_lang"),
        ("dinov2", "dinov2"), ("dinov3", "dinov3"), ("dinov1", "dino"),
        ("webssl_mae", "webssl_mae"), ("webssl_dino", "dino"),
        ("pixio", "pixio"), ("eupe", "eupe"), ("raev2", "raev2"),
        ("toklip", "toklip"), ("unitok", "unitok"), ("uniar", "uniar"),
        ("vilau", "vilau"), ("i-jepa", "ijepa"), ("ijepa", "ijepa"),
    ):
        if lowered.startswith(prefix) or loader.startswith(prefix):
            return family
    return "other"


def batch_size(loader: str) -> int:
    if any(token in loader for token in ("5b", "7b", "vit7b", "gigantic", "_g14", "_g16")):
        return 4
    if any(
        token in loader
        for token in (
            "1b", "2b", "3b", "giant", "huge", "vit1b", "vith", "vitl",
            "_h14", "_l14", "_l16", "raev2", "ijepa", "uniar",
        )
    ) or loader == "dinov3":
        return 8
    if any(token in loader for token in ("large", "vitb", "_b16", "_b32")):
        return 16
    return 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets", type=Path,
        default=WORKSPACE / "lar" / "configs" / "e3_targets.csv",
        help="Canonical CSV containing at least name (and preferably family).",
    )
    parser.add_argument(
        "--protocol-roots", type=Path, nargs="+",
        default=(
            WORKSPACE / "outputs" / "vae_linear_probing_dinov2_single_paperlr",
            WORKSPACE / "outputs" / "vae_linear_probing_dinov2_single_noaug_cached_paperlr_bn",
            WORKSPACE / "outputs" / "food101_linear_probing_dinov2_single_surface",
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=WORKSPACE / "lar" / "configs" / "models_e3.yaml",
    )
    parser.add_argument("--allow-unsupported", action="store_true")
    return parser.parse_args()


def protocol_index(roots: list[Path] | tuple[Path, ...]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("**/protocol.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            loader = str(payload.get("model", "")).strip()
            if loader:
                index.setdefault(loader, path)
    return index


def main() -> None:
    args = parse_args()
    with args.targets.open("r", encoding="utf-8-sig", newline="") as handle:
        target_rows = list(csv.DictReader(handle))
    names = [row.get("name", "").strip() for row in target_rows if row.get("name", "").strip()]
    if not names:
        raise RuntimeError(f"No model names in {args.targets}")
    if len(names) != len(set(names)):
        raise RuntimeError("Target table contains duplicate model names")

    supported = supported_loaders()
    protocols = protocol_index(args.protocol_roots)
    models = []
    errors = []
    for row, name in zip((row for row in target_rows if row.get("name", "").strip()), names):
        loader = canonical_loader(name)
        if loader not in supported:
            errors.append(f"{name}: unsupported loader {loader}")
            continue
        protocol = protocols.get(loader)
        configured_family = row.get("family", "").strip().lower()
        family = (
            configured_family if configured_family in EXPECTED_FAMILIES
            else infer_family(name, loader)
        )
        models.append(
            {
                "name": name,
                "family": family,
                "loader_name": loader,
                "probing_protocol": (
                    None if protocol is None else str(protocol.relative_to(WORKSPACE))
                ),
                "batch_size": batch_size(loader),
                "enabled": True,
            }
        )
    if errors and not args.allow_unsupported:
        raise RuntimeError("Cannot configure the full target pool:\n" + "\n".join(errors))
    for error in errors:
        print(f"WARNING: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump({"models": models}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"wrote {len(models)}/{len(names)} models to {args.output}")


if __name__ == "__main__":
    main()
