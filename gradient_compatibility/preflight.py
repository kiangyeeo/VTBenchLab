from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from lar.model_adapters import _loader_args, fe
from lar.prepare_e3_models import supported_loaders

from .utils import atomic_write_json, canonical_hash, load_config, resolve_path


def _paths_for_loader(loader: str) -> list[Path]:
    args = _loader_args()
    continuous = Path(args.continuous_model_root)
    if loader in fe.SIGLIP2_SPECS:
        return [Path(getattr(args, fe.SIGLIP2_SPECS[loader][1]))]
    if loader in fe.MC1_SPECS:
        return [Path(getattr(args, fe.MC1_SPECS[loader][1]))]
    if loader in fe.MC2_TIMM_SPECS:
        return [Path(getattr(args, fe.MC2_TIMM_SPECS[loader][1]))]
    if loader in fe.MC2_DISTILLED_SPECS:
        return [Path(getattr(args, fe.MC2_DISTILLED_SPECS[loader][0]))]
    for table_name in (
        "PE_SPECS", "DINO_VIT_SPECS", "DINOV3_CONVNEXT_SPECS",
        "WEBSSL_DINO_SPECS", "WEBSSL_MAE_SPECS", "PIXIO_SPECS",
    ):
        table = getattr(fe, table_name)
        if loader in table:
            return [continuous / table[loader][0]]
    if loader in fe.EUPE_SPECS:
        directory = fe.EUPE_SPECS[loader][0]
        return [continuous / directory / f"{directory}.pt"]
    if loader == "clip_openai__l14":
        return [Path(args.clip_openai_model_path)]
    if loader == "toklip_s":
        return [Path(args.toklip_path), Path(args.toklip_s_checkpoint), Path(args.toklip_vq_checkpoint)]
    if loader == "toklip_l":
        return [Path(args.toklip_path), Path(args.toklip_l_checkpoint), Path(args.toklip_vq_checkpoint)]
    if loader == "unitok":
        return [Path(args.unitok_path), Path(args.unitok_checkpoint)]
    if loader == "uniar_bsq":
        return [Path(args.uniar_path), Path(args.uniar_checkpoint)]
    if loader == "vilau":
        return [Path(args.vilau_path), Path(args.vilau_model_path), Path(args.vilau_siglip_config)]
    if loader in fe.RAEV2_SPECS:
        root = Path(args.raev2_model_root)
        stats = fe.RAEV2_SPECS[loader]["stats"]
        paths = [root / stats, Path(args.raev2_path)]
        if loader in {"raev2", "dinov3"}:
            paths += [Path(args.dinov3_path), root / fe.DINOV3_L_CHECKPOINT]
        else:
            paths.append(root / fe.IJEPA_H_CHECKPOINT)
        return paths
    return []


def run(config: dict) -> None:
    errors = []
    names = list(config["tokenizers"])
    loaders = [spec["loader_name"] for spec in config["tokenizers"].values()]
    unsupported = sorted(set(loaders) - supported_loaders())
    if unsupported:
        errors.append(f"unsupported loaders: {unsupported}")
    shard_size = int(config["runtime"]["token_shard_size"])
    invalid_batches = [
        (name, int(config["tokenizers"][name]["extract_batch_size"]))
        for name in names
        if shard_size % int(config["tokenizers"][name]["extract_batch_size"])
    ]
    if invalid_batches:
        errors.append(f"extract batches do not divide shard size {shard_size}: {invalid_batches}")

    required = [resolve_path(config, config["llm"]["path"])]
    data = config["data"]
    required += [
        resolve_path(config, data[key])
        for key in (
            "coco_train_annotations", "coco_train_images", "coco_val_annotations",
            "coco_val_images", "scienceqa_dir",
        )
    ]
    for loader in sorted(set(loaders)):
        paths = _paths_for_loader(loader)
        if not paths:
            errors.append(f"no preflight path rule for loader {loader}")
        required.extend(paths)
    missing = sorted({str(path) for path in required if not path.exists()})
    if missing:
        errors.append("missing paths:\n  " + "\n  ".join(missing))

    artifact_root = resolve_path(config, config["runtime"]["artifact_root"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(artifact_root).free / 2**40
    if free < 0.25:
        errors.append(f"only {free:.2f} TiB free under {artifact_root}")
    if errors:
        raise RuntimeError("Full-sweep preflight failed:\n- " + "\n- ".join(errors))
    snapshot = {
        "schema_version": 1,
        "protocol": config["protocol"],
        "runtime": {
            key: config["runtime"][key]
            for key in ("feature_dtype", "token_shard_size")
        },
        "data": config["data"],
        "llm": config["llm"],
        "projector_training": config["projector_training"],
        "probe": config["probe"],
        "tokenizers": config["tokenizers"],
    }
    snapshot["fingerprint"] = canonical_hash(snapshot)
    snapshot_path = artifact_root / "protocol_snapshot.json"
    if snapshot_path.is_file():
        previous = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != snapshot["fingerprint"]:
            raise RuntimeError(
                f"Frozen protocol changed under existing artifact root {artifact_root}; "
                "use a new artifact_root"
            )
    else:
        atomic_write_json(snapshot_path, snapshot)
    print(
        f"Preflight OK: {len(names)} tokenizers, {len(set(loaders))} loaders, "
        f"{free:.2f} TiB free."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Static checks for the frozen sweep")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
