from __future__ import annotations

import argparse
import gc
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from tqdm import tqdm

from vtm_lcg.config import (
    load_phase1_full_config,
    resolve_project_path,
    torch_dtype_from_name,
)
from vtm_lcg.eval import compute_vtm_lcg_scores
from vtm_lcg.eval.sanity_checks import (
    caption_keep_mask,
    make_deterministic_mask,
    select_caption_texts,
    spatially_shuffle_visible_tokens,
)
from vtm_lcg.models import FrozenClipTextConditioner, MaskedVisualPredictor
from vtm_lcg.train.streaming_data import (
    Phase0ShardCache,
    validate_karpathy_split_caches,
)
from vtm_lcg.train.train_predictor import (
    atomic_save_safetensors,
    autocast_context,
    build_model,
    make_scheduler,
    masked_mse,
    normalize_visual,
    resolve_device,
    seed_everything,
    utc_now,
)
from vtm_lcg.utils import (
    atomic_write_json,
    code_fingerprint,
    sha256_file,
    sha256_json,
)


def _resolve_text_weight(checkpoint: Path) -> Path:
    for filename in ("model.safetensors", "pytorch_model.bin"):
        candidate = checkpoint / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No CLIP text weights found in {checkpoint}")


def build_text_identity(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    checkpoint = Path(config["text"]["checkpoint"]).resolve()
    asset_hashes: dict[str, str] = {}
    for filename in (
        "config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        path = checkpoint / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing CLIP tokenizer asset: {path}")
        asset_hashes[filename] = sha256_file(path)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(_resolve_text_weight(checkpoint)),
        "tokenizer_assets": asset_hashes,
        "max_length": int(config["text"]["max_length"]),
        "mode": str(config["text"]["mode"]),
        "conditioner_code_sha256": sha256_file(
            project_root / "vtm_lcg" / "models" / "text_conditioner.py"
        ),
    }


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def recursive_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: recursive_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(recursive_to_cpu(item) for item in value)
    return value


def optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def train_streaming_epoch(
    *,
    model: MaskedVisualPredictor,
    text_conditioner: FrozenClipTextConditioner,
    cache: Phase0ShardCache,
    mean: torch.Tensor,
    std: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    epoch: int,
    verify_checksum: bool,
) -> dict[str, float]:
    model.train()
    predictor = config["predictor"]
    training = config["training"]
    evaluation = config["evaluation"]
    error_sum = 0.0
    element_count_sum = 0
    kept_captions = 0
    example_count = 0
    batches = cache.iter_batches(
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=seed + epoch,
        verify_checksum=verify_checksum,
    )
    progress = tqdm(
        batches,
        total=cache.batch_count(int(training["batch_size"])),
        desc=f"{cache.tokenizer_id} epoch {epoch + 1}",
        unit="batch",
        dynamic_ncols=True,
    )
    for raw_values, record_indices in progress:
        visual = normalize_visual(raw_values, mean, std, device)
        record_indices_list = [int(value) for value in record_indices.tolist()]
        masked_positions = make_deterministic_mask(
            record_indices_list,
            token_count=visual.shape[1],
            mask_ratio=float(predictor["mask_ratio"]),
            seed=int(evaluation["mask_seed"]) + seed,
            epoch=epoch,
            device=device,
        )
        captions = select_caption_texts(
            cache.records,
            record_indices_list,
            seed=int(evaluation["caption_seed"]) + seed,
            epoch=epoch,
            fixed_first=False,
        )
        text_embeddings, text_attention = text_conditioner.encode(captions)
        keep_caption = caption_keep_mask(
            record_indices_list,
            dropout=float(predictor["caption_dropout"]),
            seed=int(evaluation["caption_seed"]) + seed + 7_919,
            epoch=epoch,
            device=device,
        )
        text_attention = text_attention & keep_caption.unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, str(training["precision"])):
            prediction = model(
                visual,
                masked_positions,
                text_embeddings=text_embeddings,
                text_attention_mask=text_attention,
            )
            loss = masked_mse(prediction, visual, masked_positions)
        loss.backward()
        optimizer.step()
        scheduler.step()

        element_count = int(masked_positions.sum().item() * visual.shape[-1])
        error_sum += float(loss.detach().item()) * element_count
        element_count_sum += element_count
        kept_captions += int(keep_caption.sum().item())
        example_count += int(visual.shape[0])
        progress.set_postfix(loss=f"{loss.detach().item():.4f}")
    return {
        "loss": error_sum / element_count_sum,
        "caption_keep_fraction": kept_captions / example_count,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


@torch.inference_mode()
def evaluate_streaming(
    *,
    model: MaskedVisualPredictor,
    text_conditioner: FrozenClipTextConditioner,
    cache: Phase0ShardCache,
    mean: torch.Tensor,
    std: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    full_sanity_checks: bool,
    verify_checksum: bool,
) -> dict[str, Any]:
    model.eval()
    predictor = config["predictor"]
    training = config["training"]
    evaluation = config["evaluation"]
    if str(evaluation["caption_mode"]) != "first":
        raise ValueError("The full-COCO v1 runner currently supports caption_mode=first")
    modes = ["L_mean", "L_visual", "L_visual_text"]
    if full_sanity_checks:
        modes.extend(
            [
                "L_visual_shuffled_text",
                "L_visual_spatial_shuffle",
                "L_no_visible",
            ]
        )
    error_sums = {mode: 0.0 for mode in modes}
    element_count_sum = 0
    batches = cache.iter_batches(
        batch_size=int(evaluation["batch_size"]),
        shuffle=False,
        seed=0,
        verify_checksum=verify_checksum,
    )
    for raw_values, record_indices in tqdm(
        batches,
        total=cache.batch_count(int(evaluation["batch_size"])),
        desc=f"evaluate {cache.split_name} {cache.tokenizer_id}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    ):
        visual = normalize_visual(raw_values, mean, std, device)
        indices = [int(value) for value in record_indices.tolist()]
        masked_positions = make_deterministic_mask(
            indices,
            token_count=visual.shape[1],
            mask_ratio=float(predictor["mask_ratio"]),
            seed=int(evaluation["mask_seed"]),
            epoch=0,
            device=device,
        )
        targets = visual[masked_positions].float()
        element_count = targets.numel()
        error_sums["L_mean"] += float(targets.square().sum().item())

        with autocast_context(device, str(training["precision"])):
            visual_prediction = model(visual, masked_positions)
        error_sums["L_visual"] += float(
            (visual_prediction[masked_positions].float() - targets)
            .square()
            .sum()
            .item()
        )

        true_captions = select_caption_texts(
            cache.records,
            indices,
            seed=int(evaluation["caption_seed"]),
            epoch=0,
            fixed_first=True,
        )
        true_embeddings, true_attention = text_conditioner.encode(true_captions)
        with autocast_context(device, str(training["precision"])):
            text_prediction = model(
                visual,
                masked_positions,
                text_embeddings=true_embeddings,
                text_attention_mask=true_attention,
            )
        error_sums["L_visual_text"] += float(
            (text_prediction[masked_positions].float() - targets)
            .square()
            .sum()
            .item()
        )

        if full_sanity_checks:
            shuffled_captions = [
                str(cache.records[(index + 1) % cache.record_count]["captions"][0])
                for index in indices
            ]
            shuffled_embeddings, shuffled_attention = text_conditioner.encode(
                shuffled_captions
            )
            with autocast_context(device, str(training["precision"])):
                shuffled_prediction = model(
                    visual,
                    masked_positions,
                    text_embeddings=shuffled_embeddings,
                    text_attention_mask=shuffled_attention,
                )
            error_sums["L_visual_shuffled_text"] += float(
                (shuffled_prediction[masked_positions].float() - targets)
                .square()
                .sum()
                .item()
            )
            spatial_values = spatially_shuffle_visible_tokens(
                visual,
                masked_positions,
                indices,
                seed=int(evaluation["shuffle_seed"]),
            )
            with autocast_context(device, str(training["precision"])):
                spatial_prediction = model(spatial_values, masked_positions)
                no_visible_prediction = model(
                    visual,
                    masked_positions,
                    hide_all_visual=True,
                )
            error_sums["L_visual_spatial_shuffle"] += float(
                (spatial_prediction[masked_positions].float() - targets)
                .square()
                .sum()
                .item()
            )
            error_sums["L_no_visible"] += float(
                (no_visible_prediction[masked_positions].float() - targets)
                .square()
                .sum()
                .item()
            )
        element_count_sum += element_count
    losses = {
        mode: error_sums[mode] / element_count_sum for mode in modes
    }
    return {
        "losses": losses,
        "scores": compute_vtm_lcg_scores(losses),
        "target_element_count": element_count_sum,
    }


def build_full_run_identity(
    *,
    tokenizer_id: str,
    caches: dict[str, Phase0ShardCache],
    text_identity: dict[str, Any],
    config: dict[str, Any],
    project_root: Path,
    seed: int,
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "protocol": "coco_karpathy_full_streaming_v1",
        "tokenizer_id": tokenizer_id,
        "phase0_cache_keys": {
            split_name: cache.cache_key for split_name, cache in caches.items()
        },
        "dataset_fingerprints": {
            split_name: cache.dataset_metadata["dataset_fingerprint"]
            for split_name, cache in caches.items()
        },
        "text": text_identity,
        "predictor": config["predictor"],
        "training": config["training"],
        "evaluation": config["evaluation"],
        "seed": seed,
        "phase1_code_sha256": code_fingerprint(project_root / "vtm_lcg"),
    }
    identity["run_key"] = sha256_json(identity)
    return identity


def load_completed_result(
    output_dir: Path,
    identity: dict[str, Any],
) -> dict[str, Any] | None:
    identity_path = output_dir / "run_identity.json"
    result_path = output_dir / "result.json"
    checkpoint_path = output_dir / "predictor.safetensors"
    if not (identity_path.is_file() and result_path.is_file() and checkpoint_path.is_file()):
        return None
    if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        return None
    print(
        f"{result['tokenizer_id']} seed={result['predictor_seed']}: "
        "completed full-COCO result reused"
    )
    return result


def train_full_run(
    *,
    caches: dict[str, Phase0ShardCache],
    text_conditioner: FrozenClipTextConditioner,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    output_dir: Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    tokenizer_id = caches["train"].tokenizer_id
    training = config["training"]
    seed_everything(seed)
    mean_cpu, std_cpu = caches["train"].train_normalization()
    mean = mean_cpu.to(device).view(1, 1, -1)
    std = std_cpu.to(device).view(1, 1, -1)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "run_identity.json", identity)
    atomic_save_safetensors(
        output_dir / "normalization.safetensors",
        {"mean": mean_cpu, "std": std_cpu},
    )

    model = build_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    steps_per_epoch = caches["train"].batch_count(int(training["batch_size"]))
    scheduler = make_scheduler(
        optimizer,
        total_steps=steps_per_epoch * int(training["epochs"]),
        warmup_ratio=float(training["warmup_ratio"]),
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_objective = float("inf")
    best_epoch = -1
    start_epoch = 0
    latest_state_path = output_dir / "latest_training_state.pt"
    if latest_state_path.is_file():
        state = torch.load(latest_state_path, map_location="cpu", weights_only=False)
        if state.get("run_key") == identity["run_key"]:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            optimizer_state_to_device(optimizer, device)
            scheduler.load_state_dict(state["scheduler"])
            history = state["history"]
            best_state = state["best_state"]
            best_objective = float(state["best_objective"])
            best_epoch = int(state["best_epoch"])
            start_epoch = int(state["completed_epoch"])
            print(
                f"{tokenizer_id} seed={seed}: resuming at epoch {start_epoch + 1}"
            )

    started_at = time.perf_counter()
    verify_first = bool(training["verify_checksums_first_epoch"])
    for epoch in range(start_epoch, int(training["epochs"])):
        train_metrics = train_streaming_epoch(
            model=model,
            text_conditioner=text_conditioner,
            cache=caches["train"],
            mean=mean,
            std=std,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            seed=seed,
            epoch=epoch,
            verify_checksum=verify_first and epoch == 0,
        )
        validation = evaluate_streaming(
            model=model,
            text_conditioner=text_conditioner,
            cache=caches["validation"],
            mean=mean,
            std=std,
            config=config,
            device=device,
            full_sanity_checks=False,
            verify_checksum=verify_first and epoch == 0,
        )
        objective = 0.5 * (
            validation["losses"]["L_visual"]
            + validation["losses"]["L_visual_text"]
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": validation,
                "validation_objective": objective,
            }
        )
        if objective < best_objective:
            best_objective = objective
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        atomic_write_json(output_dir / "history.json", history)
        atomic_torch_save(
            latest_state_path,
            {
                "run_key": identity["run_key"],
                "completed_epoch": epoch + 1,
                "model": recursive_to_cpu(model.state_dict()),
                "optimizer": recursive_to_cpu(optimizer.state_dict()),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "best_state": best_state,
                "best_objective": best_objective,
                "best_epoch": best_epoch,
            },
        )
        print(
            f"{tokenizer_id} seed={seed} epoch={epoch + 1:02d}/"
            f"{training['epochs']} train={train_metrics['loss']:.6f} "
            f"val_visual={validation['losses']['L_visual']:.6f} "
            f"val_text={validation['losses']['L_visual_text']:.6f} "
            f"VTM={validation['scores']['VTM']:.4f} "
            f"LCG={validation['scores']['LCG']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Full-COCO training produced no best checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = output_dir / "predictor.safetensors"
    atomic_save_safetensors(checkpoint_path, best_state)
    test_metrics = evaluate_streaming(
        model=model,
        text_conditioner=text_conditioner,
        cache=caches["test"],
        mean=mean,
        std=std,
        config=config,
        device=device,
        full_sanity_checks=True,
        verify_checksum=True,
    )
    result = {
        "schema_version": 1,
        "protocol": "coco_karpathy_full_streaming_v1",
        "tokenizer_id": tokenizer_id,
        "predictor_seed": seed,
        "run_key": identity["run_key"],
        "completed_at": utc_now(),
        "parameter_count": model.parameter_count(),
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "elapsed_seconds_this_invocation": time.perf_counter() - started_at,
        "record_counts": {
            split_name: cache.record_count for split_name, cache in caches.items()
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "normalization": {
            "fit_split": "train",
            "channel_std_min": float(std_cpu.min().item()),
            "channel_std_max": float(std_cpu.max().item()),
        },
        "test": test_metrics,
    }
    atomic_write_json(output_dir / "result.json", result)
    return result


def write_summary(
    artifact_root: Path,
    results: list[dict[str, Any]],
    *,
    expected_counts: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "protocol": "coco_karpathy_full_streaming_v1",
        "generated_at": utc_now(),
        "record_counts": expected_counts,
        "results": results,
    }
    atomic_write_json(artifact_root / "summary.json", payload)
    lines = [
        "# VTM-LCG Full COCO Karpathy Summary",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Records: train={expected_counts['train']}, "
        f"validation={expected_counts['validation']}, test={expected_counts['test']}",
        "",
        "| Tokenizer | Seed | Best epoch | L_mean | L_visual | L_text | "
        "L_shuffled | L_spatial | VTM | LCG | LCG specific | Caption ✓ | Spatial ✓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for result in results:
        losses = result["test"]["losses"]
        scores = result["test"]["scores"]
        lines.append(
            f"| {result['tokenizer_id']} | {result['predictor_seed']} | "
            f"{result['best_epoch']} | {losses['L_mean']:.6f} | "
            f"{losses['L_visual']:.6f} | {losses['L_visual_text']:.6f} | "
            f"{losses['L_visual_shuffled_text']:.6f} | "
            f"{losses['L_visual_spatial_shuffle']:.6f} | "
            f"{scores['VTM']:.4f} | {scores['LCG']:.4f} | "
            f"{scores['LCG_specific']:.4f} | "
            f"{'✓' if scores['caption_specificity_pass'] else '✗'} | "
            f"{'✓' if scores['spatial_structure_pass'] else '✗'} |"
        )
    lines.append("")
    summary_path = artifact_root / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = summary_path.with_suffix(".md.tmp")
    temporary_path.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary_path, summary_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VTM-LCG on the full official COCO Karpathy splits"
    )
    parser.add_argument("--config", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--tokenizer", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    config, project_root = load_phase1_full_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
        config["evaluation"]["batch_size"] = args.batch_size
    if args.artifact_root is not None:
        config["artifact_root"] = str(
            resolve_project_path(project_root, args.artifact_root)
        )
    if str(config["text"]["mode"]) != "online_frozen":
        raise ValueError("Full COCO requires text.mode=online_frozen")

    summary_payloads: dict[str, dict[str, Any]] = {}
    for split_name, summary_path in config["phase0_summaries"].items():
        path = Path(summary_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing Phase 0 {split_name} summary. Run "
                "scripts/run_phase0_coco_karpathy_full.sh first: "
                f"{path}"
            )
        summary_payloads[split_name] = json.loads(path.read_text(encoding="utf-8"))
    available_sets = [
        {item["tokenizer_id"] for item in summary["tokenizers"]}
        for summary in summary_payloads.values()
    ]
    available = set.intersection(*available_sets)
    if args.all:
        selected_ids = sorted(available)
    else:
        unknown = sorted(set(args.tokenizer) - available)
        if unknown:
            raise ValueError(f"Unknown tokenizer ids: {unknown}")
        selected_ids = list(args.tokenizer)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        print(
            f"full COCO: gpu={properties.name} free={free_bytes / 2**30:.1f}GiB "
            f"total={total_bytes / 2**30:.1f}GiB"
        )
    torch.set_float32_matmul_precision("high")
    text_identity = build_text_identity(config, project_root)
    expected_counts = {
        key: int(value) for key, value in config["expected_counts"].items()
    }
    artifact_root = Path(config["artifact_root"])
    pending: list[
        tuple[
            dict[str, Phase0ShardCache],
            int,
            dict[str, Any],
            Path,
        ]
    ] = []
    results: list[dict[str, Any]] = []

    for tokenizer_id in selected_ids:
        caches = {
            split_name: Phase0ShardCache.from_summary(
                Path(config["phase0_summaries"][split_name]),
                tokenizer_id,
                split_name=split_name,
            )
            for split_name in ("train", "validation", "test")
        }
        validate_karpathy_split_caches(
            caches["train"],
            caches["validation"],
            caches["test"],
        )
        actual_counts = {
            split_name: cache.record_count for split_name, cache in caches.items()
        }
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"Unexpected full Karpathy counts for {tokenizer_id}: "
                f"{actual_counts} != {expected_counts}"
            )
        for seed_value in config["training"]["seeds"]:
            seed = int(seed_value)
            identity = build_full_run_identity(
                tokenizer_id=tokenizer_id,
                caches=caches,
                text_identity=text_identity,
                config=config,
                project_root=project_root,
                seed=seed,
            )
            output_dir = (
                artifact_root
                / "predictors"
                / tokenizer_id
                / f"seed_{seed}"
                / identity["run_key"]
            )
            completed = load_completed_result(output_dir, identity)
            if completed is not None:
                results.append(completed)
            else:
                pending.append((caches, seed, identity, output_dir))

    print(
        f"full COCO: counts={expected_counts} tokenizers={selected_ids} "
        f"pending_runs={len(pending)}"
    )
    if args.preflight_only:
        return 0
    if pending:
        precision = (
            torch_dtype_from_name(config["training"]["precision"])
            if device.type == "cuda"
            else torch.float32
        )
        text_conditioner = FrozenClipTextConditioner(
            Path(config["text"]["checkpoint"]),
            max_length=int(config["text"]["max_length"]),
            device=device,
            dtype=precision,
        )
        for caches, seed, identity, output_dir in pending:
            result = train_full_run(
                caches=caches,
                text_conditioner=text_conditioner,
                config=config,
                device=device,
                seed=seed,
                output_dir=output_dir,
                identity=identity,
            )
            results.append(result)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    results.sort(key=lambda item: (item["tokenizer_id"], item["predictor_seed"]))
    write_summary(artifact_root, results, expected_counts=expected_counts)
    print(f"full COCO: summary={artifact_root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
