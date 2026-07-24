from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from vtm_lcg.cache.text import CaptionEmbeddingStore, ensure_text_cache
from vtm_lcg.config import (
    load_phase1_config,
    resolve_project_path,
    torch_dtype_from_name,
)
from vtm_lcg.eval import compute_vtm_lcg_scores
from vtm_lcg.eval.sanity_checks import (
    caption_keep_mask,
    make_deterministic_mask,
    select_caption_ids,
    shuffled_caption_id_map,
    spatially_shuffle_visible_tokens,
)
from vtm_lcg.models import MaskedVisualPredictor
from vtm_lcg.train.data import (
    IndexedVisualDataset,
    fit_channel_standardization,
    load_phase0_summary,
    load_phase0_visual_cache,
    make_split_indices,
)
from vtm_lcg.utils import (
    atomic_write_json,
    code_fingerprint,
    sha256_file,
    sha256_json,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    normalized = name.lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def autocast_context(device: torch.device, precision_name: str):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch_dtype_from_name(precision_name)
    if dtype not in (torch.float16, torch.bfloat16):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def atomic_save_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        save_file(
            {
                key: value.detach().to("cpu").contiguous()
                for key, value in tensors.items()
            },
            str(temporary_path),
        )
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def make_loader(
    dataset: IndexedVisualDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def build_model(config: dict[str, Any]) -> MaskedVisualPredictor:
    predictor = config["predictor"]
    return MaskedVisualPredictor(
        visual_input_dim=int(predictor["visual_input_dim"]),
        text_input_dim=int(predictor["text_input_dim"]),
        model_dim=int(predictor["model_dim"]),
        depth=int(predictor["depth"]),
        num_heads=int(predictor["num_heads"]),
        mlp_ratio=int(predictor["mlp_ratio"]),
        dropout=float(predictor["dropout"]),
        grid_shape=(16, 16),
    )


def normalize_visual(
    raw_values: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    values = raw_values.to(device=device, dtype=torch.float32, non_blocking=True)
    return (values - mean) / std


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        raise ValueError("masked MSE requires at least one target token")
    difference = prediction[mask].float() - target[mask].float()
    return difference.square().mean()


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> LambdaLR:
    warmup_steps = int(round(total_steps * warmup_ratio))

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / warmup_steps
        remaining = max(total_steps - warmup_steps, 1)
        progress = min(max(step - warmup_steps, 0) / remaining, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)


def train_epoch(
    *,
    model: MaskedVisualPredictor,
    loader: DataLoader,
    records: list[dict[str, Any]],
    text_store: CaptionEmbeddingStore,
    mean: torch.Tensor,
    std: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    epoch: int,
) -> dict[str, float]:
    model.train()
    predictor = config["predictor"]
    training = config["training"]
    evaluation = config["evaluation"]
    squared_error_sum = 0.0
    target_element_count = 0
    kept_caption_count = 0
    example_count = 0

    for raw_values, record_indices in loader:
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
        caption_ids = select_caption_ids(
            records,
            record_indices_list,
            seed=int(evaluation["caption_seed"]) + seed,
            epoch=epoch,
            fixed_first=False,
        )
        text_embeddings, text_attention = text_store.get(caption_ids)
        text_embeddings = text_embeddings.to(device, non_blocking=True)
        text_attention = text_attention.to(device, non_blocking=True)
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
        squared_error_sum += float(loss.detach().item()) * element_count
        target_element_count += element_count
        kept_caption_count += int(keep_caption.sum().item())
        example_count += int(visual.shape[0])

    return {
        "loss": squared_error_sum / target_element_count,
        "caption_keep_fraction": kept_caption_count / example_count,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


@torch.inference_mode()
def evaluate_predictor(
    *,
    model: MaskedVisualPredictor,
    loader: DataLoader,
    records: list[dict[str, Any]],
    text_store: CaptionEmbeddingStore,
    mean: torch.Tensor,
    std: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    full_sanity_checks: bool,
) -> dict[str, Any]:
    model.eval()
    predictor = config["predictor"]
    training = config["training"]
    evaluation = config["evaluation"]
    modes = ["L_mean", "L_visual", "L_visual_text"]
    if full_sanity_checks:
        modes.extend(
            [
                "L_visual_shuffled_text",
                "L_visual_spatial_shuffle",
                "L_no_visible",
            ]
        )
    squared_error_sums = {mode: 0.0 for mode in modes}
    target_element_count = 0
    evaluation_indices = loader.dataset.indices
    shuffled_map = (
        shuffled_caption_id_map(records, evaluation_indices)
        if full_sanity_checks
        else {}
    )

    for raw_values, record_indices in loader:
        visual = normalize_visual(raw_values, mean, std, device)
        record_indices_list = [int(value) for value in record_indices.tolist()]
        masked_positions = make_deterministic_mask(
            record_indices_list,
            token_count=visual.shape[1],
            mask_ratio=float(predictor["mask_ratio"]),
            seed=int(evaluation["mask_seed"]),
            epoch=0,
            device=device,
        )
        targets = visual[masked_positions].float()
        element_count = targets.numel()
        squared_error_sums["L_mean"] += float(targets.square().sum().item())

        with autocast_context(device, str(training["precision"])):
            visual_prediction = model(visual, masked_positions)
        squared_error_sums["L_visual"] += float(
            (visual_prediction[masked_positions].float() - targets)
            .square()
            .sum()
            .item()
        )

        true_caption_ids = select_caption_ids(
            records,
            record_indices_list,
            seed=int(evaluation["caption_seed"]),
            epoch=0,
            fixed_first=True,
        )
        text_embeddings, text_attention = text_store.get(true_caption_ids)
        text_embeddings = text_embeddings.to(device, non_blocking=True)
        text_attention = text_attention.to(device, non_blocking=True)
        with autocast_context(device, str(training["precision"])):
            text_prediction = model(
                visual,
                masked_positions,
                text_embeddings=text_embeddings,
                text_attention_mask=text_attention,
            )
        squared_error_sums["L_visual_text"] += float(
            (text_prediction[masked_positions].float() - targets)
            .square()
            .sum()
            .item()
        )

        if full_sanity_checks:
            shuffled_caption_ids = [
                shuffled_map[index] for index in record_indices_list
            ]
            shuffled_embeddings, shuffled_attention = text_store.get(
                shuffled_caption_ids
            )
            shuffled_embeddings = shuffled_embeddings.to(device, non_blocking=True)
            shuffled_attention = shuffled_attention.to(device, non_blocking=True)
            with autocast_context(device, str(training["precision"])):
                shuffled_prediction = model(
                    visual,
                    masked_positions,
                    text_embeddings=shuffled_embeddings,
                    text_attention_mask=shuffled_attention,
                )
            squared_error_sums["L_visual_shuffled_text"] += float(
                (shuffled_prediction[masked_positions].float() - targets)
                .square()
                .sum()
                .item()
            )

            spatially_shuffled = spatially_shuffle_visible_tokens(
                visual,
                masked_positions,
                record_indices_list,
                seed=int(evaluation["shuffle_seed"]),
            )
            with autocast_context(device, str(training["precision"])):
                spatial_prediction = model(spatially_shuffled, masked_positions)
                no_visible_prediction = model(
                    visual,
                    masked_positions,
                    hide_all_visual=True,
                )
            squared_error_sums["L_visual_spatial_shuffle"] += float(
                (spatial_prediction[masked_positions].float() - targets)
                .square()
                .sum()
                .item()
            )
            squared_error_sums["L_no_visible"] += float(
                (no_visible_prediction[masked_positions].float() - targets)
                .square()
                .sum()
                .item()
            )
        target_element_count += element_count

    losses = {
        mode: squared_error_sums[mode] / target_element_count for mode in modes
    }
    return {
        "losses": losses,
        "scores": compute_vtm_lcg_scores(losses),
        "target_element_count": target_element_count,
    }


def train_one_run(
    *,
    tokenizer_id: str,
    values: torch.Tensor,
    records: list[dict[str, Any]],
    splits: dict[str, list[int]],
    text_store: CaptionEmbeddingStore,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    output_dir: Path,
    run_identity: dict[str, Any],
) -> dict[str, Any]:
    seed_everything(seed)
    training = config["training"]
    mean_cpu, std_cpu = fit_channel_standardization(
        values,
        splits["train"],
        epsilon=float(training["stats_epsilon"]),
    )
    if not bool(torch.isfinite(mean_cpu).all() and torch.isfinite(std_cpu).all()):
        raise RuntimeError("Train-only visual normalization is non-finite")
    if not bool((std_cpu > 1.0e-6).all()):
        raise RuntimeError("Train-only visual normalization contains collapsed channels")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_safetensors(
        output_dir / "normalization.safetensors",
        {"mean": mean_cpu, "std": std_cpu},
    )
    atomic_write_json(output_dir / "run_identity.json", run_identity)

    mean = mean_cpu.to(device).view(1, 1, -1)
    std = std_cpu.to(device).view(1, 1, -1)
    train_dataset = IndexedVisualDataset(values, splits["train"])
    validation_dataset = IndexedVisualDataset(values, splits["validation"])
    test_dataset = IndexedVisualDataset(values, splits["test"])
    validation_loader = make_loader(
        validation_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        num_workers=int(training["num_workers"]),
        shuffle=False,
        seed=seed,
        device=device,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        num_workers=int(training["num_workers"]),
        shuffle=False,
        seed=seed,
        device=device,
    )

    model = build_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    steps_per_epoch = math.ceil(len(train_dataset) / int(training["batch_size"]))
    total_steps = int(training["epochs"]) * steps_per_epoch
    scheduler = make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=float(training["warmup_ratio"]),
    )
    history: list[dict[str, Any]] = []
    best_objective = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    started_at = time.perf_counter()

    for epoch in range(int(training["epochs"])):
        train_loader = make_loader(
            train_dataset,
            batch_size=int(training["batch_size"]),
            num_workers=int(training["num_workers"]),
            shuffle=True,
            seed=seed + epoch,
            device=device,
        )
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            records=records,
            text_store=text_store,
            mean=mean,
            std=std,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            seed=seed,
            epoch=epoch,
        )
        validation = evaluate_predictor(
            model=model,
            loader=validation_loader,
            records=records,
            text_store=text_store,
            mean=mean,
            std=std,
            config=config,
            device=device,
            full_sanity_checks=False,
        )
        validation_objective = 0.5 * (
            validation["losses"]["L_visual"]
            + validation["losses"]["L_visual_text"]
        )
        epoch_result = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation,
            "validation_objective": validation_objective,
        }
        history.append(epoch_result)
        if validation_objective < best_objective:
            best_objective = validation_objective
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        atomic_write_json(output_dir / "history.json", history)
        print(
            f"{tokenizer_id} seed={seed} epoch={epoch + 1:02d}/"
            f"{training['epochs']} train={train_metrics['loss']:.6f} "
            f"val_visual={validation['losses']['L_visual']:.6f} "
            f"val_text={validation['losses']['L_visual_text']:.6f} "
            f"VTM={validation['scores']['VTM']:.4f} "
            f"LCG={validation['scores']['LCG']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = output_dir / "predictor.safetensors"
    atomic_save_safetensors(checkpoint_path, best_state)
    test_metrics = evaluate_predictor(
        model=model,
        loader=test_loader,
        records=records,
        text_store=text_store,
        mean=mean,
        std=std,
        config=config,
        device=device,
        full_sanity_checks=True,
    )
    result = {
        "schema_version": 1,
        "tokenizer_id": tokenizer_id,
        "predictor_seed": seed,
        "run_key": run_identity["run_key"],
        "completed_at": utc_now(),
        "parameter_count": model.parameter_count(),
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "elapsed_seconds": time.perf_counter() - started_at,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "normalization": {
            "fit_split": "train",
            "train_images": len(splits["train"]),
            "epsilon": float(training["stats_epsilon"]),
            "channel_std_min": float(std_cpu.min().item()),
            "channel_std_max": float(std_cpu.max().item()),
        },
        "test": test_metrics,
    }
    atomic_write_json(output_dir / "result.json", result)
    return result


def build_run_identity(
    *,
    tokenizer_summary: dict[str, Any],
    text_identity: dict[str, Any],
    splits: dict[str, list[int]],
    config: dict[str, Any],
    project_root: Path,
    seed: int,
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "tokenizer_id": tokenizer_summary["tokenizer_id"],
        "phase0_cache_key": tokenizer_summary["cache_key"],
        "text_cache_key": text_identity["cache_key"],
        "split_sha256": sha256_json(splits),
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
    run_identity: dict[str, Any],
) -> dict[str, Any] | None:
    identity_path = output_dir / "run_identity.json"
    result_path = output_dir / "result.json"
    checkpoint_path = output_dir / "predictor.safetensors"
    if not (identity_path.is_file() and result_path.is_file() and checkpoint_path.is_file()):
        return None
    if json.loads(identity_path.read_text(encoding="utf-8")) != run_identity:
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        return None
    print(
        f"{result['tokenizer_id']} seed={result['predictor_seed']}: "
        "completed predictor result reused"
    )
    return result


def write_phase1_summary(
    artifact_root: Path,
    results: list[dict[str, Any]],
    *,
    dataset_fingerprint: str,
) -> None:
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "dataset_fingerprint": dataset_fingerprint,
        "results": results,
    }
    atomic_write_json(artifact_root / "summary.json", payload)
    lines = [
        "# VTM-LCG Phase 1 Summary",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Dataset fingerprint: `{dataset_fingerprint}`",
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
        description="Train and evaluate the VTM-LCG Phase 1 masked predictor"
    )
    parser.add_argument("--config", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--tokenizer", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    config, project_root = load_phase1_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.artifact_root is not None:
        config["artifact_root"] = str(
            resolve_project_path(project_root, args.artifact_root)
        )
    if int(config["training"]["epochs"]) <= 0:
        raise ValueError("epochs must be positive")

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        print(
            f"phase1: gpu={properties.name} free={free_bytes / 2**30:.1f}GiB "
            f"total={total_bytes / 2**30:.1f}GiB"
        )
    torch.set_float32_matmul_precision("high")

    phase0_summary = load_phase0_summary(Path(config["phase0_summary"]))
    available = {
        item["tokenizer_id"]: item for item in phase0_summary["tokenizers"]
    }
    if args.all:
        selected_ids = list(available)
    else:
        unknown = sorted(set(args.tokenizer) - set(available))
        if unknown:
            raise ValueError(
                f"Unknown tokenizer ids {unknown}; choices={sorted(available)}"
            )
        selected_ids = list(args.tokenizer)

    reference_cache_dir = Path(available[selected_ids[0]]["cache_dir"])
    records_payload = json.loads(
        (reference_cache_dir / "records.json").read_text(encoding="utf-8")
    )
    records = records_payload["records"]
    dataset_fingerprint = records_payload["dataset"]["dataset_fingerprint"]
    split_config = config["split"]
    splits = make_split_indices(
        len(records),
        train_count=int(split_config["train"]),
        validation_count=int(split_config["validation"]),
        test_count=int(split_config["test"]),
        seed=int(split_config["seed"]),
    )
    artifact_root = Path(config["artifact_root"])
    split_payload = {
        "schema_version": 1,
        "dataset_fingerprint": dataset_fingerprint,
        "seed": int(split_config["seed"]),
        "indices": splits,
        "image_ids": {
            split_name: [records[index]["image_id"] for index in indices]
            for split_name, indices in splits.items()
        },
    }
    split_path = artifact_root / "split.json"
    if split_path.is_file():
        if json.loads(split_path.read_text(encoding="utf-8")) != split_payload:
            raise RuntimeError(f"Existing Phase 1 split changed: {split_path}")
    else:
        atomic_write_json(split_path, split_payload)

    print(
        f"phase1: records={len(records)} split="
        f"{len(splits['train'])}/{len(splits['validation'])}/{len(splits['test'])} "
        f"tokenizers={selected_ids} seeds={config['training']['seeds']}"
    )
    if args.preflight_only:
        return 0

    text_store, text_identity, text_cache_path = ensure_text_cache(
        records=records,
        dataset_fingerprint=dataset_fingerprint,
        text_config=config["text"],
        artifact_root=artifact_root,
        project_root=project_root,
        device=device,
        precision_name=str(config["training"]["precision"]),
    )
    print(
        f"phase1: shared text cache={text_cache_path} "
        f"shape={tuple(text_store.embeddings.shape)}"
    )

    results: list[dict[str, Any]] = []
    for tokenizer_id in selected_ids:
        tokenizer_summary = available[tokenizer_id]
        run_specs: list[tuple[int, dict[str, Any], Path]] = []
        for seed_value in config["training"]["seeds"]:
            seed = int(seed_value)
            run_identity = build_run_identity(
                tokenizer_summary=tokenizer_summary,
                text_identity=text_identity,
                splits=splits,
                config=config,
                project_root=project_root,
                seed=seed,
            )
            output_dir = (
                artifact_root
                / "predictors"
                / tokenizer_id
                / f"seed_{seed}"
                / run_identity["run_key"]
            )
            completed = load_completed_result(output_dir, run_identity)
            if completed is not None:
                results.append(completed)
            else:
                run_specs.append((seed, run_identity, output_dir))
        if not run_specs:
            continue

        print(f"{tokenizer_id}: loading and verifying the Phase 0 visual cache")
        visual_cache = load_phase0_visual_cache(phase0_summary, tokenizer_id)
        if visual_cache.dataset_fingerprint != dataset_fingerprint:
            raise RuntimeError("Phase 0 tokenizer caches use different datasets")
        if visual_cache.records != records:
            raise RuntimeError("Phase 0 tokenizer records differ")
        for seed, run_identity, output_dir in run_specs:
            result = train_one_run(
                tokenizer_id=tokenizer_id,
                values=visual_cache.values,
                records=records,
                splits=splits,
                text_store=text_store,
                config=config,
                device=device,
                seed=seed,
                output_dir=output_dir,
                run_identity=run_identity,
            )
            results.append(result)
        del visual_cache
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results.sort(key=lambda item: (item["tokenizer_id"], item["predictor_seed"]))
    write_phase1_summary(
        artifact_root,
        results,
        dataset_fingerprint=dataset_fingerprint,
    )
    print(f"phase1: summary={artifact_root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

