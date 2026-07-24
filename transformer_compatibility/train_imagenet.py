#!/usr/bin/env python
"""Train a controlled readout on frozen MetaCLIP ImageNet patch tokens."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


EXPERIMENT_ROOT = Path(__file__).resolve().parent
WORKSPACE = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from extractors import build_tokenizer
from readouts import build_readout, trainable_parameter_count


READOUT_NAMES = ("gap_linear", "gap_mlp", "transformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ImageNet-1k visual-tokenizer Transformer compatibility probe"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs" / "imagenet_mc1_protocol.json",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--readout", required=True, choices=READOUT_NAMES)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--data-root", type=Path, default=WORKSPACE / "data" / "imagenet1k")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPERIMENT_ROOT / "outputs" / "imagenet1k",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Per-forward batch. Gradient accumulation preserves the configured global batch.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float16", "float32"),
        default=None,
        help="Defaults to the protocol config. Changing this is recorded in protocol.json.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device. The full protocol is intended for CUDA; cpu is useful for debugging.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow writing into a non-empty run directory. Existing files are not deleted.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open() as handle:
        config = json.load(handle)
    return config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms(dataset_config: dict[str, Any]):
    image_size = int(dataset_config["image_size"])
    mean = tuple(float(value) for value in dataset_config["mean"])
    std = tuple(float(value) for value in dataset_config["std"])
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                image_size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def build_loaders(
    *,
    data_root: Path,
    dataset_config: dict[str, Any],
    micro_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    train_transform, eval_transform = build_transforms(dataset_config)
    train_dataset = datasets.ImageFolder(data_root / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_root / "val", transform=eval_transform)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise RuntimeError("ImageNet train and val class_to_idx mappings differ")
    expected_classes = int(dataset_config["num_classes"])
    if len(train_dataset.classes) != expected_classes:
        raise RuntimeError(
            f"Expected {expected_classes} ImageNet classes, got {len(train_dataset.classes)}"
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    metadata = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "class_to_idx_sha256": hashlib.sha256(
            json.dumps(train_dataset.class_to_idx, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    return train_loader, val_loader, metadata


def cycle_loader(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


def build_optimizer(
    module: nn.Module,
    training_config: dict[str, Any],
) -> torch.optim.Optimizer:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias") or "cls_token" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    parameter_groups = [
        {"params": decay, "weight_decay": float(training_config["weight_decay"])},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=float(training_config["learning_rate"]),
        betas=tuple(float(value) for value in training_config["betas"]),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: dict[str, Any],
) -> torch.optim.lr_scheduler.LambdaLR:
    max_updates = int(training_config["max_updates"])
    warmup_updates = int(training_config["warmup_updates"])
    min_lr_ratio = float(training_config["min_lr_ratio"])

    def multiplier(step: int) -> float:
        if warmup_updates > 0 and step < warmup_updates:
            return float(step + 1) / float(warmup_updates)
        progress = (step - warmup_updates) / max(1, max_updates - warmup_updates)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def precision_context(device: torch.device, precision: str):
    if precision == "float32" or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "float16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


@torch.no_grad()
def evaluate(
    *,
    tokenizer: nn.Module,
    readout: nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    criterion: nn.Module,
) -> dict[str, float | int]:
    tokenizer.eval()
    readout.eval()
    loss_sum = 0.0
    correct1 = 0
    correct5 = 0
    sample_count = 0
    started = time.perf_counter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with precision_context(device, precision):
            tokens = tokenizer(images)
            logits = readout(tokens)
            loss = criterion(logits, targets)
        batch_size = targets.shape[0]
        predictions = logits.topk(k=5, dim=1).indices
        correct = predictions.eq(targets[:, None])
        correct1 += int(correct[:, :1].sum().item())
        correct5 += int(correct.sum().item())
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size

    return {
        "samples": sample_count,
        "loss": loss_sum / sample_count,
        "top1": 100.0 * correct1 / sample_count,
        "top5": 100.0 * correct5 / sample_count,
        "elapsed_seconds": time.perf_counter() - started,
    }


def compute_log_aulc(evaluations: list[dict[str, Any]], metric: str) -> float:
    ordered = sorted(evaluations, key=lambda item: int(item["step"]))
    max_step = int(ordered[-1]["step"])
    if max_step <= 0:
        return float(ordered[-1][metric])
    area = 0.0
    for left, right in zip(ordered, ordered[1:]):
        x0 = math.log(int(left["step"]) + 1)
        x1 = math.log(int(right["step"]) + 1)
        area += 0.5 * (float(left[metric]) + float(right[metric])) * (x1 - x0)
    return area / math.log(max_step + 1)


def save_checkpoint(
    *,
    path: Path,
    step: int,
    readout: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler,
    protocol_fingerprint: str,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "readout": readout.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "protocol_fingerprint": protocol_fingerprint,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training_config = config["training"]
    if args.model not in config["models"]:
        choices = ", ".join(sorted(config["models"]))
        raise ValueError(f"Unknown model '{args.model}'. Configured models: {choices}")
    model_config = config["models"][args.model]
    if args.seed not in [int(value) for value in training_config["seeds"]]:
        raise ValueError(f"Seed {args.seed} is outside the configured protocol seeds")

    data_root = args.data_root.expanduser().resolve()
    if not (data_root / "train").is_dir() or not (data_root / "val").is_dir():
        raise FileNotFoundError(f"Expected ImageNet train/val directories under {data_root}")

    micro_batch_size = (
        int(args.micro_batch_size)
        if args.micro_batch_size is not None
        else int(training_config["default_micro_batch_size"])
    )
    eval_batch_size = (
        int(args.eval_batch_size)
        if args.eval_batch_size is not None
        else int(training_config["default_eval_batch_size"])
    )
    global_batch_size = int(training_config["global_batch_size"])
    if micro_batch_size <= 0 or global_batch_size % micro_batch_size != 0:
        raise ValueError(
            f"micro batch {micro_batch_size} must divide global batch {global_batch_size}"
        )
    accumulation_steps = global_batch_size // micro_batch_size
    precision = args.precision or training_config["amp_dtype"]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    seed_everything(args.seed)

    run_dir = (
        args.output_root.expanduser().resolve()
        / args.model
        / args.readout
        / f"seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_files = list(run_dir.iterdir())
    if existing_files and not args.allow_existing_output:
        names = ", ".join(sorted(path.name for path in existing_files)[:5])
        raise FileExistsError(
            f"Run directory is not empty: {run_dir} ({names}). "
            "Choose another --output-root or pass --allow-existing-output."
        )

    tokenizer = build_tokenizer(model_config, config["surface"], WORKSPACE)
    tokenizer.to(device).eval()
    readout, readout_metadata = build_readout(
        name=args.readout,
        input_dim=int(model_config["input_dim"]),
        num_classes=int(config["dataset"]["num_classes"]),
        readout_configs=config["readouts"],
    )
    readout.to(device)

    train_loader, val_loader, dataset_metadata = build_loaders(
        data_root=data_root,
        dataset_config=config["dataset"],
        micro_batch_size=micro_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    optimizer = build_optimizer(readout, training_config)
    scheduler = build_scheduler(optimizer, training_config)
    scaler = build_grad_scaler(device, precision)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training_config["label_smoothing"])
    )

    effective_protocol = {
        "protocol_version": config["protocol_version"],
        "started_at": utc_now(),
        "model": args.model,
        "readout": args.readout,
        "seed": args.seed,
        "dataset": {
            **config["dataset"],
            **dataset_metadata,
            "root": str(data_root),
        },
        "surface": {
            **config["surface"],
            "input_dim": tokenizer.surface.input_dim,
            "token_count": tokenizer.surface.token_count,
            "grid_shape": list(tokenizer.surface.grid_shape),
            "checkpoint": tokenizer.surface.checkpoint,
            "checkpoint_size_bytes": Path(tokenizer.surface.checkpoint).stat().st_size,
            "timm_model": tokenizer.surface.timm_model,
        },
        "readout_config": readout_metadata,
        "training": {
            **training_config,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "eval_batch_size": eval_batch_size,
            "num_workers": args.num_workers,
            "precision": precision,
            "device": str(device),
            "trainable_parameters": trainable_parameter_count(readout),
            "frozen_tokenizer_parameters": sum(
                parameter.numel() for parameter in tokenizer.parameters()
            ),
        },
    }
    protocol_text = json.dumps(effective_protocol, sort_keys=True)
    protocol_fingerprint = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()
    effective_protocol["protocol_fingerprint"] = protocol_fingerprint
    atomic_write_json(run_dir / "protocol.json", effective_protocol)

    print(json.dumps(
        {
            "run_dir": str(run_dir),
            "model": args.model,
            "readout": args.readout,
            "seed": args.seed,
            "surface_shape": [
                tokenizer.surface.token_count,
                tokenizer.surface.input_dim,
            ],
            "trainable_parameters": trainable_parameter_count(readout),
            "global_batch_size": global_batch_size,
            "micro_batch_size": micro_batch_size,
            "accumulation_steps": accumulation_steps,
        },
        indent=2,
    ))

    metrics_path = run_dir / "metrics.jsonl"
    eval_steps = set(int(value) for value in training_config["eval_steps"])
    max_updates = int(training_config["max_updates"])
    if max_updates not in eval_steps or 0 not in eval_steps:
        raise ValueError("eval_steps must contain both 0 and max_updates")

    evaluations: list[dict[str, Any]] = []

    def run_evaluation(step: int) -> None:
        result = evaluate(
            tokenizer=tokenizer,
            readout=readout,
            loader=val_loader,
            device=device,
            precision=precision,
            criterion=criterion,
        )
        record = {
            "timestamp": utc_now(),
            "split": "val",
            "step": step,
            "model": args.model,
            "readout": args.readout,
            "seed": args.seed,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **result,
        }
        evaluations.append(record)
        append_jsonl(metrics_path, record)
        print(json.dumps(record, sort_keys=True))

    run_evaluation(0)
    train_iterator = cycle_loader(train_loader)
    readout.train()
    optimizer.zero_grad(set_to_none=True)
    log_started = time.perf_counter()
    running_loss = 0.0

    for step in range(1, max_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        for _ in range(accumulation_steps):
            images, targets = next(train_iterator)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.no_grad():
                with precision_context(device, precision):
                    tokens = tokenizer(images)
            with precision_context(device, precision):
                logits = readout(tokens)
                loss = criterion(logits, targets)
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            update_loss += float(loss.detach().item()) / accumulation_steps

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            readout.parameters(),
            max_norm=float(training_config["gradient_clip_norm"]),
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += update_loss

        if step % 50 == 0:
            elapsed = time.perf_counter() - log_started
            train_record = {
                "timestamp": utc_now(),
                "split": "train",
                "step": step,
                "model": args.model,
                "readout": args.readout,
                "seed": args.seed,
                "loss_50_update_mean": running_loss / 50.0,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "gradient_norm": float(gradient_norm),
                "updates_per_second": 50.0 / elapsed,
            }
            append_jsonl(metrics_path, train_record)
            print(json.dumps(train_record, sort_keys=True))
            running_loss = 0.0
            log_started = time.perf_counter()

        if step in eval_steps:
            run_evaluation(step)
            save_checkpoint(
                path=run_dir / f"checkpoint_step{step}.pt",
                step=step,
                readout=readout,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                protocol_fingerprint=protocol_fingerprint,
            )
            readout.train()

    summary = {
        "protocol_version": config["protocol_version"],
        "protocol_fingerprint": protocol_fingerprint,
        "completed_at": utc_now(),
        "model": args.model,
        "readout": args.readout,
        "seed": args.seed,
        "trainable_parameters": trainable_parameter_count(readout),
        "evaluations": evaluations,
        "top1_aulc_log_updates": compute_log_aulc(evaluations, "top1"),
        "top5_aulc_log_updates": compute_log_aulc(evaluations, "top5"),
        "final_top1": float(evaluations[-1]["top1"]),
        "final_top5": float(evaluations[-1]["top5"]),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
