from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from safetensors.torch import load_file
from torch.optim import AdamW
from tqdm import tqdm

from vtm_lcg.config import load_cvrvtm_config
from vtm_lcg.eval.sanity_checks import spatially_shuffle_visible_tokens
from vtm_lcg.train.data import make_split_indices
from vtm_lcg.train.train_predictor import (
    atomic_save_safetensors,
    autocast_context,
    make_scheduler,
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

from .cache import CrossViewShardDescriptor, validate_cross_view_shard
from .model import CrossViewResidualPredictor
from .protocol import (
    compute_cvrvtm_scores,
    cross_view_loss_sums,
    make_deterministic_block_mask,
    residualize_cross_view,
)


@dataclass
class CrossViewShardCache:
    split_name: str
    tokenizer_id: str
    cache_key: str
    cache_dir: Path
    manifest: dict[str, Any]
    records: list[dict[str, Any]]
    stats: dict[str, Any]
    descriptors: list[CrossViewShardDescriptor]

    @classmethod
    def from_summary(
        cls,
        summary_path: Path,
        tokenizer_id: str,
        *,
        split_name: str,
    ) -> "CrossViewShardCache":
        summary_path = summary_path.resolve()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("protocol") != "cross_view_aligned_cache_v1":
            raise RuntimeError(f"Not a CV-RVTM cache summary: {summary_path}")
        if not summary.get("all_acceptance_checks_passed"):
            raise RuntimeError(f"CV-RVTM cache failed acceptance: {summary_path}")
        results = {item["tokenizer_id"]: item for item in summary["tokenizers"]}
        if tokenizer_id not in results:
            raise KeyError(f"{tokenizer_id!r} is absent from {summary_path}")
        result = results[tokenizer_id]
        cache_dir = Path(result["cache_dir"])
        manifest = json.loads(
            (cache_dir / "manifest.json").read_text(encoding="utf-8")
        )
        records_payload = json.loads(
            (cache_dir / "records.json").read_text(encoding="utf-8")
        )
        stats = json.loads((cache_dir / "stats.json").read_text(encoding="utf-8"))
        descriptors = sorted(
            (
                CrossViewShardDescriptor.from_dict(payload)
                for payload in manifest["shards"]
            ),
            key=lambda item: item.index,
        )
        if not manifest.get("complete"):
            raise RuntimeError(f"Incomplete CV-RVTM cache: {cache_dir}")
        records = records_payload["records"]
        if sum(len(item.image_ids) for item in descriptors) != len(records):
            raise RuntimeError(f"CV-RVTM shard/record mismatch: {cache_dir}")
        return cls(
            split_name=split_name,
            tokenizer_id=tokenizer_id,
            cache_key=str(result["cache_key"]),
            cache_dir=cache_dir,
            manifest=manifest,
            records=records,
            stats=stats,
            descriptors=descriptors,
        )

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def shard_size(self) -> int:
        return int(self.manifest["shard_size"])

    def batch_count(
        self,
        batch_size: int,
        indices: Sequence[int] | None,
    ) -> int:
        if indices is None:
            return sum(
                math.ceil(len(descriptor.image_ids) / batch_size)
                for descriptor in self.descriptors
            )
        selected = {int(index) for index in indices}
        total = 0
        for descriptor in self.descriptors:
            start = descriptor.index * self.shard_size
            stop = start + len(descriptor.image_ids)
            count = sum(index in selected for index in range(start, stop))
            if count:
                total += math.ceil(count / batch_size)
        return total

    def normalization(
        self,
        *,
        indices: Sequence[int] | None,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if indices is None:
            mean = torch.tensor(self.stats["channel_mean"], dtype=torch.float32)
            std = torch.tensor(self.stats["channel_std"], dtype=torch.float32)
            return mean, std
        channel_sum = torch.zeros(
            int(self.manifest["hidden_dim"]),
            dtype=torch.float64,
        )
        channel_square_sum = torch.zeros_like(channel_sum)
        observation_count = 0
        for view_a, view_b, _ in self.iter_batches(
            batch_size=self.shard_size,
            indices=indices,
            shuffle=False,
            seed=0,
            verify_checksum=False,
        ):
            values = torch.cat((view_a, view_b), dim=0).to(torch.float64)
            channel_sum += values.sum(dim=(0, 1))
            channel_square_sum += values.square().sum(dim=(0, 1))
            observation_count += int(values.shape[0] * values.shape[1])
        mean64 = channel_sum / observation_count
        variance = (
            channel_square_sum / observation_count - mean64.square()
        ).clamp_min(0)
        return mean64.float(), (variance + epsilon).sqrt().float()

    def iter_batches(
        self,
        *,
        batch_size: int,
        indices: Sequence[int] | None,
        shuffle: bool,
        seed: int,
        verify_checksum: bool,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if batch_size <= 0 or batch_size > self.shard_size:
            raise ValueError(
                f"batch_size must be in [1,{self.shard_size}] for shard streaming"
            )
        selected = (
            None if indices is None else {int(index) for index in indices}
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        descriptor_order = (
            torch.randperm(len(self.descriptors), generator=generator).tolist()
            if shuffle
            else list(range(len(self.descriptors)))
        )
        yielded = 0
        for descriptor_position in descriptor_order:
            descriptor = self.descriptors[descriptor_position]
            path = self.cache_dir / "shards" / descriptor.filename
            if verify_checksum:
                view_a, view_b, image_ids = validate_cross_view_shard(
                    path,
                    descriptor,
                    verify_checksum=True,
                )
            else:
                tensors = load_file(str(path), device="cpu")
                view_a = tensors["view_a"]
                view_b = tensors["view_b"]
                image_ids = tensors["image_ids"]
            offset = descriptor.index * self.shard_size
            record_indices = torch.arange(
                offset,
                offset + view_a.shape[0],
                dtype=torch.int64,
            )
            expected_ids = [
                int(self.records[index]["image_id"])
                for index in record_indices.tolist()
            ]
            if [int(value) for value in image_ids.tolist()] != expected_ids:
                raise RuntimeError(f"CV-RVTM record order mismatch: {path}")
            if selected is not None:
                keep = torch.tensor(
                    [int(index) in selected for index in record_indices.tolist()],
                    dtype=torch.bool,
                )
                view_a = view_a[keep]
                view_b = view_b[keep]
                record_indices = record_indices[keep]
            if view_a.shape[0] == 0:
                continue
            if shuffle:
                order = torch.randperm(view_a.shape[0], generator=generator)
                view_a = view_a[order]
                view_b = view_b[order]
                record_indices = record_indices[order]
            for start in range(0, view_a.shape[0], batch_size):
                stop = min(start + batch_size, view_a.shape[0])
                yielded += stop - start
                yield (
                    view_a[start:stop],
                    view_b[start:stop],
                    record_indices[start:stop],
                )
        expected_count = self.record_count if selected is None else len(selected)
        if yielded != expected_count:
            raise RuntimeError(
                f"CV-RVTM yielded {yielded} records, expected {expected_count}"
            )


def _normalize(
    values: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    values = values.to(device=device, dtype=torch.float32, non_blocking=True)
    return (values - mean) / std


def _build_model(config: dict[str, Any]) -> CrossViewResidualPredictor:
    predictor = config["predictor"]
    return CrossViewResidualPredictor(
        visual_input_dim=int(predictor["visual_input_dim"]),
        model_dim=int(predictor["model_dim"]),
        depth=int(predictor["depth"]),
        num_heads=int(predictor["num_heads"]),
        mlp_ratio=int(predictor["mlp_ratio"]),
        dropout=float(predictor["dropout"]),
        grid_shape=tuple(int(value) for value in predictor["grid_shape"]),
    )


def _make_mask(
    record_indices: torch.Tensor,
    config: dict[str, Any],
    *,
    seed: int,
    epoch: int,
    device: torch.device,
) -> torch.Tensor:
    predictor = config["predictor"]
    return make_deterministic_block_mask(
        record_indices,
        grid_shape=tuple(int(value) for value in predictor["grid_shape"]),
        block_shape=tuple(int(value) for value in predictor["block_shape"]),
        mask_ratio=float(predictor["mask_ratio"]),
        seed=int(config["evaluation"]["mask_seed"]) + seed,
        epoch=epoch,
        device=device,
    )


def _symmetric_directions(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    mask: torch.Tensor,
    record_indices: torch.Tensor,
    *,
    symmetric: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not symmetric:
        return view_a, view_b, mask, record_indices
    return (
        torch.cat((view_a, view_b), dim=0),
        torch.cat((view_b, view_a), dim=0),
        torch.cat((mask, mask), dim=0),
        torch.cat((record_indices, record_indices), dim=0),
    )


def train_epoch(
    *,
    model: CrossViewResidualPredictor,
    cache: CrossViewShardCache,
    indices: Sequence[int] | None,
    mean: torch.Tensor,
    std: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    epoch: int,
) -> dict[str, float]:
    model.train()
    training = config["training"]
    predictor = config["predictor"]
    error_sum = 0.0
    element_count_sum = 0
    batches = cache.iter_batches(
        batch_size=int(training["batch_size"]),
        indices=indices,
        shuffle=True,
        seed=seed + epoch,
        verify_checksum=False,
    )
    progress = tqdm(
        batches,
        total=cache.batch_count(int(training["batch_size"]), indices),
        desc=f"CV-RVTM {cache.tokenizer_id} epoch {epoch + 1}",
        unit="batch",
        dynamic_ncols=True,
    )
    for raw_a, raw_b, record_indices in progress:
        view_a = _normalize(raw_a, mean, std, device)
        view_b = _normalize(raw_b, mean, std, device)
        mask = _make_mask(
            record_indices,
            config,
            seed=seed,
            epoch=epoch,
            device=device,
        )
        record_indices = record_indices.to(device)
        source, target, mask, _ = _symmetric_directions(
            view_a,
            view_b,
            mask,
            record_indices,
            symmetric=bool(predictor["symmetric"]),
        )
        source_residual, target_residual, _ = residualize_cross_view(
            source,
            target,
            mask,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, str(training["precision"])):
            prediction = model(source_residual, mask)
            difference = (
                prediction[mask].float() - target_residual[mask].float()
            )
            loss = difference.square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        element_count = difference.numel()
        error_sum += float(loss.detach().item()) * element_count
        element_count_sum += element_count
        progress.set_postfix(loss=f"{loss.detach().item():.4f}")
    return {
        "residual_prediction_loss": error_sum / element_count_sum,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def _deterministic_noise(
    values: torch.Tensor,
    record_indices: torch.Tensor,
    *,
    seed: int,
    direction_offset: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for row, record_index in enumerate(record_indices.tolist()):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(seed) + int(record_index) * 193 + direction_offset
        )
        rows.append(
            torch.randn(
                values[row].shape,
                generator=generator,
                dtype=torch.float32,
            )
        )
    return torch.stack(rows, dim=0).to(values.device)


def _evaluate_variant(
    *,
    model: CrossViewResidualPredictor,
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    mask: torch.Tensor,
    record_indices: torch.Tensor,
    config: dict[str, Any],
    variant: str,
) -> dict[str, float | int]:
    evaluation = config["evaluation"]
    predictor = config["predictor"]
    if variant == "collapsed":
        view_a = view_a.mean(dim=1, keepdim=True).expand_as(view_a)
        view_b = view_b.mean(dim=1, keepdim=True).expand_as(view_b)
    elif variant == "noise":
        noise_std = float(evaluation["noise_std"])
        view_a = view_a + noise_std * _deterministic_noise(
            view_a,
            record_indices,
            seed=int(evaluation["sanity_seed"]),
            direction_offset=0,
        )
        view_b = view_b + noise_std * _deterministic_noise(
            view_b,
            record_indices,
            seed=int(evaluation["sanity_seed"]),
            direction_offset=1_000_003,
        )
    elif variant not in {"main", "spatial_shuffle"}:
        raise ValueError(f"Unknown CV-RVTM evaluation variant: {variant}")
    source, target, direction_mask, direction_indices = _symmetric_directions(
        view_a,
        view_b,
        mask,
        record_indices,
        symmetric=bool(predictor["symmetric"]),
    )
    source_residual, target_residual, _ = residualize_cross_view(
        source,
        target,
        direction_mask,
    )
    if variant == "spatial_shuffle":
        source_residual = spatially_shuffle_visible_tokens(
            source_residual,
            direction_mask,
            direction_indices,
            seed=int(evaluation["sanity_seed"]),
        )
    prediction = model(source_residual, direction_mask)
    return cross_view_loss_sums(
        prediction,
        target,
        target_residual,
        direction_mask,
    )


@torch.inference_mode()
def evaluate(
    *,
    model: CrossViewResidualPredictor,
    cache: CrossViewShardCache,
    indices: Sequence[int] | None,
    mean: torch.Tensor,
    std: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    full_sanity_checks: bool,
) -> dict[str, Any]:
    model.eval()
    training = config["training"]
    variants = ["main"]
    if full_sanity_checks:
        variants.extend(("collapsed", "noise", "spatial_shuffle"))
    sums = {
        variant: {
            "total_sum": 0.0,
            "residual_null_sum": 0.0,
            "residual_prediction_sum": 0.0,
            "element_count": 0,
        }
        for variant in variants
    }
    batches = cache.iter_batches(
        batch_size=int(config["evaluation"]["batch_size"]),
        indices=indices,
        shuffle=False,
        seed=0,
        verify_checksum=full_sanity_checks,
    )
    for raw_a, raw_b, record_indices_cpu in tqdm(
        batches,
        total=cache.batch_count(int(config["evaluation"]["batch_size"]), indices),
        desc=f"evaluate CV-RVTM {cache.split_name} {cache.tokenizer_id}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    ):
        view_a = _normalize(raw_a, mean, std, device)
        view_b = _normalize(raw_b, mean, std, device)
        mask = _make_mask(
            record_indices_cpu,
            config,
            seed=0,
            epoch=0,
            device=device,
        )
        record_indices = record_indices_cpu.to(device)
        with autocast_context(device, str(training["precision"])):
            for variant in variants:
                values = _evaluate_variant(
                    model=model,
                    view_a=view_a,
                    view_b=view_b,
                    mask=mask,
                    record_indices=record_indices,
                    config=config,
                    variant=variant,
                )
                for key, value in values.items():
                    sums[variant][key] += value
    results: dict[str, Any] = {}
    for variant in variants:
        element_count = int(sums[variant]["element_count"])
        losses = {
            "L_total": float(sums[variant]["total_sum"]) / element_count,
            "L_residual_null": (
                float(sums[variant]["residual_null_sum"]) / element_count
            ),
            "L_residual_prediction": (
                float(sums[variant]["residual_prediction_sum"]) / element_count
            ),
        }
        results[variant] = {
            "losses": losses,
            "scores": compute_cvrvtm_scores(losses),
            "target_element_count": element_count,
        }
    main = results.pop("main")
    if results:
        main["sanity_checks"] = results
        main["sanity_pass"] = {
            "collapsed_below_main": (
                results["collapsed"]["scores"]["CVRVTM"]
                < main["scores"]["CVRVTM"]
            ),
            "noise_below_main": (
                results["noise"]["scores"]["CVRVTM"]
                < main["scores"]["CVRVTM"]
            ),
            "spatial_shuffle_below_main": (
                results["spatial_shuffle"]["scores"]["CVRVTM"]
                < main["scores"]["CVRVTM"]
            ),
        }
    return main


def _load_caches_and_indices(
    config: dict[str, Any],
    tokenizer_id: str,
) -> tuple[
    dict[str, CrossViewShardCache],
    dict[str, Sequence[int] | None],
]:
    if "phase0_summary" in config:
        cache = CrossViewShardCache.from_summary(
            Path(config["phase0_summary"]),
            tokenizer_id,
            split_name="shared",
        )
        split_config = config["split"]
        split = make_split_indices(
            cache.record_count,
            train_count=int(split_config["train"]),
            validation_count=int(split_config["validation"]),
            test_count=int(split_config["test"]),
            seed=int(split_config["seed"]),
        )
        expected_tokens = math.prod(
            int(value) for value in config["predictor"]["grid_shape"]
        )
        actual_shape = (
            int(cache.manifest["token_count"]),
            int(cache.manifest["hidden_dim"]),
        )
        expected_shape = (
            expected_tokens,
            int(config["predictor"]["visual_input_dim"]),
        )
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"CV-RVTM cache/predictor shape mismatch: "
                f"{actual_shape} != {expected_shape}"
            )
        return (
            {"train": cache, "validation": cache, "test": cache},
            split,
        )
    caches = {
        split_name: CrossViewShardCache.from_summary(
            Path(config["phase0_summaries"][split_name]),
            tokenizer_id,
            split_name=split_name,
        )
        for split_name in ("train", "validation", "test")
    }
    ids = [
        {int(record["image_id"]) for record in caches[name].records}
        for name in ("train", "validation", "test")
    ]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise RuntimeError("CV-RVTM full split image ids overlap")
    actual_counts = {
        name: cache.record_count for name, cache in caches.items()
    }
    expected_counts = {
        name: int(value) for name, value in config["expected_counts"].items()
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Unexpected CV-RVTM split counts: "
            f"{actual_counts} != {expected_counts}"
        )
    expected_shape = (
        math.prod(int(value) for value in config["predictor"]["grid_shape"]),
        int(config["predictor"]["visual_input_dim"]),
    )
    actual_shapes = {
        (
            int(cache.manifest["token_count"]),
            int(cache.manifest["hidden_dim"]),
        )
        for cache in caches.values()
    }
    if actual_shapes != {expected_shape}:
        raise RuntimeError(
            f"CV-RVTM cache/predictor shape mismatch: "
            f"{actual_shapes} != {expected_shape}"
        )
    return caches, {"train": None, "validation": None, "test": None}


def _available_tokenizers(config: dict[str, Any]) -> set[str]:
    summary_paths = (
        [config["phase0_summary"]]
        if "phase0_summary" in config
        else list(config["phase0_summaries"].values())
    )
    available: set[str] | None = None
    for path in summary_paths:
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        current = {item["tokenizer_id"] for item in summary["tokenizers"]}
        available = current if available is None else available & current
    return available or set()


def train_run(
    *,
    tokenizer_id: str,
    caches: dict[str, CrossViewShardCache],
    indices: dict[str, Sequence[int] | None],
    config: dict[str, Any],
    project_root: Path,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "protocol": "cvrvtm_v1",
        "tokenizer_id": tokenizer_id,
        "cache_keys": {
            name: cache.cache_key for name, cache in caches.items()
        },
        "split": config.get("split"),
        "predictor": config["predictor"],
        "training": config["training"],
        "evaluation": config["evaluation"],
        "seed": seed,
        "code_sha256": code_fingerprint(project_root / "vtm_lcg" / "cvrvtm"),
    }
    run_key = sha256_json(identity)
    identity["run_key"] = run_key
    output_dir = (
        Path(config["artifact_root"])
        / "predictors"
        / tokenizer_id
        / f"seed_{seed}"
        / run_key
    )
    identity_path = output_dir / "run_identity.json"
    result_path = output_dir / "result.json"
    checkpoint_path = output_dir / "predictor.safetensors"
    if identity_path.is_file() and result_path.is_file() and checkpoint_path.is_file():
        existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing_identity == identity
            and result.get("checkpoint_sha256") == sha256_file(checkpoint_path)
        ):
            print(f"{tokenizer_id} seed={seed}: completed CV-RVTM result reused")
            return result
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(identity_path, identity)
    seed_everything(seed)
    training = config["training"]
    mean_cpu, std_cpu = caches["train"].normalization(
        indices=indices["train"],
        epsilon=float(training["stats_epsilon"]),
    )
    mean = mean_cpu.to(device).view(1, 1, -1)
    std = std_cpu.to(device).view(1, 1, -1)
    atomic_save_safetensors(
        output_dir / "normalization.safetensors",
        {"mean": mean_cpu, "std": std_cpu},
    )
    model = _build_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    steps_per_epoch = caches["train"].batch_count(
        int(training["batch_size"]),
        indices["train"],
    )
    scheduler = make_scheduler(
        optimizer,
        total_steps=steps_per_epoch * int(training["epochs"]),
        warmup_ratio=float(training["warmup_ratio"]),
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_objective = float("inf")
    best_epoch = -1
    started = time.perf_counter()
    for epoch in range(int(training["epochs"])):
        train_metrics = train_epoch(
            model=model,
            cache=caches["train"],
            indices=indices["train"],
            mean=mean,
            std=std,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            seed=seed,
            epoch=epoch,
        )
        validation = evaluate(
            model=model,
            cache=caches["validation"],
            indices=indices["validation"],
            mean=mean,
            std=std,
            config=config,
            device=device,
            full_sanity_checks=False,
        )
        objective = float(validation["losses"]["L_residual_prediction"])
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
        print(
            f"{tokenizer_id} seed={seed} epoch={epoch + 1:02d}/"
            f"{training['epochs']} "
            f"train={train_metrics['residual_prediction_loss']:.6f} "
            f"val_pred={objective:.6f} "
            f"CVRVTM={validation['scores']['CVRVTM']:.6f}"
        )
    if best_state is None:
        raise RuntimeError("CV-RVTM training produced no checkpoint")
    model.load_state_dict(best_state)
    atomic_save_safetensors(checkpoint_path, best_state)
    test = evaluate(
        model=model,
        cache=caches["test"],
        indices=indices["test"],
        mean=mean,
        std=std,
        config=config,
        device=device,
        full_sanity_checks=True,
    )
    result = {
        "schema_version": 1,
        "protocol": "cvrvtm_v1",
        "tokenizer_id": tokenizer_id,
        "predictor_seed": seed,
        "run_key": run_key,
        "completed_at": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": model.parameter_count(),
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "normalization": {
            "fit_split": "train",
            "channel_std_min": float(std_cpu.min().item()),
            "channel_std_max": float(std_cpu.max().item()),
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "test": test,
    }
    atomic_write_json(result_path, result)
    return result


def write_summary(
    artifact_root: Path,
    results: list[dict[str, Any]],
) -> None:
    payload = {
        "schema_version": 1,
        "protocol": "cvrvtm_v1",
        "generated_at": utc_now(),
        "results": results,
    }
    atomic_write_json(artifact_root / "summary.json", payload)
    lines = [
        "# CV-RVTM Summary",
        "",
        "| Tokenizer | Seed | Best epoch | L total | L residual null | "
        "L residual pred | CV-RVTM | Residual energy | Residual predictability | "
        "Collapse ✓ | Noise ✓ | Spatial ✓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for result in results:
        test = result["test"]
        losses = test["losses"]
        scores = test["scores"]
        checks = test["sanity_pass"]
        lines.append(
            f"| {result['tokenizer_id']} | {result['predictor_seed']} | "
            f"{result['best_epoch']} | {losses['L_total']:.6f} | "
            f"{losses['L_residual_null']:.6f} | "
            f"{losses['L_residual_prediction']:.6f} | "
            f"{scores['CVRVTM']:.6f} | "
            f"{scores['residual_energy_ratio']:.6f} | "
            f"{scores['residual_predictability']:.6f} | "
            f"{'✓' if checks['collapsed_below_main'] else '✗'} | "
            f"{'✓' if checks['noise_below_main'] else '✗'} | "
            f"{'✓' if checks['spatial_shuffle_below_main'] else '✗'} |"
        )
    lines.append("")
    artifact_root.mkdir(parents=True, exist_ok=True)
    temporary = artifact_root / "summary.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, artifact_root / "summary.md")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the cross-view residual VTM predictor"
    )
    parser.add_argument("--config", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--tokenizer", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config, project_root = load_cvrvtm_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
        config["evaluation"]["batch_size"] = args.batch_size
    available = _available_tokenizers(config)
    if args.all:
        selected = sorted(available)
    else:
        unknown = sorted(set(args.tokenizer) - available)
        if unknown:
            raise ValueError(f"Unknown tokenizer ids: {unknown}")
        selected = list(args.tokenizer)
    device = resolve_device(args.device)
    results: list[dict[str, Any]] = []
    for tokenizer_id in selected:
        caches, indices = _load_caches_and_indices(config, tokenizer_id)
        for seed_value in config["training"]["seeds"]:
            results.append(
                train_run(
                    tokenizer_id=tokenizer_id,
                    caches=caches,
                    indices=indices,
                    config=config,
                    project_root=project_root,
                    device=device,
                    seed=int(seed_value),
                )
            )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results.sort(key=lambda item: (item["tokenizer_id"], item["predictor_seed"]))
    write_summary(Path(config["artifact_root"]), results)
    print(f"CV-RVTM summary: {Path(config['artifact_root']) / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
