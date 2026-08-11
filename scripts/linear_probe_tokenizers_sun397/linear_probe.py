#!/usr/bin/env python
"""SUN397 fixed-surface linear probing aligned with the ImageNet protocol."""

from importlib.util import module_from_spec, spec_from_file_location
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from fvcore.common.checkpoint import Checkpointer
from torchmetrics import MetricCollection


WORKSPACE = Path(__file__).resolve().parents[2]
BASE_SCRIPT_DIR = WORKSPACE / "scripts" / "linear_probe_tokenizers"
BASE_SCRIPT = BASE_SCRIPT_DIR / "linear_probe.py"
OUTPUT_ROOT = WORKSPACE / "outputs" / "sun397_linear_probing_dinov2_single_surface"
PROTOCOL_VERSION = "tokenizer_linear_probe_sun397_single_surface_v1"
DATASET_NAME = "sun397"
NUM_CLASSES = 397
EXPECTED_SPLIT_SIZES = {
    "train": 76_127,
    "validation": 10_875,
    "test": 21_750,
}
SAFE_FEATURE_MICROBATCH_SIZES = {
    "toklip_s": 256,
    "toklip_l": 256,
    "unitok": 256,
    "vqgan": 16,
}

if str(BASE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_SCRIPT_DIR))

_spec = spec_from_file_location("tokenizer_imagenet_linear_probe", BASE_SCRIPT)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load the base linear-probe module from {BASE_SCRIPT}")
base = module_from_spec(_spec)
_spec.loader.exec_module(base)


def _parse_args():
    parser = base._build_parser()
    parser.description = (
        "SUN397 single-surface linear probing using the tokenizer ImageNet protocol"
    )
    parser.set_defaults(
        data_root=str(WORKSPACE / "data" / "hf_datasets"),
        output_root=str(OUTPUT_ROOT),
        feature_microbatch_size=None,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs; one epoch is ceil(train_size / 1024) updates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for head initialization, the shuffled sample stream, and augmentations.",
    )
    return parser.parse_args()


def _configure_base_protocol(*, epochs: int, epoch_length: int, seed: int) -> None:
    base.PROTOCOL_VERSION = PROTOCOL_VERSION
    base.NUM_CLASSES = NUM_CLASSES
    base.EPOCHS = int(epochs)
    base.EPOCH_LENGTH = int(epoch_length)
    base.MAX_UPDATES = int(epochs * epoch_length)
    base.EVAL_PERIOD_UPDATES = int(epoch_length)
    base.SEED = int(seed)


def _build_sun_accuracy_metric() -> MetricCollection:
    """Keep ImageNet-compatible micro accuracy and add an imbalance diagnostic."""
    micro = base.build_metric(
        base.MetricType.MEAN_ACCURACY,
        num_classes=NUM_CLASSES,
    )
    macro = base.build_metric(
        base.MetricType.MEAN_PER_CLASS_ACCURACY,
        num_classes=NUM_CLASSES,
    )
    return MetricCollection(
        {
            "top-1": micro["top-1"],
            "top-5": micro["top-5"],
            "macro_top-1": macro["top-1"],
            "macro_top-5": macro["top-5"],
        }
    )


def _evaluate_validation_heads(
    feature_model,
    head_grid,
    data_loader,
    iteration: int,
    output_dir: Path,
) -> dict:
    return base._evaluate_heads(
        feature_model,
        head_grid,
        data_loader,
        iteration,
        output_dir,
        metric=_build_sun_accuracy_metric(),
    )


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected one JSON object in {path}")
    return payload


def _load_history_iteration(path: Path, iteration: int) -> dict | None:
    if not path.is_file():
        return None
    match = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            if payload.get("iteration") == iteration:
                match = payload
    return match


def _load_completed_validation(output_dir: Path, iteration: int) -> dict | None:
    result_path = output_dir / "results_eval_linear.json"
    history_path = output_dir / "metrics_history.jsonl"
    payload = _load_json(result_path) if result_path.is_file() else None
    if payload is not None and not _is_complete_validation_payload(payload, iteration):
        payload = None
    if payload is None:
        payload = _load_history_iteration(history_path, iteration)
    if payload is None or not _is_complete_validation_payload(payload, iteration):
        return None
    if not base._metrics_history_has_iteration(history_path, iteration):
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    base._atomic_json_dump(result_path, payload)
    return payload


def _is_complete_validation_payload(payload: dict, iteration: int) -> bool:
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("iteration") != iteration
    ):
        return False
    selected_name = payload.get("best_classifier", {}).get("name")
    selected = next(
        (
            classifier
            for classifier in payload.get("classifiers", [])
            if classifier.get("name") == selected_name
        ),
        None,
    )
    return selected is not None and {
        "top-1",
        "top-5",
        "macro_top-1",
        "macro_top-5",
    }.issubset(selected.get("metrics", {}))


def _completed_test_matches(
    output_dir: Path,
    *,
    iteration: int,
    selected_classifier_name: str,
) -> bool:
    path = output_dir / "results_test_linear.json"
    if not path.is_file():
        return False
    payload = _load_json(path)
    return (
        payload.get("protocol_version") == PROTOCOL_VERSION
        and payload.get("iteration") == iteration
        and payload.get("selected_classifier", {}).get("name")
        == selected_classifier_name
    )


def _make_protocol(
    args,
    bundle,
    effective_lrs: list[float],
    *,
    train_size: int,
    val_size: int,
    test_size: int,
) -> dict:
    protocol = base._make_protocol(args, bundle, effective_lrs)
    protocol.pop("fingerprint", None)
    samples_per_epoch = base.EPOCH_LENGTH * base.BATCH_SIZE
    protocol.update(
        {
            "version": PROTOCOL_VERSION,
            "dataset": "SUN397",
            "dataset_backend": "local Hugging Face parquet",
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "train_samples": train_size,
            "validation_samples": val_size,
            "test_samples": test_size,
            "samples_drawn_per_epoch": samples_per_epoch,
            "epoch_coverage_ratio": samples_per_epoch / train_size,
            "epoch_semantics": (
                "ceil(train_samples/global_batch_size) consecutive batches from the "
                "shuffled infinite sampler; approximately one full train pass"
            ),
            "validation_head_selection": "best validation top-1 over the 13 LR heads",
            "test_policy": (
                "evaluate exactly once after training using the head selected on final validation"
            ),
            "reported_metrics": {
                "top-1": "micro accuracy; used for LR-head selection to match ImageNet",
                "top-5": "micro top-5 accuracy",
                "macro_top-1": "unweighted mean per-class top-1 accuracy",
                "macro_top-5": "unweighted mean per-class top-5 accuracy",
            },
            "sun397_split_note": (
                "dpdl-benchmark/sun397 all-image 70/10/20-style split; not the "
                "canonical ten SUN397 50-train/50-test partitions"
            ),
        }
    )
    protocol["fingerprint"] = base._protocol_fingerprint(protocol)
    return protocol


@torch.no_grad()
def _evaluate_selected_test_head(
    feature_model,
    head_grid,
    data_loader,
    *,
    validation_results: dict,
    iteration: int,
    output_dir: Path,
) -> dict:
    selected = validation_results["best_classifier"]
    name = selected["name"]
    if name not in head_grid.heads:
        raise RuntimeError(f"Validation selected an unknown classifier: {name}")

    metric = _build_sun_accuracy_metric()
    _stats, raw_results = base.evaluate(
        feature_model,
        data_loader,
        {name: base.LinearPostprocessor(head_grid.heads[name])},
        {name: metric},
        torch.cuda.current_device(),
    )
    values = raw_results[name]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "iteration": iteration,
        "selection_split": "validation",
        "evaluation_split": "test",
        "selected_classifier": selected,
        "test_metrics": {key: float(value.item()) for key, value in values.items()},
    }
    base._atomic_json_dump(output_dir / "results_test_linear.json", payload)
    base.LOGGER.info("Final SUN397 test result: %s", payload)
    return payload


def main() -> int:
    args = _parse_args()
    if args.feature_microbatch_size is None:
        args.feature_microbatch_size = SAFE_FEATURE_MICROBATCH_SIZES.get(
            args.model,
            base.FEATURE_MICROBATCH_SIZE,
        )
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.feature_microbatch_size <= 0:
        raise ValueError("--feature-microbatch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This protocol requires one CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, found {torch.cuda.device_count()}; "
            "set CUDA_VISIBLE_DEVICES to one device."
        )

    base.distributed.enable(overwrite=True)
    if base.distributed.get_global_size() != 1:
        raise RuntimeError(
            f"Expected world_size=1, got {base.distributed.get_global_size()}"
        )
    base._seed_everything(args.seed)

    device = torch.device("cuda", torch.cuda.current_device())
    output_dir = Path(
        args.output_dir
        or Path(args.output_root) / base.OUTPUT_NAMES[args.model] / f"seed{args.seed}"
    ).resolve()
    if args.no_resume and output_dir.is_dir():
        conflicting_artifacts = [
            path
            for path in (
                output_dir / "last_checkpoint",
                output_dir / "metrics_history.jsonl",
                output_dir / "results_eval_linear.json",
                output_dir / "results_test_linear.json",
            )
            if path.exists()
        ]
        conflicting_artifacts.extend(sorted(output_dir.glob("*.pth")))
        if conflicting_artifacts:
            raise RuntimeError(
                "--no-resume would mix a fresh trajectory with existing run artifacts: "
                + ", ".join(str(path) for path in conflicting_artifacts)
                + ". Choose a new --output-dir."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    base.setup_logging(output=str(output_dir), level=base.logging.INFO)

    data_root = Path(args.data_root).expanduser().resolve()
    dataset_root = data_root / DATASET_NAME
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing SUN397 dataset directory: {dataset_root}")

    train_dataset_str = f"HFDataset:name={DATASET_NAME}:split=TRAIN:root={data_root}"
    val_dataset_str = f"HFDataset:name={DATASET_NAME}:split=VAL:root={data_root}"
    test_dataset_str = f"HFDataset:name={DATASET_NAME}:split=TEST:root={data_root}"

    bundle = base.load_feature_bundle(args.model, args, device)
    base._configure_cuda_math()
    feature_model = base.FrozenFeatureModel(
        bundle,
        device=device,
        microbatch_size=args.feature_microbatch_size,
    ).to(device).eval()

    train_dataset = base.make_dataset(
        dataset_str=train_dataset_str,
        transform=bundle.train_transform,
    )
    val_dataset = base.make_dataset(
        dataset_str=val_dataset_str,
        transform=bundle.eval_transform,
    )
    test_dataset = base.make_dataset(
        dataset_str=test_dataset_str,
        transform=bundle.eval_transform,
    )
    for split_name, dataset in (
        ("train", train_dataset),
        ("validation", val_dataset),
        ("test", test_dataset),
    ):
        actual_split = getattr(dataset, "split", None)
        if actual_split != split_name:
            raise RuntimeError(
                f"Requested SUN397 {split_name}, but the loader resolved {actual_split!r}; "
                "refusing a split fallback that could leak test data"
            )
        if len(dataset) != EXPECTED_SPLIT_SIZES[split_name]:
            raise RuntimeError(
                f"Expected {EXPECTED_SPLIT_SIZES[split_name]} samples in SUN397 "
                f"{split_name}, found {len(dataset)}; this is not the audited local split"
            )
        targets = np.asarray(dataset.get_targets(), dtype=np.int64)
        class_count = len(np.unique(targets))
        if class_count != NUM_CLASSES:
            raise RuntimeError(
                f"Expected {NUM_CLASSES} SUN397 classes in {split_name}, found {class_count}"
            )

    epoch_length = math.ceil(len(train_dataset) / base.BATCH_SIZE)
    _configure_base_protocol(
        epochs=args.epochs,
        epoch_length=epoch_length,
        seed=args.seed,
    )

    sample = train_dataset[0][0].unsqueeze(0).to(device)
    in_dim = base._validate_feature_model(feature_model, sample)
    head_grid, parameter_groups, effective_lrs = base._build_heads(in_dim, device)
    if any(parameter.requires_grad for parameter in feature_model.parameters()):
        raise RuntimeError("Frozen feature model unexpectedly has trainable parameters")

    protocol = _make_protocol(
        args,
        bundle,
        effective_lrs,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
    )
    base._write_or_validate_protocol(output_dir, protocol)
    base.LOGGER.info("Protocol: %s", json.dumps(protocol, sort_keys=True))
    base.LOGGER.info("Feature dimension=%d; heads=%d", in_dim, len(head_grid.heads))

    optimizer = torch.optim.SGD(parameter_groups, momentum=0.9, weight_decay=0.0)
    head_parameter_ids = {id(parameter) for parameter in head_grid.parameters()}
    optimizer_parameter_ids = {
        id(parameter)
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    }
    if optimizer_parameter_ids != head_parameter_ids:
        raise RuntimeError("Optimizer parameters must be exactly the 13 linear heads")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        base.MAX_UPDATES,
        eta_min=0.0,
    )
    checkpointer = Checkpointer(
        head_grid,
        str(output_dir),
        optimizer=optimizer,
        scheduler=scheduler,
    )
    checkpoint = checkpointer.resume_or_load("", resume=not args.no_resume)
    start_update = int(checkpoint.get("iteration", -1)) + 1
    if start_update < 0 or start_update > base.MAX_UPDATES:
        raise RuntimeError(f"Invalid checkpoint update: {start_update}")

    val_loader = base.make_data_loader(
        dataset=val_dataset,
        batch_size=base.EVAL_BATCH_SIZE,
        num_workers=args.num_workers,
        shuffle=False,
        seed=args.seed,
        sampler_type=base.SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
    )
    test_loader = base.make_data_loader(
        dataset=test_dataset,
        batch_size=base.EVAL_BATCH_SIZE,
        num_workers=args.num_workers,
        shuffle=False,
        seed=args.seed,
        sampler_type=base.SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
    )

    if start_update == base.MAX_UPDATES:
        final_validation = _load_completed_validation(
            output_dir,
            base.MAX_UPDATES,
        )
        if final_validation is None:
            final_validation = _evaluate_validation_heads(
                feature_model,
                head_grid,
                val_loader,
                base.MAX_UPDATES,
                output_dir,
            )
        selected_name = final_validation["best_classifier"]["name"]
        if _completed_test_matches(
            output_dir,
            iteration=base.MAX_UPDATES,
            selected_classifier_name=selected_name,
        ):
            base.LOGGER.info(
                "SUN397 run is already complete at update %d; final validation and test "
                "outputs match the checkpoint",
                base.MAX_UPDATES,
            )
            return 0
        _evaluate_selected_test_head(
            feature_model,
            head_grid,
            test_loader,
            validation_results=final_validation,
            iteration=base.MAX_UPDATES,
            output_dir=output_dir,
        )
        return 0

    train_loader = base.make_data_loader(
        dataset=train_dataset,
        batch_size=base.BATCH_SIZE,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed,
        sampler_type=base.SamplerType.SHARDED_INFINITE,
        sampler_advance=start_update * base.BATCH_SIZE,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    base.LOGGER.info(
        "Starting SUN397 %s seed=%d from update %d/%d; epochs=%d, "
        "updates/epoch=%d, samples/epoch=%d, train_size=%d",
        args.model,
        args.seed,
        start_update,
        base.MAX_UPDATES,
        args.epochs,
        base.EPOCH_LENGTH,
        base.EPOCH_LENGTH * base.BATCH_SIZE,
        len(train_dataset),
    )
    if (
        0 < start_update < base.MAX_UPDATES
        and start_update % base.EPOCH_LENGTH == 0
        and not base._metrics_history_has_iteration(
            output_dir / "metrics_history.jsonl", start_update
        )
    ):
        base.LOGGER.info(
            "Recovered checkpoint is missing validation at update %d; evaluating now",
            start_update,
        )
        _evaluate_validation_heads(
            feature_model,
            head_grid,
            val_loader,
            start_update,
            output_dir,
        )

    if start_update < base.MAX_UPDATES:
        metric_logger = base.MetricLogger(delimiter="  ")
        remaining_batches = itertools.islice(
            train_loader,
            base.MAX_UPDATES - start_update,
        )
        update = start_update
        for images, labels in metric_logger.log_every(
            remaining_batches,
            10,
            "Training",
            base.MAX_UPDATES,
            start_update,
        ):
            if images.shape[0] != base.BATCH_SIZE:
                raise RuntimeError(
                    f"Expected optimization batch {base.BATCH_SIZE}, got {images.shape[0]}"
                )
            labels = labels.to(device, non_blocking=True)
            features = feature_model(images)
            if features.shape[0] != base.BATCH_SIZE:
                raise RuntimeError(
                    f"Expected feature batch {base.BATCH_SIZE}, got {features.shape[0]}"
                )
            logits = head_grid(features)
            losses = [
                nn.functional.cross_entropy(output, labels)
                for output in logits.values()
            ]
            loss = torch.stack(losses).sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            completed_updates = update + 1
            if update % 10 == 0:
                metric_logger.update(
                    loss=float(loss.item()),
                    lr=float(optimizer.param_groups[0]["lr"]),
                )
            if (
                completed_updates % base.EPOCH_LENGTH == 0
                and completed_updates < base.MAX_UPDATES
            ):
                checkpointer.save("running_checkpoint_linear_eval", iteration=update)
                _evaluate_validation_heads(
                    feature_model,
                    head_grid,
                    val_loader,
                    completed_updates,
                    output_dir,
                )
            update += 1

    checkpointer.save("model_final", iteration=base.MAX_UPDATES - 1)
    final_validation = _evaluate_validation_heads(
        feature_model,
        head_grid,
        val_loader,
        base.MAX_UPDATES,
        output_dir,
    )
    _evaluate_selected_test_head(
        feature_model,
        head_grid,
        test_loader,
        validation_results=final_validation,
        iteration=base.MAX_UPDATES,
        output_dir=output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
