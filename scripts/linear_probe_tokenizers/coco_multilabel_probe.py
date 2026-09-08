#!/usr/bin/env python
"""COCO-2014 multi-label probing aligned with the tokenizer linear-probe protocol."""

import gc
from importlib.util import module_from_spec, spec_from_file_location
import itertools
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from fvcore.common.checkpoint import Checkpointer
from sklearn.metrics import average_precision_score


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "linear_probe.py"
OUTPUT_ROOT = WORKSPACE / "outputs" / "coco2014_multilabel_linear_probing"
PROTOCOL_VERSION = "tokenizer_linear_probe_coco2014_multilabel_v1"
NUM_CLASSES = 80
EXPECTED_LABELED_IMAGES = {"train2014": 82_081, "val2014": 40_137}

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

_spec = spec_from_file_location("tokenizer_imagenet_linear_probe", BASE_SCRIPT)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load the base linear-probe module from {BASE_SCRIPT}")
base = module_from_spec(_spec)
_spec.loader.exec_module(base)


class CocoMultilabelDataset(Dataset):
    """COCO images with an 80-dimensional category-presence target."""

    def __init__(self, root: Path, split: str, transform, *, include_unlabeled: bool = False):
        self.root = root
        self.split = split
        self.transform = transform
        image_root = root / split
        annotation_path = root / "annotations" / f"instances_{split}.json"
        if not image_root.is_dir():
            raise FileNotFoundError(f"Missing COCO image directory: {image_root}")
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing COCO instance annotations: {annotation_path}")

        with annotation_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        categories = sorted(payload["categories"], key=lambda item: int(item["id"]))
        if len(categories) != NUM_CLASSES:
            raise RuntimeError(
                f"Expected {NUM_CLASSES} COCO categories in {annotation_path}, "
                f"found {len(categories)}"
            )
        category_to_column = {
            int(category["id"]): column for column, category in enumerate(categories)
        }
        targets_by_image = {
            int(image["id"]): np.zeros(NUM_CLASSES, dtype=np.uint8)
            for image in payload["images"]
        }
        for annotation in payload["annotations"]:
            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])
            targets_by_image[image_id][category_to_column[category_id]] = 1

        images = []
        targets = []
        missing_paths = []
        for image in sorted(payload["images"], key=lambda item: int(item["id"])):
            target = targets_by_image[int(image["id"])]
            if not include_unlabeled and not target.any():
                continue
            path = image_root / image["file_name"]
            if not path.is_file():
                missing_paths.append(path)
                if len(missing_paths) >= 5:
                    break
            images.append(path)
            targets.append(target)
        if missing_paths:
            raise FileNotFoundError(
                "Missing COCO image files (showing up to five): "
                + ", ".join(str(path) for path in missing_paths)
            )

        self.images = images
        self.targets = np.stack(targets, axis=0)
        self.category_ids = [int(category["id"]) for category in categories]
        self.category_names = [str(category["name"]) for category in categories]
        self.include_unlabeled = bool(include_unlabeled)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        with Image.open(self.images[index]) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = torch.from_numpy(self.targets[index].copy()).float()
        return image, target


def _parse_args():
    parser = base._build_parser()
    parser.allow_abbrev = False
    parser.description = (
        "COCO-2014 80-class multi-label linear probing for visual tokenizers"
    )
    parser.set_defaults(
        data_root=str(WORKSPACE / "data" / "gvt" / "raw" / "coco"),
        output_root=str(OUTPUT_ROOT),
        feature_microbatch_size=None,
        stop_after_epoch=None,
    )
    stop_action = next(
        action for action in parser._actions if action.dest == "stop_after_epoch"
    )
    stop_action.help = (
        "Operational epoch cutoff for a resumable run (default: finish --epochs)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs; each epoch is ceil(82081 / 1024) updates.",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help=(
            "Include COCO images with no instance annotations as all-negative targets. "
            "The default excludes them, matching the common 82081/40137 protocol."
        ),
    )
    return parser.parse_args()


def _default_feature_microbatch_size(model: str) -> int:
    """Reuse each model's already-audited ImageNet runner default when available."""
    runner = SCRIPT_DIR / f"run_{model}.sh"
    if runner.is_file():
        match = re.search(
            r'FEATURE_MICROBATCH_SIZE="\$\{FEATURE_MICROBATCH_SIZE:-(\d+)\}"',
            runner.read_text(encoding="utf-8"),
        )
        if match is not None:
            return int(match.group(1))
    return int(base.FEATURE_MICROBATCH_SIZE)


def _uses_batch_norm(model: str) -> bool:
    return model in {
        *base.PIXIO_SPECS,
        *base.WEBSSL_DINO_SPECS,
        *base.WEBSSL_MAE_SPECS,
    }


def _configure_protocol(*, epochs: int, epoch_length: int) -> None:
    base.PROTOCOL_VERSION = PROTOCOL_VERSION
    base.NUM_CLASSES = NUM_CLASSES
    base.EPOCHS = int(epochs)
    base.EPOCH_LENGTH = int(epoch_length)
    base.MAX_UPDATES = int(epochs * epoch_length)
    base.EVAL_PERIOD_UPDATES = int(epoch_length)
    base.SEED = 0


def _safe_divide(numerator: np.ndarray | float, denominator: np.ndarray | float):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(np.asarray(numerator), dtype=np.float64),
        where=np.asarray(denominator) != 0,
    )


def _classification_metrics(targets: np.ndarray, logits: np.ndarray) -> tuple[dict, list]:
    """Compute standard COCO multi-label mAP and threshold/top-3 P/R/F1."""
    if targets.shape != logits.shape or targets.ndim != 2:
        raise ValueError(
            f"Expected matching [N,C] targets/logits, got {targets.shape}/{logits.shape}"
        )
    targets_bool = targets.astype(bool, copy=False)
    per_class_ap = average_precision_score(targets_bool, logits, average=None)
    metrics = {
        "mAP": float(np.mean(per_class_ap) * 100.0),
        "micro_AP": float(
            average_precision_score(targets_bool, logits, average="micro") * 100.0
        ),
    }

    def add_precision_recall_f1(prefix: str, predictions: np.ndarray) -> None:
        true_positive = np.logical_and(predictions, targets_bool).sum(axis=0)
        predicted_positive = predictions.sum(axis=0)
        actual_positive = targets_bool.sum(axis=0)
        class_precision = _safe_divide(true_positive, predicted_positive)
        class_recall = _safe_divide(true_positive, actual_positive)
        class_precision_mean = float(class_precision.mean())
        class_recall_mean = float(class_recall.mean())
        class_f1 = float(
            _safe_divide(
                2.0 * class_precision_mean * class_recall_mean,
                class_precision_mean + class_recall_mean,
            )
        )
        overall_precision = float(
            _safe_divide(true_positive.sum(), predicted_positive.sum())
        )
        overall_recall = float(_safe_divide(true_positive.sum(), actual_positive.sum()))
        overall_f1 = float(
            _safe_divide(
                2.0 * overall_precision * overall_recall,
                overall_precision + overall_recall,
            )
        )
        metrics.update(
            {
                f"{prefix}CP": class_precision_mean * 100.0,
                f"{prefix}CR": class_recall_mean * 100.0,
                f"{prefix}CF1": class_f1 * 100.0,
                f"{prefix}OP": overall_precision * 100.0,
                f"{prefix}OR": overall_recall * 100.0,
                f"{prefix}OF1": overall_f1 * 100.0,
            }
        )

    add_precision_recall_f1("", logits >= 0.0)
    top3 = np.zeros_like(targets_bool)
    top3_columns = np.argpartition(logits, kth=-3, axis=1)[:, -3:]
    np.put_along_axis(top3, top3_columns, True, axis=1)
    add_precision_recall_f1("top3_", top3)
    return metrics, [float(value * 100.0) for value in per_class_ap]


@torch.no_grad()
def _evaluate_heads(feature_model, head_grid, data_loader, iteration: int, output_dir: Path, category_names):
    prediction_chunks = {name: [] for name in head_grid.heads}
    target_chunks = []
    was_training = head_grid.training
    head_grid.eval()
    try:
        for images, targets in data_loader:
            features = feature_model(images)
            outputs = head_grid(features)
            target_chunks.append(targets.to(dtype=torch.uint8, device="cpu"))
            for name, logits in outputs.items():
                prediction_chunks[name].append(logits.float().cpu())
            del images, targets, features, outputs
    finally:
        head_grid.train(was_training)

    targets = torch.cat(target_chunks, dim=0).numpy()
    classifiers = []
    for name, chunks in prediction_chunks.items():
        logits = torch.cat(chunks, dim=0).numpy()
        metrics, per_class_ap = _classification_metrics(targets, logits)
        head = head_grid.heads[name]
        classifiers.append(
            {
                "name": name,
                "readout": "fixed_preselected_representation",
                "base_lr": head.base_lr,
                "configured_lr": head.base_lr,
                "effective_lr": head.effective_lr,
                "lr_boundary": (
                    "low"
                    if head.base_lr == min(base.BASE_LEARNING_RATES)
                    else "high"
                    if head.base_lr == max(base.BASE_LEARNING_RATES)
                    else "interior"
                ),
                "metrics": metrics,
                "per_class_AP": dict(zip(category_names, per_class_ap)),
            }
        )
        del logits
    classifiers.sort(key=lambda item: item["base_lr"])
    best = max(classifiers, key=lambda item: item["metrics"]["mAP"])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "iteration": iteration,
        "selection_metric": "mAP",
        "best_classifier": {
            "name": best["name"],
            "mAP": best["metrics"]["mAP"],
            "micro_AP": best["metrics"]["micro_AP"],
            "OF1": best["metrics"]["OF1"],
            "base_lr": best["base_lr"],
            "configured_lr": best["configured_lr"],
            "effective_lr": best["effective_lr"],
            "lr_boundary": best["lr_boundary"],
            "readout": best["readout"],
        },
        "classifiers": classifiers,
    }
    base._atomic_json_dump(output_dir / "results_eval_multilabel.json", payload)
    with (output_dir / "metrics_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    base.LOGGER.info("Validation at update %d: best=%s", iteration, payload["best_classifier"])
    return payload


def _make_protocol(args, bundle, effective_lrs, train_dataset, val_dataset, use_batch_norm):
    protocol = {
        "version": PROTOCOL_VERSION,
        "model": args.model,
        "representation": bundle.representation,
        "transform": bundle.transform_description,
        "train_augmentation": (
            "enabled; use the tokenizer's bundle.train_transform exactly as in the "
            "ImageNet linear probe"
        ),
        "validation_transform": "deterministic bundle.eval_transform",
        "backbone_precision": bundle.backbone_precision,
        "linear_head_precision": "float32",
        "checkpoint_paths": bundle.checkpoint_paths,
        "dataset": "MS COCO 2014 multi-label classification",
        "dataset_root": str(Path(args.data_root).expanduser().resolve()),
        "train_split": "train2014",
        "validation_split": "val2014",
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "include_images_without_instance_labels": args.include_unlabeled,
        "category_order": "ascending official COCO category_id",
        "num_classes": NUM_CLASSES,
        "single_gpu": True,
        "global_batch_size": base.BATCH_SIZE,
        "feature_extraction_microbatch_size": args.feature_microbatch_size,
        "feature_microbatch_semantics": (
            "frozen backbone only; concatenate one FP32 [global_batch,D] tensor before heads"
        ),
        "validation_batch_size": base.EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "epochs": base.EPOCHS,
        "epoch_length_updates": base.EPOCH_LENGTH,
        "epoch_semantics": "ceil(train_samples/global_batch_size) shuffled infinite-sampler batches",
        "max_updates": base.MAX_UPDATES,
        "eval_period_updates": base.EVAL_PERIOD_UPDATES,
        "seed": base.SEED,
        "readout_search": False,
        "multi_block_search": False,
        "feature_normalization": use_batch_norm,
        "loss": "BCEWithLogits; sum over 80 classes then mean over batch",
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 0.0,
        "bias": True,
        "head_weight_init": "normal(mean=0,std=0.01)",
        "head_bias_init": 0.0,
        "lr_schedule": "cosine_to_zero",
        "warmup_updates": 0,
        "lr_scaling": "effective_lr = base_lr * global_batch_size / 256",
        "base_learning_rates": list(base.BASE_LEARNING_RATES),
        "effective_learning_rates": effective_lrs,
        "validation_head_selection": "highest macro class-wise average precision (mAP)",
        "validation_note": (
            "val2014 is used both for LR-head selection and reporting, matching the ImageNet "
            "linear-probe train/val surface; no independent COCO test labels are available"
        ),
        "reported_metric_units": "percentage points",
        "cuda_matmul_tf32": True,
        "cudnn_tf32": True,
    }
    if use_batch_norm:
        protocol.update(
            {
                "feature_normalization_type": "BatchNorm1d",
                "feature_normalization_placement": "immediately before each linear classifier",
                "batch_norm_affine": False,
                "batch_norm_eps": 1e-6,
                "batch_norm_momentum": 0.1,
                "batch_norm_track_running_stats": True,
                "batch_norm_training_batch_size": base.BATCH_SIZE,
                "batch_norm_reason": "required for WebSSL and Pixio only",
            }
        )
    protocol["fingerprint"] = base._protocol_fingerprint(protocol)
    return protocol


def _validation_exists(output_dir: Path, iteration: int) -> bool:
    path = output_dir / "metrics_history.jsonl"
    return path.is_file() and base._metrics_history_has_iteration(path, iteration)


def main() -> int:
    args = _parse_args()
    if args.feature_microbatch_size is None:
        args.feature_microbatch_size = _default_feature_microbatch_size(args.model)
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.stop_after_epoch is None:
        args.stop_after_epoch = args.epochs
    if not 1 <= args.stop_after_epoch <= args.epochs:
        raise ValueError("--stop-after-epoch must be in [1, --epochs]")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.feature_microbatch_size <= 0:
        raise ValueError("--feature-microbatch-size must be positive")
    if args.model not in base.PIXIO_SPECS and args.pixio_readout != "post-ln":
        raise ValueError("--pixio-readout pre-ln is only valid for Pixio models")
    if not torch.cuda.is_available():
        raise RuntimeError("This protocol requires one CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, found {torch.cuda.device_count()}; "
            "set CUDA_VISIBLE_DEVICES to one device."
        )

    base.distributed.enable(overwrite=True)
    if base.distributed.get_global_size() != 1:
        raise RuntimeError(f"Expected world_size=1, got {base.distributed.get_global_size()}")
    base._seed_everything(0)

    device = torch.device("cuda", torch.cuda.current_device())
    use_batch_norm = _uses_batch_norm(args.model)
    output_name = base.OUTPUT_NAMES[args.model]
    if use_batch_norm and args.model not in base.PIXIO_SPECS:
        output_name = f"{output_name}_bn"
    if args.model in base.PIXIO_SPECS and args.pixio_readout == "pre-ln":
        output_name = f"{args.model}_pre_ln_bn"
    output_dir = Path(args.output_dir or Path(args.output_root) / output_name).resolve()
    if args.no_resume and output_dir.is_dir():
        conflicting_artifacts = [
            path
            for path in (
                output_dir / "last_checkpoint",
                output_dir / "metrics_history.jsonl",
                output_dir / "results_eval_multilabel.json",
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
    bundle = base._load_probe_feature_bundle(args, device)
    base._configure_cuda_math()
    feature_model = base.FrozenFeatureModel(
        bundle, device=device, microbatch_size=args.feature_microbatch_size
    ).to(device).eval()

    train_dataset = CocoMultilabelDataset(
        data_root, "train2014", bundle.train_transform,
        include_unlabeled=args.include_unlabeled,
    )
    val_dataset = CocoMultilabelDataset(
        data_root, "val2014", bundle.eval_transform,
        include_unlabeled=args.include_unlabeled,
    )
    if train_dataset.category_ids != val_dataset.category_ids:
        raise RuntimeError("COCO train2014 and val2014 category orders differ")
    if not args.include_unlabeled:
        for dataset in (train_dataset, val_dataset):
            expected = EXPECTED_LABELED_IMAGES[dataset.split]
            if len(dataset) != expected:
                raise RuntimeError(
                    f"Expected {expected} labeled {dataset.split} images, found {len(dataset)}"
                )

    epoch_length = math.ceil(len(train_dataset) / base.BATCH_SIZE)
    _configure_protocol(epochs=args.epochs, epoch_length=epoch_length)
    stop_update = args.stop_after_epoch * base.EPOCH_LENGTH

    sample = train_dataset[0][0].unsqueeze(0).to(device)
    in_dim = base._validate_feature_model(feature_model, sample)
    head_grid, parameter_groups, effective_lrs = base._build_heads(
        in_dim, device, use_batch_norm=use_batch_norm
    )
    if any(parameter.requires_grad for parameter in feature_model.parameters()):
        raise RuntimeError("Frozen feature model unexpectedly has trainable parameters")

    protocol = _make_protocol(
        args, bundle, effective_lrs, train_dataset, val_dataset, use_batch_norm
    )
    base._write_or_validate_protocol(output_dir, protocol)
    base.LOGGER.info("Protocol: %s", json.dumps(protocol, sort_keys=True))
    base.LOGGER.info(
        "Feature dimension=%d; heads=%d; BatchNorm=%s",
        in_dim, len(head_grid.heads), use_batch_norm,
    )

    optimizer = torch.optim.SGD(parameter_groups, momentum=0.9, weight_decay=0.0)
    head_parameter_ids = {id(parameter) for parameter in head_grid.parameters()}
    optimizer_parameter_ids = {
        id(parameter)
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    }
    if optimizer_parameter_ids != head_parameter_ids:
        raise RuntimeError("Optimizer parameters must be exactly the 13 probe heads")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, base.MAX_UPDATES, eta_min=0.0
    )
    checkpointer = Checkpointer(
        head_grid, str(output_dir), optimizer=optimizer, scheduler=scheduler
    )
    checkpoint = checkpointer.resume_or_load("", resume=not args.no_resume)
    start_update = int(checkpoint.get("iteration", -1)) + 1
    if start_update < 0 or start_update > base.MAX_UPDATES:
        raise RuntimeError(f"Invalid checkpoint update: {start_update}")
    if start_update % base.EPOCH_LENGTH != 0:
        raise RuntimeError(
            f"Checkpoint update {start_update} is not an epoch boundary of "
            f"{base.EPOCH_LENGTH} updates"
        )
    if start_update > stop_update:
        raise RuntimeError(
            f"Checkpoint is at update {start_update}, beyond requested cutoff {stop_update}"
        )
    if start_update > 0:
        fingerprint = checkpoint.get("protocol_fingerprint")
        if fingerprint != protocol["fingerprint"]:
            raise RuntimeError("Checkpoint protocol fingerprint does not match protocol.json")
        expected_epoch = start_update // base.EPOCH_LENGTH
        if checkpoint.get("completed_epoch") != expected_epoch:
            raise RuntimeError("Checkpoint completed_epoch does not match its iteration")
    if scheduler.last_epoch != start_update:
        raise RuntimeError(
            "Scheduler state does not match checkpoint iteration: "
            f"last_epoch={scheduler.last_epoch}, start_update={start_update}"
        )

    val_loader = base.make_data_loader(
        dataset=val_dataset,
        batch_size=base.EVAL_BATCH_SIZE,
        num_workers=args.num_workers,
        shuffle=False,
        seed=base.SEED,
        sampler_type=base.SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
    )

    if start_update == stop_update:
        if not _validation_exists(output_dir, stop_update):
            _evaluate_heads(
                feature_model, head_grid, val_loader, stop_update, output_dir,
                val_dataset.category_names,
            )
        base.LOGGER.info("Requested epoch cutoff is already complete")
        return 0

    if start_update > 0 and not _validation_exists(output_dir, start_update):
        base.LOGGER.info(
            "Recovered checkpoint is missing validation at update %d; evaluating now",
            start_update,
        )
        _evaluate_heads(
            feature_model, head_grid, val_loader, start_update, output_dir,
            val_dataset.category_names,
        )

    train_loader = base.make_data_loader(
        dataset=train_dataset,
        batch_size=base.BATCH_SIZE,
        num_workers=args.num_workers,
        shuffle=True,
        seed=base.SEED,
        sampler_type=base.SamplerType.SHARDED_INFINITE,
        sampler_advance=start_update * base.BATCH_SIZE,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    base.LOGGER.info(
        "Starting COCO multi-label %s from update %d/%d; stop epoch=%d; "
        "updates/epoch=%d; batch=%d; feature microbatch=%d",
        args.model, start_update, base.MAX_UPDATES, args.stop_after_epoch,
        base.EPOCH_LENGTH, base.BATCH_SIZE, args.feature_microbatch_size,
    )

    metric_logger = base.MetricLogger(delimiter="  ")
    train_iterator = iter(train_loader)
    remaining_batches = itertools.islice(train_iterator, stop_update - start_update)
    update = start_update
    for images, targets in metric_logger.log_every(
        remaining_batches, 10, "Training", stop_update, start_update
    ):
        if images.shape[0] != base.BATCH_SIZE:
            raise RuntimeError(
                f"Expected optimization batch {base.BATCH_SIZE}, got {images.shape[0]}"
            )
        targets = targets.to(device, non_blocking=True)
        features = feature_model(images)
        outputs = head_grid(features)
        losses = [
            nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
            / targets.shape[0]
            for logits in outputs.values()
        ]
        loss = torch.stack(losses).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        completed_updates = update + 1
        if update % 10 == 0:
            metric_logger.update(
                loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"])
            )
        if completed_updates % base.EPOCH_LENGTH == 0:
            checkpointer.save(
                "model_final" if completed_updates == base.MAX_UPDATES else "running_checkpoint_linear_eval",
                iteration=update,
                completed_epoch=completed_updates // base.EPOCH_LENGTH,
                protocol_fingerprint=protocol["fingerprint"],
            )
            if completed_updates != stop_update:
                _evaluate_heads(
                    feature_model, head_grid, val_loader, completed_updates, output_dir,
                    val_dataset.category_names,
                )
        update += 1
        del images, targets, features, outputs, losses, loss

    del remaining_batches
    base._shutdown_loader_iterator(train_iterator)
    del train_iterator, train_loader
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    if not _validation_exists(output_dir, stop_update):
        _evaluate_heads(
            feature_model, head_grid, val_loader, stop_update, output_dir,
            val_dataset.category_names,
        )
    base.LOGGER.info(
        "COCO multi-label %s reached epoch %d/%d (update %d)",
        args.model, args.stop_after_epoch, args.epochs, stop_update,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
