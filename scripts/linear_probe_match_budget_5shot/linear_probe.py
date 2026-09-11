#!/usr/bin/env python
"""Five-shot ImageNet-1K match-budget probing for the combined 64-model panel."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
FOUR_SHOT_PATH = WORKSPACE / "scripts/linear_probe_match_budget/linear_probe.py"
MANIFEST_PATH = SCRIPT_DIR / "tokenizers.tsv"
OUTPUT_ROOT = WORKSPACE / "outputs/vae_linear_probing_match_budget_5shot"

PROTOCOL_VERSION = "tokenizer_linear_probe_imagenet_5shot_1epoch_v1"
NUM_CLASSES = 1_000
SHOTS_PER_CLASS = 5
TRAIN_SAMPLES = NUM_CLASSES * SHOTS_PER_CLASS
GLOBAL_BATCH_SIZE = 1_000
EPOCHS = 1
EPOCH_LENGTH = TRAIN_SAMPLES // GLOBAL_BATCH_SIZE


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_manifest() -> tuple[tuple[int, str, str, str], ...]:
    rows = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) != 4:
                raise RuntimeError(
                    f"Expected four TSV columns at {MANIFEST_PATH}:{line_number}"
                )
            rank, label, model, head = columns
            rows.append((int(rank), label, model, head))
    if len(rows) != 64:
        raise RuntimeError(f"Expected 64 unique tokenizer rows, found {len(rows)}")
    if [rank for rank, *_rest in rows] != list(range(1, 65)):
        raise RuntimeError("Tokenizer ranks must be exactly 1..64")
    models = [model for _rank, _label, model, _head in rows]
    if len(set(models)) != len(models):
        raise RuntimeError("The combined 5-shot manifest contains duplicate model ids")
    return tuple(rows)


driver = _load_module("tokenizer_match_budget_4shot_driver", FOUR_SHOT_PATH)
MANIFEST = _load_manifest()
MODEL_CHOICES = tuple(model for _rank, _label, model, _head in MANIFEST)
if set(MODEL_CHOICES) != set(driver.MODEL_CHOICES):
    missing = sorted(set(MODEL_CHOICES) - set(driver.MODEL_CHOICES))
    unexpected = sorted(set(driver.MODEL_CHOICES) - set(MODEL_CHOICES))
    raise RuntimeError(
        f"5-shot/base model-set mismatch: missing={missing}, unexpected={unexpected}"
    )
if set(driver.FEATURE_MICROBATCH_SIZES) != set(MODEL_CHOICES):
    raise RuntimeError("Feature-microbatch defaults must cover all 64 models")

MODEL_METADATA = {
    model: {"rank": rank, "vision_encoder": label, "head": head}
    for rank, label, model, head in MANIFEST
}
expected_bn = {*driver.base.PIXIO_SPECS, *driver.base.WEBSSL_MAE_SPECS} & set(MODEL_CHOICES)
selected_bn = {model for _rank, _label, model, head in MANIFEST if head == "bn"}
if selected_bn != expected_bn:
    raise RuntimeError(
        f"BN routing mismatch: selected={sorted(selected_bn)}, expected={sorted(expected_bn)}"
    )

# Reuse the thoroughly checked 4-shot implementation while replacing every
# training-budget constant before its parser, scheduler, sampler, and protocol
# are constructed. The output root is separate, so 4-shot heads cannot resume.
driver.PROTOCOL_VERSION = PROTOCOL_VERSION
driver.SHOTS_PER_CLASS = SHOTS_PER_CLASS
driver.TRAIN_SAMPLES = TRAIN_SAMPLES
driver.EPOCHS = EPOCHS
driver.EPOCH_LENGTH = EPOCH_LENGTH
driver.OUTPUT_ROOT = OUTPUT_ROOT
driver.MANIFEST = MANIFEST
driver.EXTRA_MANIFEST = ()
driver.MODEL_CHOICES = MODEL_CHOICES
driver.MODEL_METADATA = MODEL_METADATA

driver.base.PROTOCOL_VERSION = PROTOCOL_VERSION
driver.base.BATCH_SIZE = GLOBAL_BATCH_SIZE
driver.base.EPOCHS = EPOCHS
driver.base.EPOCH_LENGTH = EPOCH_LENGTH
driver.base.MAX_UPDATES = EPOCH_LENGTH
driver.base.EVAL_PERIOD_UPDATES = EPOCH_LENGTH

EVAL_FEATURE_MICROBATCH_SIZES = {
    # TokLIP-L at 384px reaches ~77.5 GiB with 256 images per encoder chunk.
    # This affects only frozen validation chunking; the outer batch stays 1,024.
    "toklip_l": 128,
}

_five_shot_parse_args = driver._parse_args


def _parse_args():
    args = _five_shot_parse_args()
    if args.eval_feature_microbatch_size is None:
        args.eval_feature_microbatch_size = EVAL_FEATURE_MICROBATCH_SIZES.get(
            args.model, args.feature_microbatch_size
        )
    return args


driver._parse_args = _parse_args
driver.base._parse_args = _parse_args

_five_shot_base_protocol = driver._make_protocol


def _make_protocol(args, bundle, effective_lrs):
    protocol = _five_shot_base_protocol(args, bundle, effective_lrs)
    protocol.update(
        {
            "version": PROTOCOL_VERSION,
            "train_sampling": "exactly 5 distinct images per class without replacement",
            "budget_scope": (
                "5,000 frozen-tokenizer training forwards; official validation "
                "forwards are reported separately"
            ),
        }
    )
    protocol["fingerprint"] = driver.base._protocol_fingerprint(protocol)
    return protocol


driver._make_protocol = _make_protocol
driver.base._make_protocol = _make_protocol


def main() -> int:
    return driver.main()


if __name__ == "__main__":
    sys.exit(main())
