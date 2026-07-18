#!/usr/bin/env python
"""Batch-normalized variant of the tokenizer linear-probing baseline.

This entry point deliberately reuses the feature extractors and training loop
from ``scripts/linear_probe_tokenizers`` while replacing each linear head with
the MAE-style ``BatchNorm1d -> Linear`` probe recommended by Lee et al.
The original no-BN implementation and its outputs remain untouched.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import torch
from torch import nn


WORKSPACE = Path(__file__).resolve().parents[2]
BASE_SCRIPT_DIR = WORKSPACE / "scripts" / "linear_probe_tokenizers"
BASE_SCRIPT = BASE_SCRIPT_DIR / "linear_probe.py"
BN_OUTPUT_ROOT = WORKSPACE / "outputs" / "vae_linear_probing_dinov2_single_paperlr_bn"
BN_PROTOCOL_VERSION = "tokenizer_linear_probe_dinov2_single_surface_bn_v1"

if str(BASE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_SCRIPT_DIR))

_spec = spec_from_file_location("tokenizer_linear_probe_without_bn", BASE_SCRIPT)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load the baseline linear-probe module from {BASE_SCRIPT}")
_baseline = module_from_spec(_spec)
_spec.loader.exec_module(_baseline)


class BatchNormalizedLinearHead(nn.Module):
    """Frozen-affine feature BatchNorm followed by one trainable linear head."""

    def __init__(self, in_dim: int, base_lr: float, effective_lr: float):
        super().__init__()
        self.base_lr = float(base_lr)
        self.effective_lr = float(effective_lr)
        # affine=False is equivalent to fixing gamma=1 and beta=0, as in the
        # paper. Running statistics are learned from the full optimization
        # batch and used during validation.
        self.batch_norm = nn.BatchNorm1d(
            in_dim,
            eps=1e-6,
            momentum=0.1,
            affine=False,
            track_running_stats=True,
        )
        self.linear = nn.Linear(in_dim, _baseline.NUM_CLASSES, bias=True)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.batch_norm(features))


_original_make_protocol = _baseline._make_protocol


def _make_bn_protocol(args, bundle, effective_lrs: list[float]) -> dict:
    protocol = _original_make_protocol(args, bundle, effective_lrs)
    protocol.pop("fingerprint", None)
    protocol.update(
        {
            "feature_normalization": True,
            "feature_normalization_type": "BatchNorm1d",
            "feature_normalization_placement": "immediately before each linear classifier",
            "batch_norm_affine": False,
            "batch_norm_fixed_scale": 1.0,
            "batch_norm_fixed_shift": 0.0,
            "batch_norm_eps": 1e-6,
            "batch_norm_momentum": 0.1,
            "batch_norm_track_running_stats": True,
            "batch_norm_training_batch_size": _baseline.BATCH_SIZE,
            "batch_norm_applied_after_feature_microbatch_concatenation": True,
        }
    )
    protocol["fingerprint"] = _baseline._protocol_fingerprint(protocol)
    return protocol


_original_evaluate_heads = _baseline._evaluate_heads


@torch.no_grad()
def _evaluate_heads_with_running_stats(
    feature_model,
    head_grid,
    data_loader,
    iteration: int,
    output_dir: Path,
):
    """Evaluate BN with running statistics, then restore the training mode."""

    was_training = head_grid.training
    head_grid.eval()
    try:
        return _original_evaluate_heads(
            feature_model,
            head_grid,
            data_loader,
            iteration,
            output_dir,
        )
    finally:
        head_grid.train(was_training)


# Patch only the protocol surface that differs from the baseline. All data,
# feature-extraction, optimizer, schedule, LR-grid, checkpointing, and metric
# code continues to come from the original implementation.
_baseline.PROTOCOL_VERSION = BN_PROTOCOL_VERSION
_baseline.LinearHead = BatchNormalizedLinearHead
_baseline._make_protocol = _make_bn_protocol
_baseline._evaluate_heads = _evaluate_heads_with_running_stats


def main() -> int:
    # Keep direct invocations away from the old no-BN output directory. The
    # launcher scripts also pass this explicitly.
    has_output_location = any(
        argument in {"--output-root", "--output-dir"}
        or argument.startswith("--output-root=")
        or argument.startswith("--output-dir=")
        for argument in sys.argv[1:]
    )
    if not has_output_location:
        sys.argv.extend(["--output-root", str(BN_OUTPUT_ROOT)])
    return _baseline.main()


if __name__ == "__main__":
    sys.exit(main())
