#!/usr/bin/env python3
"""Profile the actual frozen feature surface for the 64-model 5-shot panel."""

from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DRIVER_PATH = SCRIPT_DIR / "linear_probe.py"
MANIFEST_PATH = SCRIPT_DIR / "tokenizers.tsv"
RESULT_PREFIX = "FLOP_RESULT="


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_models() -> list[str]:
    models = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            _rank, _label, model, _head = line.rstrip("\n").split("\t")
            models.append(model)
    if len(models) != 64 or len(set(models)) != 64:
        raise RuntimeError("Expected exactly 64 unique models")
    return models


def _profile_one(model: str) -> dict:
    import torch
    from PIL import Image
    from torch.utils.flop_counter import FlopCounterMode

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU")
    driver = _load_module("five_shot_flop_driver", DRIVER_PATH)
    args = driver.driver._build_parser().parse_args(["--model", model])
    device = torch.device("cuda", 0)
    bundle = driver.driver.base._load_probe_feature_bundle(args, device)
    image = Image.new("RGB", (1024, 1024), color=(127, 127, 127))
    inputs = bundle.eval_transform(image).unsqueeze(0).to(device)
    counter = FlopCounterMode(display=False)
    bundle.encoder.eval()
    with bundle.autocast_context(), counter:
        features = bundle.encoder(inputs)
    torch.cuda.synchronize()
    flops = int(counter.get_total_flops())
    if flops <= 0:
        raise RuntimeError(f"Profiler returned non-positive FLOPs for {model}")
    return {
        "model": model,
        "flops_per_image": flops,
        "gflops_per_image": flops / 1e9,
        "feature_shape": list(features.shape),
    }


def _profile_all() -> int:
    rows = []
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(WORKSPACE / "dinov2") + (
        f":{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else ""
    )
    for index, model in enumerate(_manifest_models(), start=1):
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--one", model],
            cwd=WORKSPACE,
            env=environment,
            capture_output=True,
            text=True,
        )
        payload_line = next(
            (
                line[len(RESULT_PREFIX) :]
                for line in process.stdout.splitlines()
                if line.startswith(RESULT_PREFIX)
            ),
            None,
        )
        if process.returncode or payload_line is None:
            print(f"[{index:02d}/64] FAILED {model}", flush=True)
            print(process.stdout[-4000:], flush=True)
            print(process.stderr[-4000:], flush=True)
            return process.returncode or 1
        payload = json.loads(payload_line)
        rows.append(payload)
        print(
            f"[{index:02d}/64] {model}: {payload['gflops_per_image']:.6f} GFLOPs/image",
            flush=True,
        )

    mean_flops_per_image = sum(row["flops_per_image"] for row in rows) / len(rows)
    summary = {
        "models": len(rows),
        "flop_convention": "PyTorch operator FLOPs; multiply and add count separately",
        "mean_flops_per_image": mean_flops_per_image,
        "mean_gflops_per_image": mean_flops_per_image / 1e9,
        "mean_training_pflops_5shot": mean_flops_per_image * 5000 / 1e15,
        "total_training_pflops_64_models": sum(
            row["flops_per_image"] * 5000 for row in rows
        )
        / 1e15,
        "rows": rows,
    }
    print("FLOP_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", choices=_manifest_models())
    args = parser.parse_args()
    if args.one:
        print(RESULT_PREFIX + json.dumps(_profile_one(args.one), sort_keys=True))
        return 0
    return _profile_all()


if __name__ == "__main__":
    sys.exit(main())
