from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "vtm_lcg").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find VTM_LCG project root above {start}")


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    project_root = find_project_root(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    config = deepcopy(payload)
    for section in ("dataset", "runtime", "preprocess", "tokenizers"):
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    dataset = config["dataset"]
    runtime = config["runtime"]
    if not isinstance(dataset, dict) or not isinstance(runtime, dict):
        raise ValueError("dataset and runtime must be mappings")
    dataset["annotations"] = str(resolve_project_path(project_root, dataset["annotations"]))
    dataset["image_root"] = str(resolve_project_path(project_root, dataset["image_root"]))
    runtime["artifact_root"] = str(resolve_project_path(project_root, runtime["artifact_root"]))

    tokenizers = config["tokenizers"]
    if not isinstance(tokenizers, list) or not tokenizers:
        raise ValueError("tokenizers must be a non-empty list")
    seen_ids: set[str] = set()
    for tokenizer in tokenizers:
        if not isinstance(tokenizer, dict):
            raise ValueError("Each tokenizer config must be a mapping")
        tokenizer_id = str(tokenizer.get("id", "")).strip()
        if not tokenizer_id or tokenizer_id in seen_ids:
            raise ValueError(f"Tokenizer ids must be unique and non-empty: {tokenizer_id!r}")
        seen_ids.add(tokenizer_id)
        tokenizer["checkpoint"] = str(
            resolve_project_path(project_root, tokenizer["checkpoint"])
        )

    return config, project_root


def load_phase1_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    project_root = find_project_root(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config = deepcopy(payload)
    for section in (
        "phase0_summary",
        "artifact_root",
        "split",
        "text",
        "predictor",
        "training",
        "evaluation",
    ):
        if section not in config:
            raise ValueError(f"Missing required Phase 1 config section: {section}")
    config["phase0_summary"] = str(
        resolve_project_path(project_root, config["phase0_summary"])
    )
    config["artifact_root"] = str(
        resolve_project_path(project_root, config["artifact_root"])
    )
    text = config["text"]
    text["checkpoint"] = str(resolve_project_path(project_root, text["checkpoint"]))
    return config, project_root


def load_phase1_full_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    project_root = find_project_root(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config = deepcopy(payload)
    for section in (
        "phase0_summaries",
        "artifact_root",
        "expected_counts",
        "text",
        "predictor",
        "training",
        "evaluation",
    ):
        if section not in config:
            raise ValueError(f"Missing required full-COCO config section: {section}")
    summaries = config["phase0_summaries"]
    for split_name in ("train", "validation", "test"):
        if split_name not in summaries:
            raise ValueError(f"Missing Phase 0 summary for split: {split_name}")
        summaries[split_name] = str(
            resolve_project_path(project_root, summaries[split_name])
        )
    config["artifact_root"] = str(
        resolve_project_path(project_root, config["artifact_root"])
    )
    config["text"]["checkpoint"] = str(
        resolve_project_path(project_root, config["text"]["checkpoint"])
    )
    return config, project_root


def load_cvrvtm_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    project_root = find_project_root(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config = deepcopy(payload)
    for section in ("artifact_root", "predictor", "training", "evaluation"):
        if section not in config:
            raise ValueError(f"Missing required CV-RVTM config section: {section}")
    has_single = "phase0_summary" in config
    has_split_summaries = "phase0_summaries" in config
    if has_single == has_split_summaries:
        raise ValueError(
            "CV-RVTM config requires exactly one of phase0_summary or phase0_summaries"
        )
    if has_single:
        if "split" not in config:
            raise ValueError("Single-cache CV-RVTM config requires split")
        config["phase0_summary"] = str(
            resolve_project_path(project_root, config["phase0_summary"])
        )
    else:
        if "expected_counts" not in config:
            raise ValueError(
                "Split-cache CV-RVTM config requires expected_counts"
            )
        summaries = config["phase0_summaries"]
        for split_name in ("train", "validation", "test"):
            if split_name not in summaries:
                raise ValueError(
                    f"Missing CV-RVTM Phase 0 summary for split: {split_name}"
                )
            summaries[split_name] = str(
                resolve_project_path(project_root, summaries[split_name])
            )
    config["artifact_root"] = str(
        resolve_project_path(project_root, config["artifact_root"])
    )
    return config, project_root


def torch_dtype_from_name(name):
    import torch

    if isinstance(name, torch.dtype):
        return name
    normalized = name.lower().replace("torch.", "")
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported dtype: {name}")
    return aliases[normalized]


def canonical_dtype_name(name) -> str:
    dtype = torch_dtype_from_name(name)
    return str(dtype).removeprefix("torch.")
