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


def torch_dtype_from_name(name: str):
    import torch

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


def canonical_dtype_name(name: str) -> str:
    dtype = torch_dtype_from_name(name)
    return str(dtype).removeprefix("torch.")

