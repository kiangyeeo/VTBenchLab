from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    repo_root = Path(config.get("repo_root", "."))
    if not repo_root.is_absolute():
        repo_root = (config_path.parent / repo_root).resolve()
    config["repo_root"] = str(repo_root)
    registry = config.get("tokenizer_registry")
    if registry is not None:
        import yaml

        model_path = resolve_path(config, registry["models_yaml"])
        names_path = resolve_path(config, registry["model_list"])
        model_rows = yaml.safe_load(model_path.read_text(encoding="utf-8"))["models"]
        by_name = {str(row["name"]): row for row in model_rows}
        aliases = {str(key): str(value) for key, value in registry.get("aliases", {}).items()}
        names = [
            line.strip()
            for line in names_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(names) != len(set(names)):
            raise RuntimeError(f"Duplicate tokenizer names in {names_path}")
        tokenizers = {}
        for name in names:
            registered_name = aliases.get(name, name)
            if registered_name not in by_name:
                raise RuntimeError(
                    f"Tokenizer {name!r} maps to missing registry row {registered_name!r}"
                )
            row = by_name[registered_name]
            if not bool(row.get("enabled", True)):
                raise RuntimeError(f"Tokenizer registry row is disabled: {registered_name}")
            tokenizers[name] = {
                "loader": "registry",
                "loader_name": str(row["loader_name"]),
                "family": str(row["family"]),
                "registry_name": registered_name,
                "extract_batch_size": int(row["batch_size"]),
            }
        config["tokenizers"] = tokenizers
    return config, config_path


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["repo_root"]) / path
    return path.resolve()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, destination)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def choose_names(requested: list[str] | None, available: Iterable[str]) -> list[str]:
    choices = list(available)
    if not requested or requested == ["all"]:
        return choices
    unknown = sorted(set(requested) - set(choices))
    if unknown:
        raise ValueError(f"Unknown names {unknown}; available={choices}")
    return requested


def device_from_config(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("runtime", {}).get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)
