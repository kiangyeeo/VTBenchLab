import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
DINO_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = DINO_ROOT.parent
REPO_ROOT = DINO_ROOT
DEFAULT_MODEL_CONFIG = PACKAGE_ROOT / "configs" / "models.yaml"
DEFAULT_DATASET_CONFIG = PACKAGE_ROOT / "configs" / "datasets.yaml"


def load_yaml(path: os.PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_model_config(path: Optional[str] = None) -> Dict[str, Any]:
    return load_yaml(path or DEFAULT_MODEL_CONFIG)


def load_dataset_config(path: Optional[str] = None) -> Dict[str, Any]:
    return load_yaml(path or DEFAULT_DATASET_CONFIG)


def repo_path(path: str | os.PathLike) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def workspace_path(path: str | os.PathLike) -> Path:
    p = Path(path)
    return p if p.is_absolute() else WORKSPACE_ROOT / p


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def split_csv(values: Optional[str], default: Iterable[str]) -> List[str]:
    if not values:
        return list(default)
    out: List[str] = []
    for value in values.split(","):
        value = value.strip()
        if value:
            out.append(value)
    return out


def write_json(path: str | os.PathLike, payload: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--dataset-config", default=str(DEFAULT_DATASET_CONFIG))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-root", default="checkpoints/dinov2_baseline")
    parser.add_argument("--feature-root", default=None)
    parser.add_argument("--result-root", default=None)


def configured_root(args: argparse.Namespace, key: str, fallback: str) -> Path:
    cfg = load_dataset_config(args.dataset_config)
    defaults = cfg.get("defaults", {})
    value = getattr(args, key.replace("-", "_"), None) or defaults.get(key) or fallback
    return repo_path(value)
