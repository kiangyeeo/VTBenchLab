#!/usr/bin/env python
"""SUN397 configuration for the shared tokenizer dataset-probe driver."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[2]
DRIVER_PATH = (
    WORKSPACE / "scripts" / "linear_probe_tokenizers" / "dataset_probe_driver.py"
)

_spec = spec_from_file_location("tokenizer_dataset_probe_driver_sun397", DRIVER_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load the dataset-probe driver from {DRIVER_PATH}")
driver = module_from_spec(_spec)
_spec.loader.exec_module(driver)


if __name__ == "__main__":
    sys.exit(driver.main())
