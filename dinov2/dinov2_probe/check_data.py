import argparse
from pathlib import Path

from .common import add_common_paths, configured_root, load_dataset_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local restricted dataset layout.")
    add_common_paths(parser)
    args = parser.parse_args()
    cfg = load_dataset_config(args.dataset_config)
    data_root = configured_root(args, "data_root", "data/dinov2_baseline")
    ok = True
    for name, info in cfg.get("restricted_sets", {}).items():
        print(f"\n[{name}]")
        for rel in info.get("required_paths", []):
            path = data_root / rel
            exists = path.exists()
            ok = ok and exists
            print(f"  {'OK ' if exists else 'MISS'} {path}")
        print(f"  prepare: {info.get('prepare', '')}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

