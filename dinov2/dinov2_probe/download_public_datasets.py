import argparse

from .common import add_common_paths, configured_root, load_dataset_config
from .datasets import build_dataset
from .model_zoo import default_image_transform


def main() -> int:
    parser = argparse.ArgumentParser(description="Download public torchvision/CLIP benchmark datasets where supported.")
    add_common_paths(parser)
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset aliases. Defaults to supported public datasets.")
    args = parser.parse_args()
    cfg = load_dataset_config(args.dataset_config)
    data_root = configured_root(args, "data_root", "data/dinov2_baseline")
    names = args.datasets.split(",") if args.datasets else cfg.get("linear_evaluation_sets", [])
    transform = default_image_transform()
    for name in names:
        name = name.strip()
        if not name:
            continue
        spec = cfg["datasets"][name]
        if spec.get("restricted"):
            print(f"[skip restricted] {name}: use check_restricted_data.sh for local layout instructions.")
            continue
        if spec["type"] not in {"clip_benchmark"}:
            print(f"[manual] {name}: expected under {data_root / spec.get('root', name)}")
            continue
        for split in {spec.get("train_split", "train"), spec.get("eval_split", "test")}:
            try:
                print(f"[download/check] {name} split={split}")
                build_dataset(
                    name,
                    split,
                    transform=transform,
                    data_root=data_root,
                    config_path=args.dataset_config,
                    download=True,
                    smoke=True,
                )
            except Exception as exc:
                print(f"[warn] {name} split={split}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

