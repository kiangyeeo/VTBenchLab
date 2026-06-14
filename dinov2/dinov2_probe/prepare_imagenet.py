import argparse
import json
import sys
from pathlib import Path

from torchvision.datasets import ImageFolder

from .common import add_common_paths, configured_root, repo_path, workspace_path


def _load_imagenet_names():
    path = workspace_path("CLIP_benchmark/clip_benchmark/datasets/en_classnames.json")
    with open(path, "r") as f:
        names = json.load(f)["imagenet1k"]
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare DINOv2 ImageNet extra metadata.")
    add_common_paths(parser)
    parser.add_argument("--imagenet-root", default=None)
    parser.add_argument("--extra-root", default=None)
    args = parser.parse_args()

    data_root = configured_root(args, "data_root", "data/dinov2_baseline")
    imagenet_root = Path(args.imagenet_root) if args.imagenet_root else data_root / "imagenet1k"
    extra_root = Path(args.extra_root) if args.extra_root else imagenet_root
    train_root = imagenet_root / "train"
    val_root = imagenet_root / "val"
    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError(f"Expected ImageNet folders at {train_root} and {val_root}")

    names = _load_imagenet_names()
    train = ImageFolder(train_root)
    labels_path = imagenet_root / "labels.txt"
    with open(labels_path, "w") as f:
        for class_name, idx in sorted(train.class_to_idx.items(), key=lambda kv: kv[1]):
            readable = names[idx] if idx < len(names) else class_name
            f.write(f"{class_name},{readable}\n")
    print(f"Wrote {labels_path}")

    sys.path.insert(0, str(repo_path(".")))
    from dinov2.data.datasets import ImageNet

    for split in (ImageNet.Split.TRAIN, ImageNet.Split.VAL):
        ds = ImageNet(split=split, root=str(imagenet_root), extra=str(extra_root))
        ds.dump_extra()
        print(f"Prepared DINOv2 extra metadata for {split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
