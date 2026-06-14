import argparse

from .common import add_common_paths, configured_root, load_dataset_config, load_model_config, split_csv
from .datasets import linear_dataset_names
from .features import extract_features


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache frozen image/video features.")
    add_common_paths(parser)
    parser.add_argument("--models", default=None, help="Comma-separated model aliases.")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset aliases.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dataset-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    args = parser.parse_args()

    model_cfg = load_model_config(args.model_config)
    dataset_cfg = load_dataset_config(args.dataset_config)
    models = split_csv(args.models, model_cfg["models"].keys())
    datasets = split_csv(args.datasets, linear_dataset_names(args.dataset_config))
    data_root = configured_root(args, "data_root", "data/dinov2_baseline")
    feature_root = configured_root(args, "feature_root", "outputs/dinov2_baseline/features")
    model_root = configured_root(args, "model_root", "checkpoints/dinov2_baseline")

    for model in models:
        for dataset in datasets:
            split = args.split
            spec = dataset_cfg["datasets"][dataset]
            if split == "train":
                split = spec.get("train_split", "train")
            elif split in {"test", "eval", "val"}:
                split = spec.get("eval_split", "test")
            extract_features(
                model_alias=model,
                dataset_name=dataset,
                split=split,
                data_root=data_root,
                feature_root=feature_root,
                model_root=model_root,
                model_config=args.model_config,
                dataset_config=args.dataset_config,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                allow_download=args.allow_download,
                dataset_download=args.dataset_download,
                overwrite=args.overwrite,
                smoke=args.smoke,
                device=args.device,
                amp_dtype=args.amp_dtype,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
