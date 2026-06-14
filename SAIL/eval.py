from evaluation import (
    imagenet_eval,
    winoground_eval,
    coco_eval,
    update_results_json,
    extract_info_from_path,
    check_epoch_exists,
)
import argparse
from model import create_model
import os
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation")
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "imagenetv2",
            "COCO",
            "winoground",
            "imagenetv1",
            "segmentation",
            "MMVP",
        ],
        default="imagenetv1",
        help="Task",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default=None,
        help="e.g sentence-transformers/all-mpnet-base-v2. If provided, will load a text model and use it for text embeddings.",
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default=None,
        help="e.g facebook/dinov2-base. If provided, will load a vision model and use it for image embeddings.",
    )
    parser.add_argument(
        "--head-weights-path",
        type=str,
        default=None,
        # required=True,
        help="Path to head weight",
    )
    parser.add_argument(
        "--linear-type",
        type=str,
        default="star",
        help="Type of linear layer to use.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="evaluation/eval_result",
        help="Path to results file",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        required=True,
        help="Path to images",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="evaluation/backbone_features",
        help="Path to images",
    )
    parser.add_argument(
        "--target-dimension",
        type=int,
        default=512,
        help="Dimension of text embeddings. Default set to 768 for all-mpnet-base-v2.",
    ),
    parser.add_argument(
        "--agg_mode",
        type=str,
        default='concat',
        help="Aggregation mode for image features.",
    ),
    parser.add_argument(
        "--width_factor",
        type=int,
        default=8,
        help="Width factor for the MLP.",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite existing results.",
    )
    parser.add_argument(
        "--seg_task_config",
        type=str,
        default="evaluation/ClearCLIP/configs/cfg_ade20k.py",
        help="Task for segmentation evaluation",
    )

    parser.add_argument(
        "--visualize_segmentation",
        default=False,
        action="store_true",
        help="Visualize segmentation results.",
    )
    parser.add_argument(
        "--sharelock",
        default=False,
        action="store_true",
        help="Use sharelock.",
    )
    args = parser.parse_args()

    # Overide args with model_config.yaml
    config_file = os.path.join(
        os.path.dirname(os.path.dirname(args.head_weights_path)), "model_config.yaml"
    )
    if os.path.exists(config_file):
        print(f"Loading model config from {config_file}")
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    return args


def main(args):
    

    epoch_num, training_info_str, model_prefix = extract_info_from_path(
        args
    )
    output_path = os.path.join(
        args.results_dir,
        args.task,
        model_prefix,
        f"{training_info_str}.json",
    )
    
    if check_epoch_exists(output_path, epoch_num) and not args.overwrite:
        print(f"Epoch {epoch_num} already exists in {args.task}, skipping.")
        return None
    elif check_epoch_exists(output_path, epoch_num) and args.overwrite:
        print(f"Epoch {epoch_num} already exists in {args.task}, overwriting.")
    model = create_model(
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
        head_weights_path=args.head_weights_path,
        linear_type=args.linear_type,
        target_dimension=args.target_dimension,
        device=args.device,
        agg_mode=args.agg_mode,
        sharelock=args.sharelock,
        width_factor=args.width_factor,
    )
    text_model_name = args.text_model.split("/")[-1]
    vision_model_name = args.vision_model.split("/")[-1]
    model.eval()

    # eval
    if args.task.lower() == "imagenetv1":
        # Check if the ImageNet folders exist, if not create them and download the data
        imagenet_dir = os.path.join(args.dataset_root_dir, "imagenet")
        if not os.path.exists(imagenet_dir):
            os.makedirs(imagenet_dir)
            print(f"Created directory: {imagenet_dir}")

            # Download ImageNet validation images
            val_url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
            val_file = os.path.join(imagenet_dir, "ILSVRC2012_img_val.tar")
            if not os.path.exists(val_file):
                print(f"Downloading ImageNet validation images to {val_file}")
                os.system(f"wget {val_url} -O {val_file}")
            else:
                print(f"ImageNet validation images already exist at {val_file}")

            # Download ImageNet devkit
            devkit_url = (
                "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
            )
            devkit_file = os.path.join(imagenet_dir, "ILSVRC2012_devkit_t12.tar.gz")
            if not os.path.exists(devkit_file):
                print(f"Downloading ImageNet devkit to {devkit_file}")
                os.system(f"wget {devkit_url} -O {devkit_file}")
            else:
                print(f"ImageNet devkit already exists at {devkit_file}")
        results = imagenet_eval(
            model,
            bs=args.batch_size,
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            images_dir=imagenet_dir,
            save_dir=args.save_dir,
            version="v1",
        )
    elif args.task.lower() == "imagenetv2":
        results = imagenet_eval(
            model,
            bs=args.batch_size,
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            images_dir=args.dataset_root_dir,
            save_dir=args.save_dir,
            version="v2",
        )
    elif args.task.lower() == "coco":

        coco_root = os.path.join(args.dataset_root_dir, "coco", "2017", "val2017")
        coco_ann_file = os.path.join(
            args.dataset_root_dir,
            "coco",
            "2017",
            "annotations",
            "captions_val2017.json",
        )
        assert os.path.exists(coco_root), f"COCO root directory does not exist: {coco_root}"
        assert os.path.exists(coco_ann_file), f"COCO annotation file does not exist: {coco_ann_file}"
        if args.agg_mode != 'concat':
            vision_model_name = vision_model_name + '_' + args.agg_mode
        results = coco_eval(
            model,
            bs=args.batch_size,
            coco_root=coco_root,
            coco_ann_file=coco_ann_file,
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            k_vals=[1, 5, 10],
            save_dir=args.save_dir,
        )
    elif args.task.lower() == "winoground":
        results = winoground_eval(
            model,
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            save_dir=args.save_dir,
        )
    elif args.task.lower() == "segmentation":
        results = segmentation_eval(
            text_model_name=args.text_model,
            vision_model_name=args.vision_model,
            head_weights_path=args.head_weights_path,
            linear_type=args.linear_type,
            target_dimension=args.target_dimension,
            device=args.device,
            task_config=args.seg_task_config,
            save_dir=args.save_dir,
            visualize=args.visualize_segmentation,
            # precision='fp16',
        )
    elif args.task.lower() == "mmvp":
        from evaluation import mmvp_eval
        mmvp_dir = "evaluation/MMVP_VLM"
        results = mmvp_eval(
            model,
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            directory=mmvp_dir,
        )
    update_results_json(output_path, epoch_num, results)


if __name__ == "__main__":
    args = parse_args()
    if args.task.lower() == "segmentation":
        try:
            from evaluation.seg_eval import segmentation_eval
        except ImportError as e:
            print(f"Segmentation evaluation not available: {e}")
            exit()
    main(args)
