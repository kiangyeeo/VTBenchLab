import argparse
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from resize_rec import restore_original, smart_padding


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MODEL_ROOT = REPO_ROOT / "tokenizer_modelzoo" / "RAEv2-models"
DEFAULT_DATA_ROOT = REPO_ROOT / "tokbench_data"
DEFAULT_RECON_ROOT = REPO_ROOT / "image_reconstruction_results"

DINOV3_L_CKPT = "encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

TOKENIZER_CONFIGS = {
    "raev2": {
        "encoder_name": (
            "dinov3mls-vit-l16"
            "[layers=1.2.3.4.5.6.7.8.9.10.11.12.13.14.15.16.17.18.19.20.21.22.23]"
        ),
        "encoder_checkpoint": DINOV3_L_CKPT,
        "encoder_checkpoint_size": 1_213_050_671,
        "decoder_checkpoint": "stage1/imagenet/dinov3l-k23/decoder.pt",
        "decoder_checkpoint_size": 1_662_766_063,
        "stats_checkpoint": "stage1/imagenet/dinov3l-k23/stats.pt",
        "stats_checkpoint_size": 2_098_901,
        "uses_dinov3": True,
    },
    "dinov3": {
        "encoder_name": "dinov3-vit-l16",
        "encoder_checkpoint": DINOV3_L_CKPT,
        "encoder_checkpoint_size": 1_213_050_671,
        "decoder_checkpoint": "stage1/imagenet/dinov3l-k1/decoder.pt",
        "decoder_checkpoint_size": 1_662_766_063,
        "stats_checkpoint": "stage1/imagenet/dinov3l-k1/stats.pt",
        "stats_checkpoint_size": 2_098_901,
        "uses_dinov3": True,
    },
    "ijepa": {
        "encoder_name": "jepa-vit-h",
        "encoder_checkpoint": "encoders/ijepa/ijepa_vith.pth",
        "encoder_checkpoint_size": 10_358_004_345,
        "decoder_checkpoint": "stage1/imagenet/jepa-h-k1/decoder.pt",
        "decoder_checkpoint_size": 1_663_945_711,
        "stats_checkpoint": "stage1/imagenet/jepa-h-k1/stats.pt",
        "stats_checkpoint_size": 2_623_189,
        "uses_dinov3": False,
    },
}

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def get_args_parser(default_model_name):
    if default_model_name not in TOKENIZER_CONFIGS:
        raise ValueError(f"Unknown tokenizer preset: {default_model_name}")

    parser = argparse.ArgumentParser("RAEv2 Stage-1 reconstruction", add_help=False)
    parser.add_argument(
        "--image_path",
        type=str,
        default=str(DEFAULT_DATA_ROOT / "images" / "text_data" / "ic13"),
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(DEFAULT_RECON_ROOT / default_model_name / "text_data" / "ic13"),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=default_model_name,
        choices=[default_model_name],
    )
    parser.add_argument("--raev2_path", type=str, default=str(SCRIPT_DIR / "RAEv2"))
    parser.add_argument("--dinov3_path", type=str, default=str(SCRIPT_DIR / "dinov3"))
    parser.add_argument("--model_root", type=str, default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument(
        "--padding_size",
        type=int,
        default=256,
        choices=[256],
        help="The released Stage-1 checkpoints are native 256x256 models.",
    )
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser


def require_dir(path, description):
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")


def require_file(path, description, expected_size=None):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if expected_size is not None:
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise OSError(
                f"Incomplete or unexpected {description}: {path} "
                f"(got {actual_size} bytes, expected {expected_size})"
            )


@contextmanager
def temporary_pretrained_layout(model_root):
    """Expose model_root as ./pretrained_models for RAEv2's I-JEPA loader."""
    previous_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="tokbench_raev2_") as tmp_dir:
        pretrained_link = Path(tmp_dir) / "pretrained_models"
        pretrained_link.symlink_to(model_root, target_is_directory=True)
        os.chdir(tmp_dir)
        try:
            yield
        finally:
            os.chdir(previous_cwd)


def resolve_model_files(args):
    config = TOKENIZER_CONFIGS[args.model_name]
    model_root = Path(args.model_root).expanduser().resolve()
    encoder_checkpoint = model_root / config["encoder_checkpoint"]
    decoder_checkpoint = model_root / config["decoder_checkpoint"]
    stats_checkpoint = model_root / config["stats_checkpoint"]

    require_dir(model_root, "RAEv2 model root")
    require_file(
        encoder_checkpoint,
        f"{args.model_name} encoder checkpoint",
        config["encoder_checkpoint_size"],
    )
    require_file(
        decoder_checkpoint,
        f"{args.model_name} decoder checkpoint",
        config["decoder_checkpoint_size"],
    )
    require_file(
        stats_checkpoint,
        f"{args.model_name} normalization statistics",
        config["stats_checkpoint_size"],
    )
    return config, model_root, decoder_checkpoint, stats_checkpoint


def load_tokenizer(args):
    if not torch.cuda.is_available():
        raise RuntimeError("These RAEv2 checkpoints require a CUDA GPU for TokBench reconstruction.")

    config, model_root, decoder_checkpoint, stats_checkpoint = resolve_model_files(args)
    raev2_path = Path(args.raev2_path).expanduser().resolve()
    raev2_src = raev2_path / "src"
    decoder_config = raev2_path / "configs" / "decoder" / "ViTXL"

    require_file(raev2_src / "stage1" / "rae.py", "RAEv2 source code")
    require_file(decoder_config / "config.json", "RAEv2 ViT-XL decoder config")

    if config["uses_dinov3"]:
        dinov3_path = Path(args.dinov3_path).expanduser().resolve()
        require_file(dinov3_path / "hubconf.py", "local DINOv3 repository")
        os.environ["DINOV3_REPO_DIR"] = str(dinov3_path)
        os.environ["DINOV3_CKPT_DIR"] = str(model_root / "encoders" / "dinov3")

    if str(raev2_src) not in sys.path:
        sys.path.insert(0, str(raev2_src))

    from stage1 import RAE

    # JEPAEncoder uses ./pretrained_models internally. A temporary symlink gives
    # it the official layout without copying weights or modifying third-party code.
    with temporary_pretrained_layout(model_root):
        tokenizer = RAE(
            encoder_name=config["encoder_name"],
            resolution=256,
            decoder_config_path=str(decoder_config),
            pretrained_decoder_path=str(decoder_checkpoint),
            normalization_stat_path=str(stats_checkpoint),
            noise_tau=0.0,
        )

    tokenizer.to("cuda")
    tokenizer.requires_grad_(False)
    tokenizer.eval()
    print(
        f"[{args.model_name}] encoder={config['encoder_name']} "
        f"decoder={decoder_checkpoint} stats={stats_checkpoint}"
    )
    return tokenizer


def split_list(items, chunk_size):
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def tensor_to_pil(image):
    image = image.detach().float().clamp(0, 1)
    image = image.mul(255).round().nan_to_num(128, 0, 255).clamp(0, 255)
    array = image.to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def validate_runtime_args(args):
    if args.padding_size != 256:
        raise ValueError(
            f"{args.model_name} uses an official native-256 checkpoint; "
            f"padding_size must be 256, got {args.padding_size}"
        )
    if args.num_chunks < 1:
        raise ValueError(f"num_chunks must be positive, got {args.num_chunks}")
    if not 0 <= args.chunk_idx < args.num_chunks:
        raise ValueError(
            f"chunk_idx must be in [0, {args.num_chunks}), got {args.chunk_idx}"
        )
    if args.batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")


def list_images(image_dir):
    return sorted(
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def main(args):
    validate_runtime_args(args)
    image_dir = Path(args.image_path).expanduser().resolve()
    require_dir(image_dir, "TokBench input image directory")

    all_names = list_images(image_dir)
    if not all_names:
        raise FileNotFoundError(f"No supported images found in {image_dir}")

    subset = np.array_split(all_names, args.num_chunks)[args.chunk_idx].tolist()
    output_dir = Path(f"{args.save_path}_{args.padding_size}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not subset:
        print(f"[{args.model_name}] chunk {args.chunk_idx} is empty; nothing to reconstruct")
        return

    tokenizer = load_tokenizer(args)
    chunks = split_list(subset, args.batch_size)

    for names in tqdm(chunks, desc=f"{args.model_name} chunk {args.chunk_idx}"):
        tensors = []
        metas = []
        for name in names:
            original = Image.open(image_dir / name).convert("RGB")
            padded, meta = smart_padding(original, (256, 256))
            tensors.append(pil_to_tensor(padded).float().div(255.0))
            metas.append(meta)

        images = torch.stack(tensors).to(device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            reconstructions = tokenizer(images).clamp(0, 1)

        for name, reconstruction, meta in zip(names, reconstructions, metas):
            reconstructed_image = tensor_to_pil(reconstruction)
            restored_image = restore_original(reconstructed_image, meta)
            restored_image.save(output_dir / name)

    print(
        f"[{args.model_name}] chunk {args.chunk_idx}/{args.num_chunks} complete "
        f"-> {output_dir}"
    )
