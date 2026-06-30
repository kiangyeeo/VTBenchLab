import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from resize_rec import restore_original, smart_padding


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_MODEL_ZOO = os.path.join(REPO_ROOT, "tokenizer_modelzoo")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "tokbench_data")
DEFAULT_RECON_ROOT = os.path.join(REPO_ROOT, "image_reconstruction_results")

# Code lives in the cloned repo (image_scripts/UniFlow), weights + config in the modelzoo.
DEFAULT_UNIFLOW_PATH = os.path.join(SCRIPT_DIR, "UniFlow")
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_MODEL_ZOO, "uniflow")

# UniFlow uses ImageNet normalization and a fixed 448x448 input (patch 14 -> 32x32 tokens).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
UNIFLOW_INPUT_SIZE = 448

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def get_args_parser():
    parser = argparse.ArgumentParser("UniFlow tokenizer reconstruction", add_help=False)
    parser.add_argument("--image_path", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "images", "text_data", "ic13"))
    parser.add_argument("--save_path", type=str, default=os.path.join(DEFAULT_RECON_ROOT, "uniflow", "text_data", "ic13"))
    parser.add_argument("--model_name", type=str, default="uniflow")
    parser.add_argument("--uniflow_path", type=str, default=DEFAULT_UNIFLOW_PATH,
                        help="UniFlow repo root that contains the `uniflow/` package (modeling code).")
    parser.add_argument("--config_path", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory with config.json (the downloaded modelzoo/uniflow dir).")
    parser.add_argument("--ckpt_path", type=str, default=os.path.join(DEFAULT_MODEL_DIR, "model.safetensors"))
    parser.add_argument("--input_size", type=int, default=UNIFLOW_INPUT_SIZE)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=sorted(_DTYPES))
    parser.add_argument("--padding_size", type=int, default=256)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser


def split_list(input_list, chunk_size):
    return [input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def require_path(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_uniflow(args, device="cuda", dtype=torch.bfloat16):
    """Build UniFlowVisionModel and load the released safetensors weights.

    `args` only needs .uniflow_path (repo root with the `uniflow/` package),
    .config_path (dir holding config.json) and .ckpt_path (model.safetensors).
    Returns the eval()-ready model on `device`/`dtype`. Shared with the k-NN script.
    """
    uniflow_path = os.path.abspath(args.uniflow_path)
    config_path = os.path.abspath(args.config_path)
    ckpt_path = os.path.abspath(args.ckpt_path)
    require_path(os.path.join(uniflow_path, "uniflow", "modeling_uniflow.py"), "UniFlow code directory")
    require_path(os.path.join(config_path, "config.json"), "UniFlow config.json")
    require_path(ckpt_path, "UniFlow checkpoint")

    # Put the repo root (not the package dir) on sys.path so `uniflow` imports as a
    # package and its relative imports resolve. flash_attention isn't importable this
    # way, so the model auto-falls back to its naive attention (no flash-attn needed).
    if uniflow_path not in sys.path:
        sys.path.insert(0, uniflow_path)
    from uniflow.configuration_uniflow import UniFlowVisionConfig
    from uniflow.modeling_uniflow import UniFlowVisionModel
    from safetensors.torch import load_file

    config = UniFlowVisionConfig.from_pretrained(config_path)
    model = UniFlowVisionModel(config)
    state_dict = load_file(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[uniflow] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[uniflow] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    model = model.to(device=device, dtype=dtype).eval()
    return model, config


def build_transform(input_size):
    # The padded image is already a square canvas, so a plain resize to input_size
    # reproduces UniFlow's center_crop_arr(.) preprocessing for square inputs.
    return transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denorm_to_pil(recon, mean, std):
    """[3,H,W] normalized tensor -> uint8 PIL image (undo ImageNet normalization)."""
    img = recon.float() * std + mean
    img = img.clamp(0, 1).mul(255).round().to(torch.uint8)
    img = img.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(img)


def main(args):
    image_save_pth = f"{args.save_path}_{args.padding_size}"
    os.makedirs(image_save_pth, exist_ok=True)

    device = "cuda"
    dtype = _DTYPES[args.dtype]
    model, _ = load_uniflow(args, device=device, dtype=dtype)
    transform = build_transform(args.input_size)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)

    all_datas = sorted(os.listdir(args.image_path))
    chunked_filenames = np.array_split(all_datas, args.num_chunks)
    subset = chunked_filenames[args.chunk_idx].tolist()
    chunk_inputs = split_list(subset, args.batch_size)

    for chunk in tqdm(chunk_inputs):
        tensors, metas, names = [], [], []
        for filename in chunk:
            original_img = Image.open(os.path.join(args.image_path, filename)).convert("RGB")
            padded_img, meta = smart_padding(original_img, (args.padding_size, args.padding_size))
            tensors.append(transform(padded_img))
            metas.append(meta)
            names.append(filename)

        input_img = torch.stack(tensors, dim=0).to(device=device, dtype=dtype)
        with torch.no_grad():
            recons = model(input_img)

        for recon, meta, filename in zip(recons, metas, names):
            rec_img = denorm_to_pil(recon, mean, std)
            if rec_img.size != (args.padding_size, args.padding_size):
                rec_img = rec_img.resize((args.padding_size, args.padding_size), Image.LANCZOS)
            final_img = restore_original(rec_img, meta)
            final_img.save(os.path.join(image_save_pth, filename))

    print(args.chunk_idx, " is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("image path check script", parents=[get_args_parser()])
    main(parser.parse_args())
