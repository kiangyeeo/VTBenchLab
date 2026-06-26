import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from resize_rec import restore_original, smart_padding


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_MODEL_ZOO = os.path.join(REPO_ROOT, "tokenizer_modelzoo")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "tokbench_data")
DEFAULT_RECON_ROOT = os.path.join(REPO_ROOT, "image_reconstruction_results")
DEFAULT_TOKLIP_DIR = os.path.join(DEFAULT_MODEL_ZOO, "TokLIP")

MODEL_CONFIGS = {
    "toklip_s": {
        "toklip_ckpt": os.path.join(DEFAULT_TOKLIP_DIR, "TokLIP_S_256.pt"),
        "image_size": 256,
    },
    "toklip_l": {
        "toklip_ckpt": os.path.join(DEFAULT_TOKLIP_DIR, "TokLIP_L_384.pt"),
        "image_size": 384,
    },
}


def get_args_parser(default_model_name):
    parser = argparse.ArgumentParser("TokLIP VQ reconstruction", add_help=False)
    parser.add_argument("--image_path", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "images", "text_data", "ic13"))
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join(DEFAULT_RECON_ROOT, default_model_name, "text_data", "ic13"),
    )
    parser.add_argument("--model_name", type=str, default=default_model_name, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--toklip_path", type=str, default=os.path.join(SCRIPT_DIR, "TokLIP"))
    parser.add_argument("--toklip_ckpt_path", type=str, default=None)
    parser.add_argument("--vq_ckpt_path", type=str, default=os.path.join(DEFAULT_TOKLIP_DIR, "vq_ds16_t2i.pt"))
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


def load_toklip_vq(args, device):
    toklip_path = os.path.abspath(args.toklip_path)
    toklip_src = os.path.join(toklip_path, "src")
    tokenizer_path = os.path.join(toklip_src, "tokenizer")
    vq_ckpt_path = os.path.abspath(args.vq_ckpt_path)
    toklip_ckpt_path = os.path.abspath(args.toklip_ckpt_path or MODEL_CONFIGS[args.model_name]["toklip_ckpt"])

    # TokLIP-S/L share this VQ tokenizer; the S/L checkpoint check keeps variant setup explicit.
    require_path(tokenizer_path, "TokLIP tokenizer source directory")
    require_path(toklip_ckpt_path, f"{args.model_name} TokLIP checkpoint")
    require_path(vq_ckpt_path, "TokLIP LlamaGen VQ checkpoint")

    sys.path.insert(0, toklip_src)
    from tokenizer.vq_model import VQ_models

    vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
    vq_model.to(device)
    vq_model.eval()

    checkpoint = torch.load(vq_ckpt_path, map_location="cpu")
    if "ema" in checkpoint:
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise ValueError(f"Could not find VQ weights in checkpoint: {vq_ckpt_path}")
    vq_model.load_state_dict(model_weight)
    return vq_model


def tensor_to_pilimg(img):
    img = torch.clamp(127.5 * img + 128.0, 0, 255)
    img = img.permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy()
    return Image.fromarray(img)


def main(args):
    image_save_pth = f"{args.save_path}_{args.padding_size}"
    os.makedirs(image_save_pth, exist_ok=True)

    device = "cuda"
    vq_model = load_toklip_vq(args, device)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    all_datas = sorted(os.listdir(args.image_path))
    chunked_filenames = np.array_split(all_datas, args.num_chunks)
    subset = chunked_filenames[args.chunk_idx].tolist()
    chunk_inputs = split_list(subset, args.batch_size)

    for chunk in tqdm(chunk_inputs):
        inputs = []
        metas = []
        names = []
        for filename in chunk:
            image_path = os.path.join(args.image_path, filename)
            original_img = Image.open(image_path).convert("RGB")
            padded_img, meta = smart_padding(original_img, (args.padding_size, args.padding_size))
            inputs.append(transform(padded_img))
            metas.append(meta)
            names.append(filename)

        input_img = torch.stack(inputs, dim=0).to(device)
        with torch.no_grad():
            quant, _, _ = vq_model.encode(input_img)
            samples = vq_model.decode(quant)

        for sample, meta, filename in zip(samples, metas, names):
            rec_img = tensor_to_pilimg(sample)
            final_img = restore_original(rec_img, meta)
            final_img.save(os.path.join(image_save_pth, filename))

    print(args.chunk_idx, " is done")
