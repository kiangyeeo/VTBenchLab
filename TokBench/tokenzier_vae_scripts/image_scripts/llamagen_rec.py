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

DEFAULT_CKPTS = {
    "llamagen_vq16": os.path.join(DEFAULT_MODEL_ZOO, "LlamaGen", "vq_ds16_c2i.pt"),
    "llamagen_vq8": os.path.join(DEFAULT_MODEL_ZOO, "LlamaGen", "vq_ds8_c2i.pt"),
}

MODEL_VARIANTS = {
    "llamagen_vq16": "VQ-16",
    "llamagen_vq8": "VQ-8",
}


def get_args_parser():
    parser = argparse.ArgumentParser("LlamaGen VQ reconstruction", add_help=False)
    parser.add_argument("--image_path", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "images", "text_data", "ic13"))
    parser.add_argument("--save_path", type=str, default=os.path.join(DEFAULT_RECON_ROOT, "llamagen_vq16", "text_data", "ic13"))
    parser.add_argument("--model_name", type=str, default="llamagen_vq16", choices=sorted(MODEL_VARIANTS))
    parser.add_argument("--llamagen_path", type=str, default=os.path.join(SCRIPT_DIR, "LlamaGen"))
    parser.add_argument("--ckpt_path", type=str, default=None)
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


def load_llamagen_vq(args, device):
    llamagen_path = os.path.abspath(args.llamagen_path)
    tokenizer_path = os.path.join(llamagen_path, "tokenizer", "tokenizer_image")
    ckpt_path = os.path.abspath(args.ckpt_path or DEFAULT_CKPTS[args.model_name])

    require_path(tokenizer_path, "LlamaGen tokenizer_image directory")
    require_path(ckpt_path, "LlamaGen checkpoint")

    sys.path.insert(0, tokenizer_path)
    from vq_model import VQ_models

    vq_model = VQ_models[MODEL_VARIANTS[args.model_name]](codebook_size=16384, codebook_embed_dim=8)
    vq_model.to(device)
    vq_model.eval()

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "ema" in checkpoint:
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise ValueError(f"Could not find model weights in checkpoint: {ckpt_path}")
    vq_model.load_state_dict(model_weight)
    return vq_model


def main(args):
    image_save_pth = f"{args.save_path}_{args.padding_size}"
    os.makedirs(image_save_pth, exist_ok=True)

    device = "cuda"
    vq_model = load_llamagen_vq(args, device)

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
        image_path = os.path.join(args.image_path, chunk[0])
        original_img = Image.open(image_path).convert("RGB")
        padded_img, meta = smart_padding(original_img, (args.padding_size, args.padding_size))

        input_img = transform(padded_img).unsqueeze(0).to(device)
        with torch.no_grad():
            latent, _, _ = vq_model.encode(input_img)
            samples = vq_model.decode(latent)

        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        rec_img = Image.fromarray(samples[0])
        final_img = restore_original(rec_img, meta)
        final_img.save(os.path.join(image_save_pth, chunk[0]))

    print(args.chunk_idx, " is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("image path check script", parents=[get_args_parser()])
    main(parser.parse_args())
