import argparse
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

from resize_rec import smart_padding, restore_original


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_MODEL_ZOO = os.path.join(REPO_ROOT, "tokenizer_modelzoo")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "tokbench_data")
DEFAULT_RECON_ROOT = os.path.join(REPO_ROOT, "image_reconstruction_results")
DEFAULT_CHAMELEON_DIR = os.path.join(DEFAULT_MODEL_ZOO, "chameleon")


def get_args_parser():
    parser = argparse.ArgumentParser("Chameleon VQGAN reconstruction", add_help=False)
    parser.add_argument("--image_path", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "images", "text_data", "ic13"))
    parser.add_argument("--save_path", type=str, default=os.path.join(DEFAULT_RECON_ROOT, "chameleon", "text_data", "ic13"))
    parser.add_argument("--chameleon_path", type=str, default=os.path.join(SCRIPT_DIR, "chameleon"))
    parser.add_argument("--cfg_path", type=str, default=os.path.join(DEFAULT_CHAMELEON_DIR, "vqgan.yaml"))
    parser.add_argument("--ckpt_path", type=str, default=os.path.join(DEFAULT_CHAMELEON_DIR, "vqgan.ckpt"))
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


def load_image_tokenizer(args):
    chameleon_path = os.path.abspath(args.chameleon_path)
    cfg_path = os.path.abspath(args.cfg_path)
    ckpt_path = os.path.abspath(args.ckpt_path)
    require_path(chameleon_path, "chameleon code directory")
    require_path(cfg_path, "Chameleon VQGAN config")
    require_path(ckpt_path, "Chameleon VQGAN checkpoint")

    sys.path.insert(0, chameleon_path)
    from chameleon.inference.image_tokenizer import ImageTokenizer

    return ImageTokenizer(cfg_path=cfg_path, ckpt_path=ckpt_path, device="cuda")


def main(args):
    padding_size = args.padding_size
    image_save_pth = "{}_{}".format(args.save_path, str(padding_size))
    os.makedirs(image_save_pth, exist_ok=True)

    image_tokenizer = load_image_tokenizer(args)

    all_datas = sorted(os.listdir(args.image_path))
    chunked_filenames = np.array_split(all_datas, args.num_chunks)
    subset = chunked_filenames[args.chunk_idx].tolist()
    chunk_inputs = split_list(subset, args.batch_size)

    for chunk in tqdm(chunk_inputs):
        image_path = os.path.join(args.image_path, chunk[0])
        original_img = Image.open(image_path).convert("RGB")
        padded_img, meta = smart_padding(original_img, (padding_size, padding_size))

        vq_code = image_tokenizer.img_tokens_from_pil(padded_img)
        feature_size = padding_size // 16
        rec_img = image_tokenizer.pil_from_img_toks(vq_code, h_latent_dim=feature_size, w_latent_dim=feature_size)
        final_img = restore_original(rec_img, meta)
        final_img.save("{}/{}".format(image_save_pth, chunk[0]))

    print(args.chunk_idx, " is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("image path check script", parents=[get_args_parser()])
    main(parser.parse_args())
