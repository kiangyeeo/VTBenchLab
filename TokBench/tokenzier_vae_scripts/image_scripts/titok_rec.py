import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from resize_rec import smart_padding, restore_original


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_MODEL_ZOO = os.path.join(REPO_ROOT, "tokenizer_modelzoo")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "tokbench_data")
DEFAULT_RECON_ROOT = os.path.join(REPO_ROOT, "image_reconstruction_results")
DEFAULT_CKPT = os.path.join(DEFAULT_MODEL_ZOO, "titok_l32_imagenet")


def get_args_parser():
    parser = argparse.ArgumentParser("TiTok reconstruction", add_help=False)
    parser.add_argument("--image_path", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "images", "text_data", "ic13"))
    parser.add_argument("--save_path", type=str, default=os.path.join(DEFAULT_RECON_ROOT, "titok_l32", "text_data", "ic13"))
    parser.add_argument("--titok_path", type=str, default=os.path.join(SCRIPT_DIR, "1d-tokenizer"))
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT)
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


def load_titok(args, device):
    titok_path = os.path.abspath(args.titok_path)
    ckpt_path = os.path.abspath(args.ckpt_path)
    require_path(titok_path, "1d-tokenizer (TiTok) code directory")
    require_path(ckpt_path, "TiTok checkpoint directory")

    sys.path.insert(0, titok_path)
    from modeling.titok import TiTok

    titok_tokenizer = TiTok.from_pretrained(ckpt_path)
    titok_tokenizer.eval()
    titok_tokenizer.requires_grad_(False)
    titok_tokenizer = titok_tokenizer.to(device)
    return titok_tokenizer


def main(args):
    padding_size = args.padding_size
    image_save_pth = "{}_{}".format(args.save_path, str(padding_size))
    os.makedirs(image_save_pth, exist_ok=True)

    device = "cuda"
    titok_tokenizer = load_titok(args, device)

    all_datas = sorted(os.listdir(args.image_path))
    chunked_filenames = np.array_split(all_datas, args.num_chunks)
    subset = chunked_filenames[args.chunk_idx].tolist()
    chunk_inputs = split_list(subset, args.batch_size)

    for chunk in tqdm(chunk_inputs):
        image_path = os.path.join(args.image_path, chunk[0])
        original_img = Image.open(image_path).convert("RGB")
        padded_img, meta = smart_padding(original_img, (padding_size, padding_size))

        image = torch.from_numpy(np.array(padded_img).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            encoded_tokens = titok_tokenizer.encode(image.to(device))[1]["min_encoding_indices"]
            reconstructed_image = titok_tokenizer.decode_tokens(encoded_tokens)
            reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
            reconstructed_image = (reconstructed_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
            rec_img = Image.fromarray(reconstructed_image)

        final_img = restore_original(rec_img, meta)
        final_img.save("{}/{}".format(image_save_pth, chunk[0]))

    print(args.chunk_idx, " is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("image path check script", parents=[get_args_parser()])
    main(parser.parse_args())
