import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

from resize_rec import smart_padding, restore_original


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_MODEL_ZOO = os.path.join(REPO_ROOT, "tokenizer_modelzoo")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "tokbench_data")
DEFAULT_RECON_ROOT = os.path.join(REPO_ROOT, "image_reconstruction_results")


def get_args_parser():
    parser = argparse.ArgumentParser("VAR VAE reconstruction", add_help=False)
    parser.add_argument("--image_path", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "images", "text_data", "ic13"))
    parser.add_argument("--save_path", type=str, default=os.path.join(DEFAULT_RECON_ROOT, "var", "text_data", "ic13"))
    parser.add_argument("--var_path", type=str, default=os.path.join(SCRIPT_DIR, "VAR"))
    parser.add_argument("--ckpt_path", type=str, default=os.path.join(DEFAULT_MODEL_ZOO, "var", "vae_ch160v4096z32.pth"))
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


def tensor_to_pilimg(img: torch.Tensor):
    img = img.add(1).mul_(0.5 * 255).round().nan_to_num_(128, 0, 255).clamp_(0, 255)
    img = img.to(dtype=torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    img = Image.fromarray(img[0])
    return img


def load_var_vae(args, patch_nums, device):
    var_path = os.path.abspath(args.var_path)
    ckpt_path = os.path.abspath(args.ckpt_path)
    require_path(var_path, "VAR code directory")
    require_path(ckpt_path, "VAR VAE checkpoint")

    sys.path.insert(0, var_path)
    from models import VQVAE

    vae_local = VQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=160,
        test_mode=True,
        share_quant_resi=4,
        v_patch_nums=patch_nums,
    ).to(device)
    vae_local.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    vae_local.eval()
    return vae_local


def main(args):
    padding_size = args.padding_size
    image_save_pth = "{}_{}".format(args.save_path, str(padding_size))
    os.makedirs(image_save_pth, exist_ok=True)

    if padding_size == 256:
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    elif padding_size == 512:
        patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
    elif padding_size == 1024:
        patch_nums = (1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64)
    else:
        assert False, "unsupport size for VAR"

    device = "cuda"
    vae_local = load_var_vae(args, patch_nums, device)

    from utils.data import normalize_01_into_pm1

    preprocess = transforms.Compose([transforms.ToTensor()])

    all_datas = sorted(os.listdir(args.image_path))
    chunked_filenames = np.array_split(all_datas, args.num_chunks)
    subset = chunked_filenames[args.chunk_idx].tolist()
    chunk_inputs = split_list(subset, args.batch_size)

    for chunk in tqdm(chunk_inputs):
        image_path = os.path.join(args.image_path, chunk[0])
        original_img = Image.open(image_path).convert("RGB")
        padded_img, meta = smart_padding(original_img, (padding_size, padding_size))
        input_img = preprocess(padded_img).unsqueeze(0).to(device)
        input_img = normalize_01_into_pm1(input_img)
        with torch.no_grad():
            rec_img = vae_local.img_to_reconstructed_img(input_img, patch_nums, last_one=True)
            rec_img = tensor_to_pilimg(rec_img)
        final_img = restore_original(rec_img, meta)
        final_img.save("{}/{}".format(image_save_pth, chunk[0]))

    print(args.chunk_idx, " is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("image path check script", parents=[get_args_parser()])
    main(parser.parse_args())
