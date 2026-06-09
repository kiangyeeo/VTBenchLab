import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from resize_rec import restore_original, smart_padding


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_MODEL_ZOO = os.path.join(REPO_ROOT, "tokenizer_modelzoo")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "tokbench_data")
DEFAULT_RECON_ROOT = os.path.join(REPO_ROOT, "image_reconstruction_results")


def get_args_parser():
    parser = argparse.ArgumentParser("Diffusers image VAE reconstruction", add_help=False)
    parser.add_argument("--image_path", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "images", "text_data", "ic13"))
    parser.add_argument("--save_path", type=str, default=os.path.join(DEFAULT_RECON_ROOT, "sd3p5", "text_data", "ic13"))
    parser.add_argument("--model_name", type=str, default="sd3p5", choices=["sdxl", "sd3p5", "flux1"])
    parser.add_argument("--model_path", type=str, default=None, help="Optional local VAE path; overrides defaults")
    parser.add_argument("--padding_size", type=int, default=256)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser


def split_list(input_list, chunk_size):
    return [input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def tensor_to_pilimg(img: torch.Tensor):
    img = (img / 2.0 + 0.5).clamp(0, 1)
    img = img.mul(255).round().nan_to_num(128, 0, 255).clamp(0, 255)
    img = img.to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(img)


def require_path(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_vae(args):
    from diffusers import AutoencoderKL

    model_path_dict = {
        "sdxl": os.path.join(DEFAULT_MODEL_ZOO, "sdxl-vae"),
        "sd3p5": os.path.join(DEFAULT_MODEL_ZOO, "sd3p5-large-vae", "vae"),
        "flux1": os.path.join(DEFAULT_MODEL_ZOO, "flux1-vae", "vae"),
    }
    model_path = os.path.abspath(args.model_path or model_path_dict[args.model_name])
    require_path(model_path, f"{args.model_name} VAE directory")
    vae = AutoencoderKL.from_pretrained(model_path)
    vae.to(dtype=torch.float32)
    vae.to(device="cuda")
    return vae.eval()


def main(args):
    image_save_pth = f"{args.save_path}_{args.padding_size}"
    os.makedirs(image_save_pth, exist_ok=True)

    vae = load_vae(args)

    all_datas = sorted(os.listdir(args.image_path))
    chunked_filenames = np.array_split(all_datas, args.num_chunks)
    subset = chunked_filenames[args.chunk_idx].tolist()
    chunk_inputs = split_list(subset, args.batch_size)

    for chunk in tqdm(chunk_inputs):
        image_path = os.path.join(args.image_path, chunk[0])
        original_img = Image.open(image_path).convert("RGB")
        padded_img, meta = smart_padding(original_img, (args.padding_size, args.padding_size))

        img_tensor = pil_to_tensor(padded_img).unsqueeze(0).float() / 255.0
        input_img = (img_tensor * 2.0 - 1.0).to(device=vae.device, dtype=vae.dtype)

        with torch.no_grad():
            latent = vae.encode(input_img, return_dict=False)[0].sample()
            rec_img = vae.decode(latent).sample.squeeze(0).cpu().detach()

        rec_img = tensor_to_pilimg(rec_img)
        final_img = restore_original(rec_img, meta)
        final_img.save(os.path.join(image_save_pth, chunk[0]))

    print(args.chunk_idx, " is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("image path check script", parents=[get_args_parser()])
    main(parser.parse_args())
