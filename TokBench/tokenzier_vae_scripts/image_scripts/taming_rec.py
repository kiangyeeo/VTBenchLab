import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from resize_rec import smart_padding, restore_original


torch.set_grad_enabled(False)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_MODEL_ZOO = os.path.join(REPO_ROOT, "tokenizer_modelzoo")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "tokbench_data")
DEFAULT_RECON_ROOT = os.path.join(REPO_ROOT, "image_reconstruction_results")
DEFAULT_VQGAN_DIR = os.path.join(DEFAULT_MODEL_ZOO, "taming_vqgan_imagenet_f16_16384")


def get_args_parser():
    parser = argparse.ArgumentParser("Taming VQGAN reconstruction", add_help=False)
    parser.add_argument("--image_path", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "images", "text_data", "ic13"))
    parser.add_argument("--save_path", type=str, default=os.path.join(DEFAULT_RECON_ROOT, "vqgan", "text_data", "ic13"))
    parser.add_argument("--taming_path", type=str, default=os.path.join(SCRIPT_DIR, "taming-transformers"))
    parser.add_argument("--config_path", type=str, default=os.path.join(DEFAULT_VQGAN_DIR, "model.yaml"))
    parser.add_argument("--ckpt_path", type=str, default=os.path.join(DEFAULT_VQGAN_DIR, "last.ckpt"))
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


def load_vqgan(config, model_cls, ckpt_path=None):
    model = model_cls(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
    return model.eval()


def preprocess_vqgan(x):
    return 2.0 * x - 1.0


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if x.mode != "RGB":
        x = x.convert("RGB")
    return x


def load_model(args, device):
    taming_path = os.path.abspath(args.taming_path)
    config_path = os.path.abspath(args.config_path)
    ckpt_path = os.path.abspath(args.ckpt_path)
    require_path(taming_path, "taming-transformers code directory")
    require_path(config_path, "Taming VQGAN config")
    require_path(ckpt_path, "Taming VQGAN checkpoint")

    sys.path.insert(0, taming_path)
    from omegaconf import OmegaConf
    from taming.models.vqgan import VQModel

    config = OmegaConf.load(config_path)
    model = load_vqgan(config, VQModel, ckpt_path=ckpt_path).to(device)
    return model


def main(args):
    padding_size = args.padding_size
    image_save_pth = "{}_{}".format(args.save_path, str(padding_size))
    os.makedirs(image_save_pth, exist_ok=True)

    device = "cuda"
    model = load_model(args, device)

    transform = transforms.Compose([transforms.ToTensor()])

    all_datas = sorted(os.listdir(args.image_path))
    chunked_filenames = np.array_split(all_datas, args.num_chunks)
    subset = chunked_filenames[args.chunk_idx].tolist()
    chunk_inputs = split_list(subset, args.batch_size)

    for chunk in tqdm(chunk_inputs):
        image_path = os.path.join(args.image_path, chunk[0])
        original_img = Image.open(image_path).convert("RGB")
        padded_img, meta = smart_padding(original_img, (padding_size, padding_size))

        input_img = transform(padded_img).unsqueeze(0).to(device)
        with torch.no_grad():
            z, _, [_, _, indices] = model.encode(preprocess_vqgan(input_img))
            xrec = model.decode(z)

        rec_img = custom_to_pil(xrec[0])
        final_img = restore_original(rec_img, meta)
        final_img.save("{}/{}".format(image_save_pth, chunk[0]))

    print(args.chunk_idx, " is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("image path check script", parents=[get_args_parser()])
    main(parser.parse_args())
