import argparse
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from resize_rec import restore_original, smart_padding


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_MODEL_ZOO = os.path.join(REPO_ROOT, "tokenizer_modelzoo")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "tokbench_data")
DEFAULT_RECON_ROOT = os.path.join(REPO_ROOT, "image_reconstruction_results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG_PATHS = {
    "magvit2_imgnet_f16": "configs/Open-MAGVIT2/gpu/imagenet_lfqgan_256_L.yaml",
    "magvit2_imgnet_f8": "configs/Open-MAGVIT2/gpu/imagenet_lfqgan_128_L.yaml",
    "magvit2_pretrain_f16_16384": "configs/Open-MAGVIT2/gpu/pretrain_lfqgan_256_16384.yaml",
    "magvit2_pretrain_f16_262144": "configs/Open-MAGVIT2/gpu/pretrain_lfqgan_256_262144.yaml",
}

DEFAULT_CKPTS = {
    "magvit2_imgnet_f16": os.path.join(
        DEFAULT_MODEL_ZOO, "Open_MAGVIT2", "imagenet_256_L", "imagenet_256_L.ckpt"
    ),
    "magvit2_imgnet_f8": os.path.join(
        DEFAULT_MODEL_ZOO, "Open_MAGVIT2", "imagenet_128_L", "imagenet_128_L.ckpt"
    ),
    "magvit2_pretrain_f16_16384": os.path.join(
        DEFAULT_MODEL_ZOO, "Open_MAGVIT2", "pretrain256_16384", "pretrain256_16384.ckpt"
    ),
    "magvit2_pretrain_f16_262144": os.path.join(
        DEFAULT_MODEL_ZOO, "Open_MAGVIT2", "pretrain256_262144", "pretrain256_262144.ckpt"
    ),
}


def get_args_parser():
    parser = argparse.ArgumentParser("Open-MAGVIT2 reconstruction", add_help=False)
    parser.add_argument("--image_path", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "images", "text_data", "ic13"))
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join(DEFAULT_RECON_ROOT, "magvit2_imgnet_f16", "text_data", "ic13"),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="magvit2_imgnet_f16",
        choices=sorted(CONFIG_PATHS),
        help="Open-MAGVIT2/IBQ model variant",
    )
    parser.add_argument("--seed_voken_path", type=str, default=os.path.join(SCRIPT_DIR, "SEED-Voken"))
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


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    image = Image.fromarray(x)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def load_vqgan_new(config, model_cls, ckpt_path):
    model = model_cls(**config.model.init_args)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.eval()


def load_model(args):
    seed_voken_path = os.path.abspath(args.seed_voken_path)
    ckpt_path = os.path.abspath(args.ckpt_path or DEFAULT_CKPTS[args.model_name])
    config_file = os.path.join(seed_voken_path, CONFIG_PATHS[args.model_name])

    require_path(seed_voken_path, "SEED-Voken code directory")
    require_path(config_file, "Open-MAGVIT2 config")
    require_path(ckpt_path, "Open-MAGVIT2 checkpoint")

    sys.path.insert(0, seed_voken_path)
    from omegaconf import OmegaConf
    from src.IBQ.models.ibqgan import IBQ
    from src.Open_MAGVIT2.models.lfqgan import VQModel
    from src.Open_MAGVIT2.models.lfqgan_pretrain import VQModel as VQModelPretrain

    model_type = "Open-MAGVIT2" if "magvit" in args.model_name else "IBQ"
    if "pretrain" in args.model_name:
        model_type = "Open-MAGVIT2-pretrain"

    model_classes = {
        "Open-MAGVIT2-pretrain": VQModelPretrain,
        "Open-MAGVIT2": VQModel,
        "IBQ": IBQ,
    }
    configs = OmegaConf.load(config_file)
    model = load_vqgan_new(configs, model_classes[model_type], ckpt_path).to(DEVICE)
    return model, model_type


def encode_decode(model, model_type, input_images):
    if model_type in ["Open-MAGVIT2", "Open-MAGVIT2-pretrain"]:
        quant, _, _, _ = model.encode(input_images)
    elif model_type == "IBQ":
        quant, _, _ = model.encode(input_images)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model.decode(quant)


def main(args):
    model, model_type = load_model(args)

    image_save_pth = f"{args.save_path}_{args.padding_size}"
    os.makedirs(image_save_pth, exist_ok=True)

    all_datas = sorted(os.listdir(args.image_path))
    chunked_filenames = np.array_split(all_datas, args.num_chunks)
    subset = chunked_filenames[args.chunk_idx].tolist()
    chunk_inputs = split_list(subset, args.batch_size)

    for chunk in tqdm(chunk_inputs):
        image_path = os.path.join(args.image_path, chunk[0])
        original_img = Image.open(image_path).convert("RGB")
        padded_img, meta = smart_padding(original_img, (args.padding_size, args.padding_size))

        padded_img = np.array(padded_img)
        padded_img = padded_img / 127.5 - 1.0
        input_images = T.ToTensor()(padded_img).unsqueeze(0).to(device=DEVICE, dtype=torch.float)

        with torch.no_grad():
            if getattr(model, "use_ema", False):
                with model.ema_scope():
                    reconstructed_images = encode_decode(model, model_type, input_images)
            else:
                reconstructed_images = encode_decode(model, model_type, input_images)

        reconstructed_image = custom_to_pil(reconstructed_images[0])
        final_img = restore_original(reconstructed_image, meta)
        final_img.save(os.path.join(image_save_pth, chunk[0]))

    print(args.chunk_idx, " is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("image path check script", parents=[get_args_parser()])
    main(parser.parse_args())
