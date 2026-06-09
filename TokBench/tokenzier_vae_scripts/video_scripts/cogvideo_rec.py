import argparse
import math
import os

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from resize_rec import resize_padding_images, restore_images


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_MODEL_ZOO = os.path.join(REPO_ROOT, "tokenizer_modelzoo")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "tokbench_data")
DEFAULT_RECON_ROOT = os.path.join(REPO_ROOT, "video_reconstruction_results")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_ZOO, "CogVideoX1.5-5B", "vae")


def get_args_parser():
    parser = argparse.ArgumentParser("CogVideoX VAE reconstruction", add_help=False)
    parser.add_argument("--video_path", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "videos", "text_data", "ds"))
    parser.add_argument("--save_path", type=str, default=os.path.join(DEFAULT_RECON_ROOT, "cogvideox_4x8x8", "text_data", "ds"))
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--short_size", type=int, default=480)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser


def split_list(input_list, chunk_size):
    return [input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def require_path(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_model(args, device, dtype):
    from diffusers import AutoencoderKLCogVideoX

    model_path = os.path.abspath(args.model_path)
    require_path(model_path, "CogVideoX VAE directory")
    model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.enable_slicing()
    model.enable_tiling()
    return model.eval()


def main(args):
    import imageio

    image_save_pth = f"{args.save_path}_{args.short_size}"
    os.makedirs(image_save_pth, exist_ok=True)

    all_datas = sorted(os.listdir(args.video_path))
    chunked_filenames = np.array_split(all_datas, args.num_chunks)
    subset = chunked_filenames[args.chunk_idx].tolist()
    chunk_inputs = split_list(subset, args.batch_size)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model = load_model(args, device, dtype)

    for chunk in tqdm(chunk_inputs):
        video_path = os.path.join(args.video_path, chunk[0])
        video_reader = imageio.get_reader(video_path, "ffmpeg")
        video_fps = video_reader.get_meta_data()["fps"]
        frames = [frame for frame in video_reader]
        video_reader.close()

        ori_length = len(frames)
        padding_length = math.ceil((ori_length - 1) / 8) * 8 + 1
        num_pad_frame = padding_length - ori_length
        frames.extend([frames[-1]] * num_pad_frame)

        input_frames, operation_metas = resize_padding_images(frames, short_size=args.short_size)
        frame_tensors = [transforms.ToTensor()(frame) for frame in input_frames]
        frames_tensor = torch.stack(frame_tensors).to(device).permute(1, 0, 2, 3).unsqueeze(0)
        frames_tensor = (frames_tensor * 2.0 - 1.0).to(dtype)

        with torch.no_grad():
            encoded_frames = model.encode(frames_tensor)[0].sample()
            decoded_frames = model.decode(encoded_frames).sample

        decoded_frames = (decoded_frames.to(dtype=torch.float32) / 2.0 + 0.5).clamp(0, 1)
        decoded_frames = decoded_frames[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        decoded_frames = (decoded_frames * 255).astype(np.uint8)
        output_frames = [frame for frame in decoded_frames]
        output_frames = restore_images(output_frames, operation_metas)[:ori_length]

        frames = np.stack(output_frames, axis=0)
        writer = imageio.get_writer(os.path.join(image_save_pth, chunk[0]), fps=video_fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    print(args.chunk_idx, " is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("video path check script", parents=[get_args_parser()])
    main(parser.parse_args())
