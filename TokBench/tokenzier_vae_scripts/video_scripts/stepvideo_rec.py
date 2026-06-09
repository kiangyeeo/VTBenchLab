import argparse
import math
import os
import sys

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
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_ZOO, "stepvideo-t2v", "vae", "vae_v2.safetensors")


def get_args_parser():
    parser = argparse.ArgumentParser("Step-Video VAE reconstruction", add_help=False)
    parser.add_argument("--video_path", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "videos", "text_data", "ds"))
    parser.add_argument("--save_path", type=str, default=os.path.join(DEFAULT_RECON_ROOT, "stepvideo_8x16x16", "text_data", "ds"))
    parser.add_argument("--stepvideo_path", type=str, default=os.path.join(SCRIPT_DIR, "Step-Video-T2V"))
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


def load_model(args, device):
    stepvideo_path = os.path.abspath(args.stepvideo_path)
    model_path = os.path.abspath(args.model_path)
    require_path(os.path.join(stepvideo_path, "stepvideo", "vae", "vae.py"), "Step-Video-T2V code")
    require_path(model_path, "Step-Video VAE checkpoint")

    sys.path.insert(0, stepvideo_path)
    from stepvideo.vae.vae import AutoencoderKL

    model = AutoencoderKL(
        z_channels=64,
        model_path=model_path,
        version=2,
    )
    return model.to(torch.bfloat16).to(device).eval()


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
    model = load_model(args, device)
    print("Initialized vae...")

    for chunk in tqdm(chunk_inputs):
        video_path = os.path.join(args.video_path, chunk[0])
        video_reader = imageio.get_reader(video_path, "ffmpeg")
        video_fps = video_reader.get_meta_data()["fps"]
        allframes = [frame for frame in video_reader]
        sizes = video_reader.get_meta_data()["size"]
        short_size = 400 if max(sizes) / min(sizes) > 1.8 and args.short_size == 480 else args.short_size
        video_reader.close()

        ori_length = len(allframes)
        padding_length = math.ceil((ori_length - 1) / 17) * 17
        num_pad_frame = padding_length - ori_length
        allframes.extend([allframes[-1]] * num_pad_frame)

        num_video_chunks = math.ceil(len(allframes) / 170)
        all_output_frames = []
        while num_video_chunks > 0:
            num_video_chunks -= 1
            frames = allframes[:170]
            allframes = allframes[170:]
            input_frames, operation_metas = resize_padding_images(frames, short_size=short_size)

            frame_tensors = [transforms.ToTensor()(frame) for frame in input_frames]
            frames_tensor = torch.stack(frame_tensors).to(device).unsqueeze(0).to(dtype)

            with torch.no_grad():
                latent = model.encode(frames_tensor)
                decoded_frames = model.decode(latent)

            decoded_frames = decoded_frames.cpu().to(dtype=torch.float32)
            decoded_frames = decoded_frames[0].squeeze(0).permute(0, 2, 3, 1).numpy()
            decoded_frames = np.clip(decoded_frames, 0, 1) * 255
            decoded_frames = decoded_frames.astype(np.uint8)
            output_frames = [frame for frame in decoded_frames]
            all_output_frames.extend(restore_images(output_frames, operation_metas))

        assert len(allframes) == 0
        all_output_frames = all_output_frames[:ori_length]

        frames = np.stack(all_output_frames, axis=0)
        writer = imageio.get_writer(os.path.join(image_save_pth, chunk[0]), fps=video_fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    print(args.chunk_idx, " is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("video path check script", parents=[get_args_parser()])
    main(parser.parse_args())
