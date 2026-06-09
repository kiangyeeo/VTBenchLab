import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

from resize_rec import resize_padding_images, restore_images


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_MODEL_ZOO = os.path.join(REPO_ROOT, "tokenizer_modelzoo")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "tokbench_data")
DEFAULT_RECON_ROOT = os.path.join(REPO_ROOT, "video_reconstruction_results")

MODEL_NAMES = {
    "CosmosCV4x8x8": "Cosmos-0.1-Tokenizer-CV4x8x8",
    "CosmosCV8x16x16": "Cosmos-0.1-Tokenizer-CV8x16x16",
    "CosmosDV4x8x8": "Cosmos-0.1-Tokenizer-DV4x8x8",
    "CosmosDV8x16x16": "Cosmos-0.1-Tokenizer-DV8x16x16",
}


def get_args_parser():
    parser = argparse.ArgumentParser("Cosmos tokenizer reconstruction", add_help=False)
    parser.add_argument("--video_path", type=str, default=os.path.join(DEFAULT_DATA_ROOT, "videos", "text_data", "ds"))
    parser.add_argument("--save_path", type=str, default=os.path.join(DEFAULT_RECON_ROOT, "CosmosDV4x8x8", "text_data", "ds"))
    parser.add_argument("--model", type=str, choices=sorted(MODEL_NAMES), default="CosmosDV4x8x8")
    parser.add_argument("--cosmos_path", type=str, default=os.path.join(SCRIPT_DIR, "Cosmos-Tokenizer"))
    parser.add_argument("--model_root", type=str, default=DEFAULT_MODEL_ZOO)
    parser.add_argument("--short_size", type=int, default=480)
    parser.add_argument("--temporal_window", type=int, default=49)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser


def split_list(input_list, chunk_size):
    return [input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def require_path(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_tokenizer(args):
    cosmos_path = os.path.abspath(args.cosmos_path)
    model_name = MODEL_NAMES[args.model]
    model_dir = os.path.join(os.path.abspath(args.model_root), model_name)
    encoder_ckpt = os.path.join(model_dir, "encoder.jit")
    decoder_ckpt = os.path.join(model_dir, "decoder.jit")

    require_path(os.path.join(cosmos_path, "cosmos_tokenizer", "video_lib.py"), "Cosmos-Tokenizer code")
    require_path(encoder_ckpt, "Cosmos encoder checkpoint")
    require_path(decoder_ckpt, "Cosmos decoder checkpoint")

    sys.path.insert(0, cosmos_path)
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer

    return CausalVideoTokenizer(
        checkpoint_enc=encoder_ckpt,
        checkpoint_dec=decoder_ckpt,
        device="cuda",
        dtype="bfloat16",
    )


def main(args):
    import imageio

    image_save_pth = f"{args.save_path}_{args.short_size}"
    os.makedirs(image_save_pth, exist_ok=True)

    all_datas = sorted(os.listdir(args.video_path))
    chunked_filenames = np.array_split(all_datas, args.num_chunks)
    subset = chunked_filenames[args.chunk_idx].tolist()
    chunk_inputs = split_list(subset, args.batch_size)

    tokenizer = load_tokenizer(args)

    for chunk in tqdm(chunk_inputs):
        video_path = os.path.join(args.video_path, chunk[0])
        video_reader = imageio.get_reader(video_path, "ffmpeg")
        video_fps = video_reader.get_meta_data()["fps"]
        frames = [frame[:, :, ::-1] for frame in video_reader]  # Cosmos expects BGR input.
        video_reader.close()

        input_frames, operation_metas = resize_padding_images(frames, short_size=args.short_size)
        input_video = np.stack(input_frames, axis=0)
        batched_input_video = np.expand_dims(input_video, axis=0)

        batched_output_video = tokenizer(batched_input_video, temporal_window=args.temporal_window)
        output_video = batched_output_video[0]
        output_frames = [frame[:, :, ::-1] for frame in output_video]
        output_frames = restore_images(output_frames, operation_metas)

        frames = np.stack(output_frames, axis=0).astype(np.uint8)
        writer = imageio.get_writer(os.path.join(image_save_pth, chunk[0]), fps=video_fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    print(args.chunk_idx, " is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("video path check script", parents=[get_args_parser()])
    main(parser.parse_args())
