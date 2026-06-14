import torch
import os
import json
import logging
import csv
import time
from tqdm import tqdm
from model import SentenceEmbedding, ImageEmbedding
from train.logger import setup_logging
from data.data_config import DATADIR
from data.utils import load_data
from data.image_dataset import create_image_dataloader
import warnings
setup_logging(log_file=None, level=logging.INFO)
warnings.filterwarnings("ignore", message="Corrupt EXIF data")
warnings.filterwarnings("ignore", message="Palette images with Transparency")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Data type')
    parser.add_argument('--vision_model_name', type=str, required=True, help='Model name')
    parser.add_argument('--text_model_name', type=str, required=True, help='Model name')
    parser.add_argument('--resume', action='store_true', help='Resume from existing embeddings')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for data processing')
    parser.add_argument('--end_index', type=int, default=None, help='End index for data processing')
    parser.add_argument('--domain', type=str, choices=['text', 'image'], required=True, help='Domain to encode')
    parser.add_argument('--source_caption', type=str, choices=['raw_caption', 'shortIB_captions', 'longIB_captions', 'shortSV_captions', 'longSV_captions', 'shortLLA_captions', 'longLLA_captions','caption'], required=True, help='Source caption')
    parser.add_argument('--save_name', type=str, default=None, help='Save name')
    parser.add_argument('--batch_size', type=int, default=32, help='Save batch size')
    parser.add_argument('--agg_mode', type=str, default='concat', help='Aggregation mode')
    parser.add_argument('--throughput', action='store_true', help='Calculate throughput')
    return parser.parse_args()

def process_batch(data, start_index, batch_size, output_dir, encode_function, resume, throughput=False):
    idx = start_index // batch_size
    total_time = 0
    total_samples = 0
    
    for batch_idx in tqdm(range(0, len(data), batch_size)):  # Use batch_size for both encoding and saving
        # Save embeddings in chunks of batch_size with torch.fp16
        
        output_path = os.path.join(output_dir, f'{idx}.pt')
        if resume and os.path.exists(output_path):
            idx += 1
            continue

        batch_data = data[batch_idx:batch_idx + batch_size]  # Use batch_size for encoding
        
        # Measure encoding time
        start_time = time.time()
        with torch.cuda.amp.autocast():  # Enable automatic mixed precision
            with torch.no_grad():
                batch_embeddings = encode_function(batch_data).cpu()  # Encode the entire batch
        end_time = time.time()
        
        # Update timing stats
        batch_time = end_time - start_time
        total_time += batch_time
        total_samples += len(batch_data)
        
        # Calculate and log throughput
        current_throughput = len(batch_data) / batch_time
        avg_throughput = total_samples / total_time
        

        if throughput:
            logging.info(f"Batch {idx} throughput: {current_throughput:.2f} samples/sec, "
                        f"Average throughput: {avg_throughput:.2f} samples/sec")
        else:
            batch_embeddings = batch_embeddings.half()  # Convert to half precision
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(batch_embeddings, output_path)
        idx += 1
    
    # Log final stats
    if total_samples > 0:
        final_throughput = total_samples / total_time
        logging.info(f"Final average throughput: {final_throughput:.2f} samples/sec")
        logging.info(f"Total processing time: {total_time:.2f} seconds")
        logging.info(f"Total samples processed: {total_samples}")


def process_batch_image(data_loader, model, start_index, batch_size, output_dir, resume, throughput=False):
    idx = start_index // batch_size
    total_time = 0
    total_samples = 0
    
    for batch_data in tqdm(data_loader):  # Use batch_size for both encoding and saving
        # Save embeddings in chunks of batch_size with torch.fp16
        output_path = os.path.join(output_dir, f'{idx}.pt')
        if resume and os.path.exists(output_path):
            print(f'{output_path} already exists, skipping...')
            idx += 1
            continue
        
        # Measure encoding time
        start_time = time.time()
        batch_data = batch_data.to('cuda')
        with torch.cuda.amp.autocast():  # Enable automatic mixed precision
            with torch.no_grad():
                batch_embeddings = model(batch_data).cpu()  # Encode the entire batch
        end_time = time.time()
        
        # Update timing stats
        batch_time = end_time - start_time
        total_time += batch_time
        total_samples += batch_size
        
        # Calculate and log throughput
        current_throughput = batch_size / batch_time
        avg_throughput = total_samples / total_time
        
        logging.info(f"Batch {idx} throughput: {current_throughput:.2f} samples/sec, "
                    f"Average throughput: {avg_throughput:.2f} samples/sec")
        
        if not throughput:
            logging.info(f"Batch {idx} processing time: {batch_time:.2f} seconds")
            batch_embeddings = batch_embeddings.half()  # Convert to half precision
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(batch_embeddings, output_path)
        idx += 1
    
    # Log final stats
    if total_samples > 0:
        final_throughput = total_samples / total_time
        logging.info(f"Final average throughput: {final_throughput:.2f} samples/sec")
        logging.info(f"Total processing time: {total_time:.2f} seconds")
        logging.info(f"Total samples processed: {total_samples}")

@torch.no_grad()
def encode_text(args, sentences, start_index):
    model_name = args.text_model_name.split('/')[-1]
    if args.save_name:
        output_dir = os.path.join('./data/tensor_data/text_embedding', model_name, args.data +'_'+ args.source_caption + '_' + args.save_name)
    else:
        output_dir = os.path.join('./data/tensor_data/text_embedding', model_name, args.data +'_'+ args.source_caption)
    print(f"Output directory: {output_dir}")
    if not args.resume and os.path.exists(output_dir):
        exit()
    model = SentenceEmbedding(args.text_model_name)
    model = model.half().to('cuda')  # Move model to GPU and convert to FP16
    model.eval()
    process_batch(sentences, start_index, args.batch_size, output_dir, model.get_sentence_embeddings, args.resume, args.throughput)

# @torch.no_grad()
# def encode_image(args, image_paths, start_index):
#     model_name = args.vision_model_name.split('/')[-1]
#     output_dir = os.path.join('./data/tensor_data/image_embedding', model_name, args.data + '_' + args.agg_mode)
#     if not args.resume and os.path.exists(output_dir):
#         logging.info(f'{output_dir} already exists, skipping...')
#         exit()
#     model = ImageEmbedding(args.vision_model_name, agg_mode=args.agg_mode)
#     model = model.half().to('cuda')  # Move model to GPU and convert to FP16
#     model.eval()
#     image_data_loader = create_image_dataloader(image_paths, model.image_processor, batch_size=args.batch_size, num_workers=4, shuffle=False)
#     process_batch_image(image_data_loader, model, start_index, args.batch_size, output_dir, args.resume, args.throughput)

@torch.no_grad()
def encode_image(args, images, start_index):
    model_name = args.vision_model_name.split('/')[-1]
    output_dir = os.path.join('./data/tensor_data/image_embedding', model_name, args.data + '_' + args.agg_mode)
    if not args.resume and os.path.exists(output_dir):
        logging.info(f'{output_dir} already exists, skipping...')
        exit()
    model = ImageEmbedding(args.vision_model_name, agg_mode=args.agg_mode)
    model = model.to('cuda')  # Move model to GPU
    model.eval()
    process_batch(images, start_index, args.batch_size, output_dir, model.get_visual_embeddings_from_directory, args.resume)


def main():
    args = parse_args()
    sentences, image_paths = load_data(DATADIR[args.data], args.source_caption, args.domain)
    start_index = args.start_index
    end_index = args.end_index if args.end_index else max(len(sentences), len(image_paths))
    logging.info(f"Start index: {start_index}, End index: {end_index}")
    logging.info(f"Number of sentences: {len(sentences)}")
    logging.info(f"Number of image_paths: {len(image_paths)}")
    if args.throughput:
        logging.warning("Only measure throughput, not saving embeddings")
    # assert begin and end 
    if args.domain == 'text':
        logging.info(f'Encoding text data {args.data} with model {args.text_model_name} of batch size {args.batch_size}...')
        sentences = sentences[start_index:end_index]
        logging.info(f"First 5 items of sentences: {sentences[:5]}")
        encode_text(args, sentences, start_index)
    elif args.domain == 'image':
        logging.info(f'Encoding image data {args.data} with model {args.vision_model_name} of batch size {args.batch_size}...')
        image_paths = image_paths[start_index:end_index]
        logging.info(f"First 5 items of image_paths paths: {image_paths[:5]}")
        encode_image(args, image_paths, start_index)

if __name__ == "__main__":
    main()