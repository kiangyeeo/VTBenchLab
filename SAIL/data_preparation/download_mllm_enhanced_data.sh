#!/bin/bash

# download the data from the url
echo "Downloading yfcc15m_3long_3short_1raw_captions_url.csv"   
wget https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions/resolve/main/yfcc15m_3long_3short_1raw_captions_url.csv
echo "Downloading cc3m_3long_3short_1raw_captions_url.csv"
wget https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions/resolve/main/cc3m_3long_3short_1raw_captions_url.csv
echo "Downloading cc12m_3long_3short_1raw_captions_url.csv"
wget https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions/resolve/main/cc12m_3long_3short_1raw_captions_url.csv

# process the data adding image paths for each image text pair
echo "Processing yfcc15m_3long_3short_1raw_captions_url.csv"
python adding_paths_to_data.py --input_csv_file yfcc15m_3long_3short_1raw_captions_url.csv
echo "Processing cc3m_3long_3short_1raw_captions_url.csv"
python adding_paths_to_data.py --input_csv_file cc3m_3long_3short_1raw_captions_url.csv
echo "Processing cc12m_3long_3short_1raw_captions_url.csv"
python adding_paths_to_data.py --input_csv_file cc12m_3long_3short_1raw_captions_url.csv

# download the images
echo "Downloading images for yfcc15m_3long_3short_1raw_captions_url.csv"
python download_images.py --csv_path yfcc15m_3long_3short_1raw_captions_url.csv --data_name yfcc15m --num_processes 64 --chunk_size 500
echo "Downloading images for cc3m_3long_3short_1raw_captions_url.csv"
python download_images.py --csv_path cc3m_3long_3short_1raw_captions_url.csv --data_name cc3m --num_processes 64 --chunk_size 500
echo "Downloading images for cc12m_3long_3short_1raw_captions_url.csv"
python download_images.py --csv_path cc12m_3long_3short_1raw_captions_url.csv --data_name cc12m --num_processes 64 --chunk_size 500

# filter the images that are not exist or corrupted
echo "Filtering images for yfcc15m_3long_3short_1raw_captions_url.csv"
python filter.py --input_csv_file yfcc15m_3long_3short_1raw_captions_url.csv --image_base_path ./ --chunk_size 5000
echo "Filtering images for cc3m_3long_3short_1raw_captions_url.csv"
python filter.py --input_csv_file cc3m_3long_3short_1raw_captions_url.csv --image_base_path ./ --chunk_size 5000
echo "Filtering images for cc12m_3long_3short_1raw_captions_url.csv"
python filter.py --input_csv_file cc12m_3long_3short_1raw_captions_url.csv --image_base_path ./ --chunk_size 5000